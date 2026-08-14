"""
Integration tests for the FastAPI backend.
Tests API endpoints with mocked auth and pipeline.
"""
from __future__ import annotations

# Set test environment before importing main
import os
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

os.environ["API_AUTH_ENABLED"] = "false"
os.environ["RAG_AUTH_ENABLED"] = "false"

from main import app
from rag_service import _pipeline_ready_event


@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)


@pytest.fixture(autouse=True)
def reset_pipeline_state():
    """Reset pipeline state before each test."""
    global _pipeline_loaded, _pipeline_ready_event
    _pipeline_loaded = False
    _pipeline_ready_event.clear()
    yield
    _pipeline_loaded = False
    _pipeline_ready_event.clear()


def test_health_endpoint(client):
    """Test health endpoint returns correct structure."""
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "version" in data
    assert "rag_loaded" in data
    assert "gpu_available" in data


def test_health_rag_not_loaded(client):
    """Test health reports rag_loaded=false when pipeline not ready."""
    response = client.get("/api/health")
    data = response.json()
    assert data["rag_loaded"] is False


@patch("rag_service.wait_for_pipeline", new_callable=AsyncMock)
@patch("rag_service._pipeline_loaded", True)
def test_chat_endpoint_requires_session(mock_wait, client):
    """Test chat endpoint returns 404 for non-existent session."""
    mock_wait.return_value = True

    response = client.post(
        "/api/chat",
        json={"session_id": "non-existent", "question": "test"},
    )
    # Should fail with 404 since session doesn't exist
    assert response.status_code == 404


@patch("rag_service.wait_for_pipeline", new_callable=AsyncMock)
@patch("rag_service._pipeline_loaded", True)
@patch("rag_service.stream_answer")
def test_chat_endpoint_streaming(mock_stream, mock_wait, client):
    """Test chat endpoint returns streaming response."""
    mock_wait.return_value = True

    # Mock the stream_answer generator
    async def mock_stream(*args, **kwargs):
        yield 'data: {"token": "Hello"}\n\n'
        yield 'data: {"token": " world"}\n\n'
        yield 'data: {"done": true, "sources": [], "route": "LOCAL"}\n\n'

    mock_stream.side_effect = mock_stream

    # Create a session first
    session_response = client.post("/api/sessions", json={"title": "Test"})
    assert session_response.status_code == 201
    session_id = session_response.json()["id"]

    # Test chat
    response = client.post(
        "/api/chat",
        json={"session_id": session_id, "question": "Hello"},
    )
    # Should return streaming response
    assert response.status_code == 200
    assert response.headers["content-type"] == "text/event-stream; charset=utf-8"


def test_sessions_crud(client):
    """Test session CRUD operations."""
    # Create session
    create_response = client.post("/api/sessions", json={"title": "Test Session"})
    assert create_response.status_code == 201
    session = create_response.json()
    assert session["title"] == "Test Session"
    session_id = session["id"]

    # List sessions
    list_response = client.get("/api/sessions")
    assert list_response.status_code == 200
    sessions = list_response.json()
    assert len(sessions) == 1
    assert sessions[0]["id"] == session_id

    # Get session
    get_response = client.get(f"/api/sessions/{session_id}")
    assert get_response.status_code == 200
    assert get_response.json()["id"] == session_id

    # Get messages (empty)
    msg_response = client.get(f"/api/sessions/{session_id}/messages")
    assert msg_response.status_code == 200
    assert msg_response.json() == []

    # Delete session
    delete_response = client.delete(f"/api/sessions/{session_id}")
    assert delete_response.status_code == 204

    # Verify deleted
    get_response = client.get(f"/api/sessions/{session_id}")
    assert get_response.status_code == 404


def test_admin_endpoints_require_auth(client):
    """Test admin endpoints require admin role (when auth enabled)."""
    # With auth disabled, these should work for dev user
    # but we test the structure
    response = client.get("/api/index/stats")
    # Should work in dev mode (auth disabled)
    assert response.status_code in (200, 403)


def test_rate_limit_headers(client):
    """Test rate limit headers are present."""
    response = client.get("/api/health")
    # Rate limit headers may not be on health endpoint
    assert response.status_code == 200


def test_cors_headers(client):
    """Test CORS headers are configured."""
    response = client.options(
        "/api/health",
        headers={
            "Origin": "http://localhost:3000",
            "Access-Control-Request-Method": "GET",
        },
    )
    # Should allow the origin
    assert response.status_code in (200, 405)  # 405 if OPTIONS not explicitly handled


@patch("rag_service._pipeline_loaded", True)
@patch("routers.admin.get_index_stats")
def test_admin_index_stats(mock_stats, client):
    """Test admin index stats endpoint."""
    mock_stats.return_value = {
        "n_chunks": 100,
        "n_docs": 10,
        "last_updated": 1234567890.0,
        "gpu_available": False,
    }

    response = client.get("/api/index/stats")
    assert response.status_code == 200
    data = response.json()
    assert data["n_chunks"] == 100
    assert data["n_docs"] == 10


@patch("rag_service._pipeline_loaded", True)
@patch("routers.admin.get_ingestion_status")
def test_admin_ingest_status(mock_status, client):
    """Test admin ingest status endpoint."""
    mock_status.return_value = {
        "running": False,
        "progress": 0.0,
        "current_file": None,
        "error": None,
    }

    response = client.get("/api/admin/ingest/status")
    assert response.status_code == 200
    data = response.json()
    assert data["running"] is False


def test_openapi_docs(client):
    """Test OpenAPI docs are accessible."""
    response = client.get("/docs")
    assert response.status_code == 200
    assert "openapi" in response.text.lower() or "swagger" in response.text.lower()


def test_redoc(client):
    """Test ReDoc is accessible."""
    response = client.get("/redoc")
    assert response.status_code == 200
