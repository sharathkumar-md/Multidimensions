"""
Auth middleware — Keycloak JWT Bearer validation.

When API_AUTH_ENABLED=false (default in dev), returns a stub dev user so all
routes work without a running Keycloak instance.

In production (API_AUTH_ENABLED=true), fetches the JWKS from Keycloak and
verifies every Bearer token's signature, expiry, and audience.
"""
from __future__ import annotations

import time
from functools import lru_cache
from typing import Optional

import httpx
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError, jwt
from loguru import logger

from api_config import api_settings
from models import UserInfo

_bearer = HTTPBearer(auto_error=False)

# ── JWKS cache (refreshed at most once per 5 minutes) ────────────────────────

_JWKS_CACHE: dict = {}
_JWKS_FETCHED_AT: float = 0.0
_JWKS_TTL = 300  # seconds


def _jwks_url() -> str:
    return (
        f"{api_settings.keycloak_server_url}/realms/"
        f"{api_settings.keycloak_realm}/protocol/openid-connect/certs"
    )


def _get_jwks() -> dict:
    global _JWKS_CACHE, _JWKS_FETCHED_AT
    now = time.monotonic()
    if now - _JWKS_FETCHED_AT < _JWKS_TTL and _JWKS_CACHE:
        return _JWKS_CACHE
    try:
        resp = httpx.get(_jwks_url(), timeout=5.0)
        resp.raise_for_status()
        _JWKS_CACHE = resp.json()
        _JWKS_FETCHED_AT = now
        logger.debug("JWKS refreshed from Keycloak")
        return _JWKS_CACHE
    except Exception as e:
        logger.error(f"Failed to fetch JWKS: {e}")
        if _JWKS_CACHE:
            logger.warning("Using stale JWKS cache")
            return _JWKS_CACHE
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Auth service unavailable — cannot fetch JWKS.",
        )


def _decode_token(token: str) -> dict:
    """Decode and verify a Keycloak JWT. Returns the claims dict."""
    jwks = _get_jwks()
    try:
        claims = jwt.decode(
            token,
            jwks,
            algorithms=["RS256"],
            audience=api_settings.keycloak_client_id,
            options={"verify_at_hash": False},
        )
        return claims
    except JWTError as e:
        logger.warning(f"JWT validation failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid or expired token: {e}",
            headers={"WWW-Authenticate": "Bearer"},
        )


def _claims_to_user(claims: dict) -> UserInfo:
    """Extract UserInfo from Keycloak token claims."""
    realm_access: dict = claims.get("realm_access", {})
    resource_access: dict = claims.get("resource_access", {})
    client_roles: list[str] = (
        resource_access.get(api_settings.keycloak_client_id, {}).get("roles", [])
    )
    roles = list(set(realm_access.get("roles", []) + client_roles))
    return UserInfo(
        sub=claims["sub"],
        email=claims.get("email", ""),
        name=claims.get("name", claims.get("preferred_username", "")),
        roles=roles,
    )


# ── FastAPI dependency ────────────────────────────────────────────────────────

async def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(_bearer),
) -> UserInfo:
    """
    FastAPI dependency — resolves to the authenticated UserInfo.
    """
    if not api_settings.auth_enabled:
        return UserInfo(
            sub="local-dev",
            email="dev@local",
            name="Local Dev",
            roles=["admin", "sales"],
        )

    if credentials is None or credentials.scheme.lower() != "bearer":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Bearer token required.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    claims = _decode_token(credentials.credentials)
    user = _claims_to_user(claims)
    logger.debug(f"Authenticated: {user.email} roles={user.roles}")
    return user


async def require_admin(user: UserInfo = Depends(get_current_user)) -> UserInfo:
    """Dependency that additionally enforces admin role."""
    if not user.is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin role required.",
        )
    return user
