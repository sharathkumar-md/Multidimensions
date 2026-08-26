"""
Auth middleware - Google OAuth JWT Bearer validation.
"""
from __future__ import annotations

import threading
import time
from typing import Optional

import httpx
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError, jwt
from loguru import logger

from api_config import api_settings
from models import UserInfo

_bearer = HTTPBearer(auto_error=False)

_JWKS_CACHE: dict = {}
_JWKS_FETCHED_AT: float = 0.0
_JWKS_TTL = 3600  # seconds
_JWKS_LOCK = threading.Lock()

def _get_jwks() -> dict:
    global _JWKS_CACHE, _JWKS_FETCHED_AT
    now = time.monotonic()
    if now - _JWKS_FETCHED_AT < _JWKS_TTL and _JWKS_CACHE:
        return _JWKS_CACHE
    
    with _JWKS_LOCK:
        now = time.monotonic()
        if now - _JWKS_FETCHED_AT < _JWKS_TTL and _JWKS_CACHE:
            return _JWKS_CACHE
        
        try:
            resp = httpx.get("https://www.googleapis.com/oauth2/v3/certs", timeout=5.0)
            resp.raise_for_status()
            _JWKS_CACHE = resp.json()
            _JWKS_FETCHED_AT = now
            return _JWKS_CACHE
        except Exception as e:
            logger.error(f"Failed to fetch Google JWKS: {e}")
            if _JWKS_CACHE:
                return _JWKS_CACHE
            raise HTTPException(status_code=503, detail="Auth unavailable")

def _decode_token(token: str) -> dict:
    jwks = _get_jwks()
    try:
        # Use keycloak_client_id field to store GOOGLE_CLIENT_ID
        client_id = api_settings.keycloak_client_id 
        claims = jwt.decode(
            token,
            jwks,
            algorithms=["RS256"],
            audience=client_id,
            options={"verify_at_hash": False},
        )
        return claims
    except JWTError as e:
        logger.warning(f"JWT validation failed: {e}")
        raise HTTPException(
            status_code=401,
            detail="Invalid Google token",
            headers={"WWW-Authenticate": "Bearer"},
        )

def _claims_to_user(claims: dict) -> UserInfo:
    return UserInfo(
        sub=claims["sub"],
        email=claims.get("email", ""),
        name=claims.get("name", ""),
        roles=["admin", "sales"],
    )

async def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(_bearer),
) -> UserInfo:
    if not api_settings.auth_enabled:
        return UserInfo(
            sub="local-dev", email="dev@local", name="Local Dev", roles=["admin", "sales"]
        )

    if credentials is None or credentials.scheme.lower() != "bearer":
        raise HTTPException(
            status_code=401, detail="Bearer token required", headers={"WWW-Authenticate": "Bearer"}
        )

    claims = _decode_token(credentials.credentials)
    user = _claims_to_user(claims)
    return user

async def require_admin(user: UserInfo = Depends(get_current_user)) -> UserInfo:
    return user
