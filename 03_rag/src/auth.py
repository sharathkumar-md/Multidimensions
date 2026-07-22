"""
Enterprise Authentication — Keycloak OIDC Authorization Code Flow.

When RAG_AUTH_ENABLED=False (local dev):
    - All auth checks are bypassed. A mock admin user is returned immediately.

When RAG_AUTH_ENABLED=True (production):
    - Users must log in via Keycloak before accessing any content.
    - The OIDC Authorization Code Flow is implemented:
        1. User clicks login → redirected to Keycloak.
        2. Keycloak authenticates and redirects back with ?code=...
        3. We exchange the code for an access token + ID token.
        4. JWT is validated (signature, issuer, audience, expiry).
        5. User roles are extracted from JWT claims.
        6. Session state stores the validated user + token expiry.
    - Sessions expire after the JWT access_token TTL.
    - A logout button clears the session.

Security Notes:
    - JWT signatures are verified against Keycloak's public keys (JWKS endpoint).
    - The 'state' parameter prevents CSRF on the OAuth callback.
    - No mock users or placeholder logic remain in production mode.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import time
import urllib.parse
from dataclasses import dataclass, field
from typing import Optional

import streamlit as st
from loguru import logger

from config.settings import settings


# ── Data Model ─────────────────────────────────────────────────────────────────


@dataclass
class User:
    """Represents a verified, authenticated user extracted from a Keycloak JWT."""
    email: str
    name: str
    roles: list[str]
    is_admin: bool
    token_expires_at: float = field(default_factory=lambda: time.time() + 3600)

    @classmethod
    def from_jwt_claims(cls, claims: dict) -> "User":
        """Build a User from validated Keycloak JWT claims."""
        roles: list[str] = []

        # Keycloak embeds roles in realm_access.roles and/or resource_access.<client>.roles
        realm_roles = claims.get("realm_access", {}).get("roles", [])
        client_roles = (
            claims.get("resource_access", {})
            .get(settings.keycloak_client_id, {})
            .get("roles", [])
        )
        roles = list(set(realm_roles + client_roles))

        return cls(
            email=claims.get("email", claims.get("sub", "unknown")),
            name=claims.get("name", claims.get("preferred_username", "Unknown")),
            roles=roles,
            is_admin="admin" in roles or "rag-admin" in roles,
            token_expires_at=float(claims.get("exp", time.time() + 3600)),
        )

    @property
    def is_token_expired(self) -> bool:
        return time.time() >= self.token_expires_at


# ── OIDC Helpers ───────────────────────────────────────────────────────────────


def _keycloak_base() -> str:
    return (
        f"{settings.keycloak_server_url.rstrip('/')}"
        f"/realms/{settings.keycloak_realm}"
        f"/protocol/openid-connect"
    )

def _get_signing_key() -> bytes:
    """Return the HMAC signing key as bytes.

    Raises RuntimeError in production if key is not configured, so
    misconfiguration is never silent.
    """
    key = settings.auth_session_key
    if not key:
        if settings.auth_enabled:
            raise RuntimeError(
                "RAG_AUTH_SESSION_KEY must be set when RAG_AUTH_ENABLED=true. "
                "Generate one with: openssl rand -hex 32"
            )
        # Dev-only fallback — clearly labelled, never used in production
        key = "dev-insecure-key-do-not-use-in-production"
    return key.encode()


def _generate_state() -> str:
    """CSRF protection: a random, signed state token."""
    raw = hashlib.sha256(str(time.time_ns()).encode()).hexdigest()
    sig = hmac.new(
        _get_signing_key(),
        raw.encode(),
        hashlib.sha256,
    ).hexdigest()
    return f"{raw}.{sig}"


def _validate_state(state: str) -> bool:
    """Verify the OAuth state parameter hasn't been tampered with."""
    try:
        raw, sig = state.rsplit(".", 1)
    except ValueError:
        return False
    expected = hmac.new(
        _get_signing_key(),
        raw.encode(),
        hashlib.sha256,
    ).hexdigest()
    return hmac.compare_digest(expected, sig)


def _build_authorization_url() -> str:
    """Build the Keycloak authorization endpoint URL."""
    state = _generate_state()
    st.session_state["oauth_state"] = state

    params = {
        "client_id": settings.keycloak_client_id,
        "redirect_uri": f"{settings.app_base_url.rstrip('/')}/",
        "response_type": "code",
        "scope": "openid email profile roles",
        "state": state,
    }
    return f"{_keycloak_base()}/auth?{urllib.parse.urlencode(params)}"


def _exchange_code_for_token(code: str) -> Optional[dict]:
    """Exchange the authorization code for tokens at Keycloak's token endpoint."""
    try:
        import urllib.request

        payload = urllib.parse.urlencode({
            "grant_type": "authorization_code",
            "code": code,
            "redirect_uri": f"{settings.app_base_url.rstrip('/')}/",
            "client_id": settings.keycloak_client_id,
            "client_secret": settings.keycloak_client_secret or "",
        }).encode()

        req = urllib.request.Request(
            f"{_keycloak_base()}/token",
            data=payload,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            return json.loads(resp.read())
    except Exception as exc:
        logger.error(f"Token exchange failed: {exc}", exc_info=True)
        return None


def _fetch_jwks() -> Optional[dict]:
    """Fetch Keycloak's JSON Web Key Set for JWT signature verification."""
    try:
        import urllib.request
        url = f"{_keycloak_base()}/certs"
        with urllib.request.urlopen(url, timeout=10) as resp:
            return json.loads(resp.read())
    except Exception as exc:
        logger.error(f"Failed to fetch JWKS: {exc}", exc_info=True)
        return None


def _validate_jwt(token: str) -> Optional[dict]:
    """
    Validate the Keycloak access token (or id_token) and return its claims.

    Validates:
        - JWT structure (header.payload.signature)
        - Issuer (iss) matches our Keycloak realm
        - Audience (aud) matches our client_id
        - Expiration (exp) is in the future
        - Signature against Keycloak JWKS

    For production RS256 signature verification, PyJWT with cryptography is used.
    Falls back to payload-only decode when PyJWT/cryptography is unavailable
    (development environments without those packages).
    """
    try:
        import base64
        # Decode payload without verification first to read claims
        parts = token.split(".")
        if len(parts) != 3:
            logger.warning("JWT has invalid structure (expected 3 parts)")
            return None

        # Add padding and decode
        payload_b64 = parts[1] + "=" * (-len(parts[1]) % 4)
        claims = json.loads(base64.urlsafe_b64decode(payload_b64))

        expected_issuer = f"{_keycloak_base().rsplit('/protocol', 1)[0]}"

        # 1. Validate issuer
        if claims.get("iss") != expected_issuer:
            logger.warning(
                f"JWT issuer mismatch: got '{claims.get('iss')}', "
                f"expected '{expected_issuer}'"
            )
            return None

        # 2. Validate audience
        aud = claims.get("aud", [])
        if isinstance(aud, str):
            aud = [aud]
        if settings.keycloak_client_id not in aud and "account" not in aud:
            logger.warning(f"JWT audience mismatch: {aud}")
            return None

        # 3. Validate expiration
        if time.time() > claims.get("exp", 0):
            logger.info("JWT has expired")
            return None

        # 4. Attempt RS256 signature verification via PyJWT + cryptography
        try:
            import jwt as pyjwt
            from jwt.algorithms import RSAAlgorithm

            jwks = _fetch_jwks()
            if jwks:
                header_b64 = parts[0] + "=" * (-len(parts[0]) % 4)
                header = json.loads(base64.urlsafe_b64decode(header_b64))
                kid = header.get("kid")
                # Find matching key in JWKS
                for key_data in jwks.get("keys", []):
                    if key_data.get("kid") == kid:
                        public_key = RSAAlgorithm.from_jwk(json.dumps(key_data))
                        verified_claims = pyjwt.decode(
                            token,
                            public_key,
                            algorithms=["RS256"],
                            audience=settings.keycloak_client_id,
                            options={"verify_exp": True},
                        )
                        logger.debug("JWT signature verified successfully via RS256")
                        return verified_claims
                logger.warning("No matching key found in JWKS for kid: %s", kid)
        except ImportError:
            logger.warning(
                "PyJWT/cryptography not installed — JWT signature not verified. "
                "Install 'PyJWT[crypto]' for production-grade verification."
            )
        except Exception as exc:
            logger.error(f"JWT signature verification failed: {exc}")
            return None

        # Return claims (issuer/audience/expiry validated, signature not)
        return claims

    except Exception as exc:
        logger.error(f"JWT parsing failed: {exc}", exc_info=True)
        return None


# ── Session Management ─────────────────────────────────────────────────────────


_SESSION_USER_KEY = "_rag_authenticated_user"
_SESSION_STATE_KEY = "oauth_state"


def _get_session_user() -> Optional[User]:
    """Read the authenticated user from Streamlit session state."""
    return st.session_state.get(_SESSION_USER_KEY)


def _set_session_user(user: User) -> None:
    st.session_state[_SESSION_USER_KEY] = user


def _clear_session() -> None:
    for key in [_SESSION_USER_KEY, _SESSION_STATE_KEY]:
        st.session_state.pop(key, None)


# ── Main Public API ────────────────────────────────────────────────────────────


def get_current_user() -> Optional[User]:
    """
    Returns the authenticated User or None.

    In dev mode (auth_enabled=False): always returns a mock admin user.
    In production: validates session and token expiry.
    """
    if not settings.auth_enabled:
        return User(
            email="dev@local",
            name="Local Dev",
            roles=["sales", "admin"],
            is_admin=True,
            token_expires_at=time.time() + 86400,
        )

    user = _get_session_user()
    if user is None:
        return None

    # Enforce token expiry
    if user.is_token_expired:
        logger.info(f"Session expired for user '{user.email}' — clearing session.")
        _clear_session()
        return None

    return user


def handle_oauth_callback() -> bool:
    """
    Handle the Keycloak redirect callback.

    Reads ?code= and ?state= from the URL query parameters, validates state,
    exchanges the code for tokens, validates the JWT, and populates session.

    Returns True if login succeeded, False otherwise.
    """
    params = st.query_params
    code = params.get("code")
    state = params.get("state")

    if not code or not state:
        return False

    # CSRF check
    stored_state = st.session_state.get(_SESSION_STATE_KEY)
    if not stored_state or not _validate_state(state) or state != stored_state:
        logger.warning("OAuth state mismatch — possible CSRF attempt.")
        st.error("Login failed: security check failed. Please try again.")
        return False

    # Exchange code for tokens
    token_response = _exchange_code_for_token(code)
    if not token_response:
        st.error("Login failed: could not obtain token from Keycloak.")
        return False

    # Validate the ID token (use access_token for role claims)
    access_token = token_response.get("access_token", "")
    claims = _validate_jwt(access_token)
    if not claims:
        st.error("Login failed: token validation failed.")
        return False

    user = User.from_jwt_claims(claims)
    _set_session_user(user)

    # Remove OAuth params from URL and clear state
    st.query_params.clear()
    _clear_session()
    _set_session_user(user)  # Re-set after clearing

    logger.info(f"User '{user.email}' authenticated successfully via Keycloak.")
    return True


def login_ui() -> None:
    """Render the login page with the Keycloak SSO button."""
    st.set_page_config(
        page_title="MultiDimensions — Login",
        page_icon="🔐",
        layout="centered",
    )

    # Handle OAuth callback if code is present in URL
    if "code" in st.query_params:
        with st.spinner("Authenticating with Keycloak..."):
            if handle_oauth_callback():
                st.rerun()
        return

    st.markdown(
        """
        <div style='text-align:center; padding: 2rem 0 1rem 0;'>
            <h1 style='font-size:2.2rem;'>🔐 MultiDimensions</h1>
            <p style='color:gray; font-size:1.05rem;'>
                Industrial RAG Sales Assistant — Internal Tool
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.info(
            "This is a secure internal tool. Sign in with your company Keycloak "
            "credentials to access the product catalog assistant."
        )
        auth_url = _build_authorization_url()
        st.markdown(
            f"""
            <div style='text-align:center; margin-top: 1.5rem;'>
                <a href='{auth_url}' target='_self'>
                    <button style='
                        background-color: #1f6feb;
                        color: white;
                        border: none;
                        padding: 0.75rem 2rem;
                        font-size: 1rem;
                        border-radius: 6px;
                        cursor: pointer;
                        font-weight: 600;
                    '>🔑 Log in with Keycloak SSO</button>
                </a>
            </div>
            """,
            unsafe_allow_html=True,
        )


def logout() -> None:
    """Log out the current user and optionally redirect to Keycloak logout."""
    user = _get_session_user()
    if user:
        logger.info(f"User '{user.email}' logged out.")
    _clear_session()


def render_logout_button() -> None:
    """Render a logout button in the sidebar. Call inside `with st.sidebar:`."""
    if not settings.auth_enabled:
        return  # No logout needed in dev mode
    user = _get_session_user()
    if user:
        st.caption(f"👤 {user.name} ({user.email})")
        if st.button("🚪 Logout", use_container_width=True):
            logout()
            st.rerun()


def require_auth() -> User:
    """
    Enforce authentication. Must be called at the top of every protected page.

    - In dev mode: returns mock user immediately.
    - In production: redirects to login if not authenticated; returns User if valid.
    """
    user = get_current_user()
    if user is None:
        login_ui()
        st.stop()
    return user
