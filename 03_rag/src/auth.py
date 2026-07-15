import urllib.parse
from dataclasses import dataclass
import streamlit as st
from config.settings import settings

@dataclass
class User:
    email: str
    roles: list[str]
    is_admin: bool

def get_current_user() -> User | None:
    """Returns the currently authenticated user, or None if not logged in."""
    if not settings.auth_enabled:
        # Mock user for local development
        return User(email="local.dev@multidimensions.com", roles=["sales", "admin"], is_admin=True)

    if "user" in st.session_state:
        return st.session_state["user"]
    
    return None

def login_ui():
    """Renders the Keycloak login button and handles the OAuth redirect."""
    st.markdown("## 🔐 Enterprise Login Required")
    st.markdown("Please sign in using your company Keycloak credentials to access the internal RAG catalog.")
    
    # In a full implementation using authlib, this would redirect to the OIDC auth endpoint:
    auth_url = f"{settings.keycloak_server_url}/realms/{settings.keycloak_realm}/protocol/openid-connect/auth"
    
    # For now, we simulate the redirect/callback. In production, this button would be a real OAuth redirect link.
    if st.button("Log in with Keycloak"):
        # SIMULATION: In reality, authlib would handle the token exchange and we'd parse the JWT.
        # We mock a successful SSO login here for the scaffold.
        st.session_state["user"] = User(
            email="sales.rep@multidimensions.com", 
            roles=["sales"], 
            is_admin=False
        )
        st.rerun()

def require_auth() -> User:
    """
    Wraps the Streamlit app. If not authenticated, halts execution 
    and shows the Keycloak login screen.
    """
    user = get_current_user()
    if not user:
        login_ui()
        st.stop()
    return user
