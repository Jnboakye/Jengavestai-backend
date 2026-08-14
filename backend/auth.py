# auth.py - Backend-owned auth + per-user data via Supabase.
#
# The frontend never talks to Supabase. It calls these endpoints; the backend
# holds all the keys, runs the OAuth / magic-link flow, verifies sessions, and
# reads/writes the database with the service-role key.
#
# Sessions use bearer tokens (the Supabase access/refresh tokens) because the
# frontend and backend live on different domains (cross-site cookies are
# unreliable). The backend still fully owns issuing/refreshing/validating them.
#
# Required env (see .env.example):
#   SUPABASE_URL, SUPABASE_ANON_KEY, SUPABASE_SERVICE_ROLE_KEY,
#   SUPABASE_JWT_SECRET, FRONTEND_URL, BACKEND_URL

import os

import jwt

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY")
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
SUPABASE_JWT_SECRET = os.getenv("SUPABASE_JWT_SECRET")
FRONTEND_URL = os.getenv("FRONTEND_URL", "http://localhost:3000")
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

auth_enabled = bool(
    SUPABASE_URL and SUPABASE_ANON_KEY and SUPABASE_SERVICE_ROLE_KEY and SUPABASE_JWT_SECRET
)

# ── Supabase clients (created lazily so the app boots without auth configured) ─
_auth_client = None   # anon key, PKCE flow — drives OAuth / magic link
_service_client = None  # service role — database access, bypasses RLS


def _clients():
    """Return (auth_client, service_client), creating them once."""
    global _auth_client, _service_client
    if not auth_enabled:
        raise RuntimeError("Supabase auth is not configured")
    if _auth_client is None:
        from supabase import create_client
        from supabase.lib.client_options import ClientOptions
        _auth_client = create_client(
            SUPABASE_URL, SUPABASE_ANON_KEY, ClientOptions(flow_type="pkce")
        )
        _service_client = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)
    return _auth_client, _service_client


def service():
    """The service-role client for database work."""
    return _clients()[1]


# ── OAuth / magic link ─────────────────────────────────────────────────────

CALLBACK = lambda: f"{BACKEND_URL}/auth/callback"  # noqa: E731


def start_google() -> str:
    """Return the Google OAuth URL to redirect the browser to."""
    auth_client, _ = _clients()
    res = auth_client.auth.sign_in_with_oauth(
        {"provider": "google", "options": {"redirect_to": CALLBACK()}}
    )
    return res.url


def start_magic_link(email: str) -> None:
    """Email the user a magic link that returns to our callback."""
    auth_client, _ = _clients()
    auth_client.auth.sign_in_with_otp(
        {"email": email, "options": {"email_redirect_to": CALLBACK()}}
    )


def exchange_code(code: str) -> dict:
    """Exchange the ?code from the callback for a session."""
    auth_client, _ = _clients()
    res = auth_client.auth.exchange_code_for_session({"auth_code": code})
    session = res.session
    return {
        "access_token": session.access_token,
        "refresh_token": session.refresh_token,
    }


def refresh(refresh_token: str) -> dict:
    """Refresh an expired session."""
    auth_client, _ = _clients()
    res = auth_client.auth.refresh_session(refresh_token)
    session = res.session
    return {
        "access_token": session.access_token,
        "refresh_token": session.refresh_token,
    }


# ── Token verification ─────────────────────────────────────────────────────

def verify(access_token: str) -> dict:
    """
    Decode + verify a Supabase access token (HS256, signed with the project's
    JWT secret). Returns {id, email} or raises jwt errors.
    """
    claims = jwt.decode(
        access_token,
        SUPABASE_JWT_SECRET,
        algorithms=["HS256"],
        audience="authenticated",
    )
    return {
        "id": claims["sub"],
        "email": claims.get("email"),
        "name": (claims.get("user_metadata") or {}).get("name"),
    }
