# auth.py - Backend-owned auth + per-user data via Supabase's REST API.
#
# We talk to Supabase over plain HTTP (requests) instead of the Python SDK to
# avoid SDK version churn. The frontend never sees Supabase — it calls the
# backend, which runs the OAuth/magic-link (PKCE) flow, verifies the session
# JWT, and reads/writes the database with the service-role key.
#
# Required env:
#   SUPABASE_URL, SUPABASE_ANON_KEY, SUPABASE_SERVICE_ROLE_KEY,
#   SUPABASE_JWT_SECRET, FRONTEND_URL, BACKEND_URL

import os
import time
import base64
import hashlib
import secrets
from urllib.parse import quote

import requests
import jwt

SUPABASE_URL = (os.getenv("SUPABASE_URL") or "").rstrip("/")
ANON_KEY = os.getenv("SUPABASE_ANON_KEY")
SERVICE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
JWT_SECRET = os.getenv("SUPABASE_JWT_SECRET")
FRONTEND_URL = (os.getenv("FRONTEND_URL") or "http://localhost:3000").rstrip("/")
BACKEND_URL = (os.getenv("BACKEND_URL") or "http://localhost:8000").rstrip("/")

auth_enabled = bool(SUPABASE_URL and ANON_KEY and SERVICE_KEY and JWT_SECRET)

# ── PKCE state (single-instance in-memory; keep the backend to one worker) ──
_pkce: dict = {}
PKCE_TTL = 600  # seconds


def _b64url(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).decode().rstrip("=")


def _new_challenge():
    verifier = _b64url(secrets.token_bytes(32))
    challenge = _b64url(hashlib.sha256(verifier.encode()).digest())
    return verifier, challenge


def _store_state(verifier: str) -> str:
    state = secrets.token_urlsafe(16)
    _pkce[state] = (verifier, time.time())
    cutoff = time.time() - PKCE_TTL
    for k in [k for k, v in _pkce.items() if v[1] < cutoff]:
        _pkce.pop(k, None)
    return state


def _take_verifier(state: str):
    entry = _pkce.pop(state, None)
    return entry[0] if entry else None


def _redirect_to(state: str) -> str:
    return f"{BACKEND_URL}/auth/callback?state={state}"


# ── OAuth / magic link ─────────────────────────────────────────────────────

def start_google() -> str:
    """Build the Supabase Google authorize URL (browser gets redirected here)."""
    verifier, challenge = _new_challenge()
    state = _store_state(verifier)
    rt = quote(_redirect_to(state), safe="")
    return (
        f"{SUPABASE_URL}/auth/v1/authorize?provider=google"
        f"&redirect_to={rt}"
        f"&code_challenge={challenge}&code_challenge_method=s256"
    )


def start_magic_link(email: str) -> None:
    """Ask Supabase to email a magic link that returns to our callback."""
    verifier, challenge = _new_challenge()
    state = _store_state(verifier)
    resp = requests.post(
        f"{SUPABASE_URL}/auth/v1/otp",
        headers={"apikey": ANON_KEY, "Content-Type": "application/json"},
        params={"redirect_to": _redirect_to(state)},
        json={
            "email": email,
            "create_user": True,
            "code_challenge": challenge,
            "code_challenge_method": "s256",
        },
        timeout=10,
    )
    resp.raise_for_status()


def exchange_code(state: str, code: str) -> dict:
    """Exchange the ?code from the callback for a session."""
    verifier = _take_verifier(state)
    if not verifier:
        raise ValueError("Missing or expired login state")
    resp = requests.post(
        f"{SUPABASE_URL}/auth/v1/token",
        headers={"apikey": ANON_KEY, "Content-Type": "application/json"},
        params={"grant_type": "pkce"},
        json={"auth_code": code, "code_verifier": verifier},
        timeout=10,
    )
    resp.raise_for_status()
    data = resp.json()
    return {"access_token": data["access_token"], "refresh_token": data["refresh_token"]}


def refresh(refresh_token: str) -> dict:
    resp = requests.post(
        f"{SUPABASE_URL}/auth/v1/token",
        headers={"apikey": ANON_KEY, "Content-Type": "application/json"},
        params={"grant_type": "refresh_token"},
        json={"refresh_token": refresh_token},
        timeout=10,
    )
    resp.raise_for_status()
    data = resp.json()
    return {"access_token": data["access_token"], "refresh_token": data["refresh_token"]}


# ── Token verification ─────────────────────────────────────────────────────

def verify(access_token: str) -> dict:
    """Verify a Supabase access token (HS256 signed with the JWT secret)."""
    claims = jwt.decode(
        access_token,
        JWT_SECRET,
        algorithms=["HS256"],
        audience="authenticated",
    )
    return {
        "id": claims["sub"],
        "email": claims.get("email"),
        "name": (claims.get("user_metadata") or {}).get("name"),
    }


# ── Database (Supabase PostgREST, service role) ────────────────────────────

def _svc_headers(extra: dict = None) -> dict:
    h = {
        "apikey": SERVICE_KEY,
        "Authorization": f"Bearer {SERVICE_KEY}",
        "Content-Type": "application/json",
    }
    if extra:
        h.update(extra)
    return h


def db_get(table: str, params: dict) -> list:
    r = requests.get(f"{SUPABASE_URL}/rest/v1/{table}", headers=_svc_headers(), params=params, timeout=10)
    r.raise_for_status()
    return r.json()


def db_insert(table: str, row: dict) -> list:
    r = requests.post(
        f"{SUPABASE_URL}/rest/v1/{table}",
        headers=_svc_headers({"Prefer": "return=representation"}),
        json=row,
        timeout=10,
    )
    r.raise_for_status()
    return r.json() if r.text.strip() else []


def db_upsert(table: str, row: dict, on_conflict: str) -> None:
    r = requests.post(
        f"{SUPABASE_URL}/rest/v1/{table}",
        headers=_svc_headers({"Prefer": "resolution=merge-duplicates"}),
        params={"on_conflict": on_conflict},
        json=row,
        timeout=10,
    )
    r.raise_for_status()


def db_delete(table: str, params: dict) -> None:
    r = requests.delete(f"{SUPABASE_URL}/rest/v1/{table}", headers=_svc_headers(), params=params, timeout=10)
    r.raise_for_status()
