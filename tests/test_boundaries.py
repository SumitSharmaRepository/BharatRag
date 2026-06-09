"""
test_boundaries.py — edge cases and protocol conformance.

Covers: empty questions, oversized questions, HEAD support,
CORS headers, and chat history isolation.
"""
import pytest
import requests
from tests.helpers import BASE_URL


# ── Tests ─────────────────────────────────────────────────────

def test_empty_question_rejected(base_url):
    """POST /query with an empty string must return 400 or 422."""
    r = requests.post(
        f"{base_url}/query",
        json={"question": "", "user_id": "boundary_user_1"},
        timeout=15,
    )
    assert r.status_code in (400, 422), (
        f"Empty question should be rejected. Got {r.status_code}: {r.text}"
    )


def test_extremely_long_question_handled(base_url):
    """A question that exceeds the length limit must return 400 or 200, never 500."""
    long_q = "What is CRAG? " * 200   # ~2800 chars, well over the 2000-char limit
    r = requests.post(
        f"{base_url}/query",
        json={"question": long_q, "user_id": "boundary_user_2"},
        timeout=30,
    )
    assert r.status_code in (400, 200), (
        f"Long question should be rejected cleanly. Got {r.status_code}: {r.text}"
    )
    assert r.status_code != 500, "Server must never 500 on an oversized question"


def test_health_accepts_get(base_url):
    """GET /health must return 200 with a 'status' key."""
    r = requests.get(f"{base_url}/health", timeout=10)
    assert r.status_code == 200
    body = r.json()
    assert "status" in body, f"Health response missing 'status' key: {body}"


def test_health_accepts_head(base_url):
    """HEAD /health must return 200 (FastAPI handles HEAD for all GET routes)."""
    r = requests.head(f"{base_url}/health", timeout=10)
    assert r.status_code == 200, f"HEAD /health should return 200, got {r.status_code}"


def test_chat_history_isolated_per_user(base_url):
    """Messages saved for user A must not appear in user B's history."""
    # Save a message for chat_user_a
    save_r = requests.post(
        f"{base_url}/chat/save",
        json={"user_id": "chat_isolation_a", "role": "user", "content": "hello from a"},
        timeout=15,
    )
    assert save_r.status_code == 200

    # Fetch history for a completely different user
    hist_r = requests.get(
        f"{base_url}/chat/history",
        params={"user_id": "chat_isolation_b_never_used"},
        timeout=15,
    )
    assert hist_r.status_code == 200
    messages = hist_r.json()
    assert messages == [], (
        f"chat_isolation_b should see no messages. Got: {messages}"
    )


def test_cors_header_present(base_url):
    """OPTIONS preflight to /query from the production origin must return CORS headers."""
    r = requests.options(
        f"{base_url}/query",
        headers={
            "Origin": "https://bharatrag.vercel.app",
            "Access-Control-Request-Method": "POST",
        },
        timeout=10,
    )
    headers_lower = {k.lower(): v for k, v in r.headers.items()}
    assert "access-control-allow-origin" in headers_lower, (
        f"CORS header missing from OPTIONS response.\nHeaders: {dict(r.headers)}"
    )
