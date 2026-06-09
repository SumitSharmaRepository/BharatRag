"""
test_api_behaviour.py — endpoint contracts and response shape.

Proves that every endpoint returns the right content-type,
the right JSON keys, and handles degenerate inputs without 500s.

The "query returns JSON not HTML" test specifically guards against
the production bug where the server sent a plain-text error that
the frontend tried to parse as JSON ("Unexpected token T").
"""
import pytest
import requests
from tests.helpers import BASE_URL


# ── Tests ─────────────────────────────────────────────────────

def test_query_returns_json_not_html(base_url):
    """
    POST /query must always return application/json, never plain text or HTML.
    Guards against the 'Unexpected token T in JSON at position 0' production bug.
    """
    r = requests.post(
        f"{base_url}/query",
        json={"question": "What is a document?", "user_id": "api_behaviour_user_1"},
        timeout=120,
    )
    content_type = r.headers.get("content-type", "")
    assert "application/json" in content_type, (
        f"Expected application/json, got: {content_type}"
    )
    body = r.text.strip()
    assert not body.startswith("<"), f"Response looks like HTML: {body[:100]}"
    assert r.status_code in (200, 400, 422), f"Unexpected status: {r.status_code}"


def test_documents_returns_active_and_archived_keys(base_url):
    """GET /documents must return a JSON object with 'active' and 'archived' keys."""
    r = requests.get(
        f"{base_url}/documents",
        params={"user_id": "api_behaviour_user_2"},
        timeout=30,
    )
    assert r.status_code == 200
    body = r.json()
    assert "active"   in body, f"'active' key missing from /documents response: {body}"
    assert "archived" in body, f"'archived' key missing from /documents response: {body}"


def test_chat_save_returns_id(base_url):
    """POST /chat/save must return a response with 'id' and 'saved': true."""
    r = requests.post(
        f"{base_url}/chat/save",
        json={
            "user_id": "api_behaviour_user_3",
            "role":    "user",
            "content": "integration test message",
        },
        timeout=15,
    )
    assert r.status_code == 200
    body = r.json()
    assert "id"    in body,           f"'id' missing from /chat/save response: {body}"
    assert body.get("saved") is True, f"'saved' should be true: {body}"


def test_chat_history_returns_list(base_url):
    """GET /chat/history must return a JSON list (may be empty)."""
    r = requests.get(
        f"{base_url}/chat/history",
        params={"user_id": "api_behaviour_user_4_fresh"},
        timeout=15,
    )
    assert r.status_code == 200
    body = r.json()
    # API returns a bare list
    assert isinstance(body, list), f"Expected a list from /chat/history, got: {type(body)}: {body}"


def test_delete_nonexistent_doc_handled(base_url):
    """DELETE on a doc that doesn't exist must return 200 or 404, never 500."""
    r = requests.delete(
        f"{base_url}/documents/nonexistent_file_xyz_abc.pdf",
        params={"user_id": "nobody_user_xyz", "mode": "permanent"},
        timeout=30,
    )
    assert r.status_code in (200, 404), (
        f"Deleting nonexistent doc should not 500. Got {r.status_code}: {r.text}"
    )
    assert r.status_code != 500, "Server must never 500 on a missing document delete"
