"""
conftest.py — session-scoped server fixture for the integration test suite.

Starts the API server if not already running on port 8000, waits up to 10s
for /health to respond, then yields BASE_URL to all tests.
Does NOT kill the server after the session (user may have started it manually).
"""
import os
import time
import subprocess
import pytest
import requests
from tests.helpers import BASE_URL

# All test user IDs whose Pinecone data is purged at session start so tests are repeatable.
_ISOLATION_USERS = [
    "user_a_isolation",
    "user_b_isolation",
    "user_c_isolation",
    "user_d_isolation",
    "user_d2_isolation",
    "user_e_isolation",
    "user_f_isolation",
    "user_h_isolation",
]


@pytest.fixture(scope="session")
def base_url():
    """
    Ensure the API is running on port 8000.
    If already up, use it as-is. Otherwise launch uvicorn and wait.
    """
    # /health hits Pinecone (describe_index_stats) so it can take 5-10s
    try:
        r = requests.get(f"{BASE_URL}/health", timeout=15)
        if r.status_code == 200:
            _purge_isolation_users(BASE_URL)
            yield BASE_URL
            return
    except Exception:
        pass

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    proc = subprocess.Popen(
        ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"],
        cwd=repo_root,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        env={**os.environ, "UPLOAD_RATE_LIMIT": "1000/hour", "DELETE_RATE_LIMIT": "1000/hour"},
    )

    deadline = time.time() + 30
    started  = False
    while time.time() < deadline:
        try:
            r = requests.get(f"{BASE_URL}/health", timeout=15)
            if r.status_code == 200:
                started = True
                break
        except Exception:
            pass
        time.sleep(1)

    if not started:
        proc.terminate()
        pytest.fail("Server did not respond to /health within 30 seconds")

    _purge_isolation_users(BASE_URL)
    yield BASE_URL
    # Intentionally not terminating proc — user may prefer it running


def _purge_isolation_users(base_url: str) -> None:
    """Permanently delete all test-user data from Pinecone before the session runs.

    This ensures repeated test runs always start from a clean state, regardless
    of what state previous runs left behind (archived, partial, or missing chunks).
    Permanent delete is a single batch Pinecone call — fast even for large docs.
    Non-existent docs return 200 so this is safe to run unconditionally.
    """
    for user_id in _ISOLATION_USERS:
        try:
            requests.delete(
                f"{base_url}/documents/CRAG.pdf",
                params={"user_id": user_id, "mode": "permanent"},
                timeout=30,
            )
        except Exception:
            pass  # best-effort cleanup; test failures will surface real problems
