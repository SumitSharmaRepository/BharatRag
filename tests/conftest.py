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

    yield BASE_URL
    # Intentionally not terminating proc — user may prefer it running
