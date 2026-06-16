"""Shared constants and HTTP helpers for the integration test suite."""
import os
import requests

BASE_URL = "http://localhost:8000"
CRAG_PDF  = os.path.join(os.path.dirname(__file__), "..", "data", "CRAG.pdf")


def upload_doc(filename: str, user_id: str, base_url: str = BASE_URL) -> dict:
    with open(filename, "rb") as f:
        r = requests.post(
            f"{base_url}/upload",
            files={"file": (os.path.basename(filename), f, "application/pdf")},
            data={"user_id": user_id},
            timeout=600,
        )
    r.raise_for_status()
    return r.json()


def query(question: str, user_id: str, base_url: str = BASE_URL, language: str = "English") -> dict:
    r = requests.post(
        f"{base_url}/query",
        json={"question": question, "language": language, "user_id": user_id},
        timeout=120,
    )
    r.raise_for_status()
    return r.json()


def delete_doc(doc_name: str, user_id: str, mode: str, base_url: str = BASE_URL) -> dict:
    r = requests.delete(
        f"{base_url}/documents/{requests.utils.quote(doc_name, safe='')}",
        params={"user_id": user_id, "mode": mode},
        timeout=120,
    )
    r.raise_for_status()
    return r.json()
