"""API contract tests that need no models or downloaded documents."""
from fastapi.testclient import TestClient

from ragkit.api import app

client = TestClient(app)


def test_health():
    assert client.get("/health").json() == {"status": "ok"}


def test_corpora_lists_both_knowledge_bases():
    d = client.get("/corpora").json()
    assert {"mining", "healthcare"} <= set(d)
    assert any("Mines" in doc["title"] for doc in d["mining"]["documents"])


def test_unknown_corpus_404():
    assert client.post("/ask", json={"question": "hello there", "corpus": "finance"}).status_code == 404
    assert client.post("/search", json={"query": "x y", "corpus": "finance"}).status_code == 404


def test_validation_errors():
    assert client.post("/ask", json={"question": "hi there", "backend": "gpt"}).status_code == 422
    assert client.post("/ask", json={"question": "hi"}).status_code == 422  # too short
    assert client.post("/ask", json={"question": "valid question", "k": 50}).status_code == 422
    assert client.post("/search", json={"query": "valid", "mode": "magic"}).status_code == 422
