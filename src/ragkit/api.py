"""FastAPI service.  Run:  uvicorn ragkit.api:app --port 8000   (PYTHONPATH=src)"""
from __future__ import annotations

from dataclasses import asdict
from typing import Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from . import service
from .llm import LLMError

app = FastAPI(title="Compliance RAG Assistant", version="1.0.0",
              description="Cited answers over WA mine-safety regulations and infection-control guidelines.")

Mode = Literal["bm25", "dense", "hybrid", "hybrid_rerank"]


class AskRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=1000)
    corpus: str = "mining"
    backend: str = Field("gemini", description="'gemini' or 'ollama[:model]'")
    k: int = Field(6, ge=1, le=12)
    mode: Mode = "hybrid_rerank"


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=2, max_length=1000)
    corpus: str = "mining"
    k: int = Field(5, ge=1, le=20)
    mode: Mode = "hybrid_rerank"


def _check_corpus(corpus: str):
    if corpus not in service.CONFIG:
        raise HTTPException(404, f"unknown corpus {corpus!r}; available: {sorted(service.CONFIG)}")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/corpora")
def corpora():
    return service.corpora()


@app.post("/search")
def search(req: SearchRequest):
    _check_corpus(req.corpus)
    hits = service.retriever(req.corpus).search(req.query, k=req.k, mode=req.mode)
    return [{"chunk_id": h.chunk.id, "doc_title": h.chunk.doc_title, "section": h.chunk.section,
             "path": h.chunk.path, "page": h.chunk.page, "score": round(h.score, 4), "retrieval": h.sources,
             "text": h.chunk.text} for h in hits]


@app.post("/ask")
def ask(req: AskRequest):
    _check_corpus(req.corpus)
    if not (req.backend == "gemini" or req.backend.startswith("ollama")):
        raise HTTPException(422, "backend must be 'gemini' or 'ollama[:model]'")
    try:
        return asdict(service.pipeline(req.corpus, req.backend, req.k, req.mode).ask(req.question))
    except LLMError as e:
        raise HTTPException(503, f"LLM backend unavailable: {e}") from e
