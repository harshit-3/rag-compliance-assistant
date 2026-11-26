"""Shared, lazily-built resources (retrievers per corpus, LLM clients) for the API and UI."""
from __future__ import annotations

import logging
import os
import threading
import time
from functools import lru_cache
from pathlib import Path

import requests
import yaml

from .ingest import build_corpus, load_chunks, save_chunks
from .llm import get_llm
from .rag import RAGPipeline
from .retrieval import HybridRetriever

ROOT = Path(os.getenv("RAG_ROOT", Path(__file__).resolve().parents[2]))
log = logging.getLogger("ragkit")
_BUILD_LOCK = threading.Lock()  # concurrent first visits must not build the same index twice


def _load_dotenv() -> None:
    """Minimal .env loader (project dir, then its parent). Real env vars always win."""
    for env in (ROOT / ".env", ROOT.parent / ".env"):
        if env.exists():
            for line in env.read_text(encoding="utf-8").splitlines():
                k, sep, v = line.strip().partition("=")
                if sep and k and not k.startswith("#"):
                    os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


_load_dotenv()
CONFIG = yaml.safe_load((ROOT / "configs" / "corpora.yaml").read_text(encoding="utf-8"))
DOMAIN = {"mining": "Western Australian mine safety regulations and codes of practice",
          "healthcare": "Australian infection prevention and control guidelines for healthcare"}


def chunks_path(corpus: str) -> Path:
    return ROOT / "data" / "chunks" / f"{corpus}.jsonl"


def download_pdfs(corpus: str) -> None:
    """Fetch the corpus PDFs from the official publishers (they are not redistributed in this repo)."""
    for doc in CONFIG[corpus]["documents"]:
        out = ROOT / "data" / "raw" / corpus / f"{doc['id']}.pdf"
        if out.exists():
            continue
        t0 = time.time()
        r = requests.get(doc["url"], timeout=(15, 120), headers={"User-Agent": "Mozilla/5.0 (research RAG demo)"})
        r.raise_for_status()
        if not r.content.startswith(b"%PDF"):
            raise ValueError(f"{doc['id']}: not a PDF (content-type {r.headers.get('content-type')})")
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(r.content)
        log.warning("downloaded %s (%.1f MB) in %.1fs", doc["id"], len(r.content) / 1e6, time.time() - t0)


def retriever(corpus: str) -> HybridRetriever:
    with _BUILD_LOCK:
        return _retriever(corpus)


@lru_cache(maxsize=4)
def _retriever(corpus: str) -> HybridRetriever:
    if corpus not in CONFIG:
        raise KeyError(corpus)
    t0 = time.time()
    path = chunks_path(corpus)
    if not path.exists():  # first run: download the PDFs and chunk them
        download_pdfs(corpus)
        save_chunks(build_corpus(corpus, CONFIG[corpus], ROOT / "data" / "raw"), path)
        log.warning("%s: chunked in %.1fs", corpus, time.time() - t0)
    r = HybridRetriever(load_chunks(path), cache_dir=ROOT / "data" / "index",
                        reranker=os.getenv("RAG_RERANKER", "BAAI/bge-reranker-base") or None)
    log.warning("%s: retriever ready in %.1fs (%d chunks)", corpus, time.time() - t0, len(r.chunks))
    return r


@lru_cache(maxsize=4)
def llm(backend: str):
    return get_llm(backend)


def pipeline(corpus: str, backend: str, k: int = 6, mode: str = "hybrid_rerank") -> RAGPipeline:
    return RAGPipeline(retriever(corpus), llm(backend), DOMAIN[corpus], k=k, mode=mode)


def corpora() -> dict:
    return {name: {"title": spec["title"], "documents": [{"id": d["id"], "title": d["title"], "source": d["source"]}
                                                          for d in spec["documents"]]}
            for name, spec in CONFIG.items()}
