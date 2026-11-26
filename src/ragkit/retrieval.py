"""Hybrid retrieval: BM25 + dense embeddings, fused with Reciprocal Rank Fusion, then a
cross-encoder reranker. Plus an exact-reference resolver for legal citations ("reg 617").

Every stage is switchable so the evaluation can measure what each one contributes.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi

from .ingest import Chunk

_TOKEN = re.compile(r"[a-z0-9]+(?:\.[0-9]+)*[a-z]?")
_STOP = set("""a an the of to in on for and or is are be by with as at from that this which what who how
when where do does must should any all it its their there if than then into under about can may""".split())
# "regulation 617", "reg. 617(2)", "r. 617", "r 621A"
_REG_REF = re.compile(r"\b(?:regulation|reg\.?|r\.?)\s*(\d{1,4}[A-Z]{0,2})\b", re.I)


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN.findall(text.lower()) if t not in _STOP]


@dataclass
class Hit:
    chunk: Chunk
    score: float
    sources: dict  # stage -> rank, for debugging / explainability


class BM25Retriever:
    def __init__(self, chunks: list[Chunk]):
        self.chunks = chunks
        self.bm25 = BM25Okapi([tokenize(c.embed_text) for c in chunks])

    def search(self, query: str, k: int = 50) -> list[tuple[int, float]]:
        scores = self.bm25.get_scores(tokenize(query))
        idx = np.argsort(-scores)[:k]
        return [(int(i), float(scores[i])) for i in idx if scores[i] > 0]


@lru_cache(maxsize=4)
def _st_model(name: str, device: str | None):
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(name, device=device)


class DenseRetriever:
    """Sentence-transformer bi-encoder. Embeddings are cached on disk per (model, corpus content)."""

    QUERY_PREFIX = {"BAAI/bge-small-en-v1.5": "Represent this sentence for searching relevant passages: ",
                    "BAAI/bge-base-en-v1.5": "Represent this sentence for searching relevant passages: "}

    def __init__(self, chunks: list[Chunk], model: str = "BAAI/bge-base-en-v1.5", cache_dir: Path | None = None,
                 device: str | None = None):
        self.chunks, self.model_name = chunks, model
        self.model = _st_model(model, device)
        digest = hashlib.sha1(("".join(c.id + c.text for c in chunks) + model).encode()).hexdigest()[:12]
        cache = (cache_dir / f"emb_{digest}.npy") if cache_dir else None
        if cache and cache.exists():
            self.emb = np.load(cache)
        else:
            self.emb = self.model.encode([c.embed_text for c in chunks], batch_size=32, normalize_embeddings=True,
                                         show_progress_bar=False, convert_to_numpy=True).astype(np.float32)
            if cache:
                cache.parent.mkdir(parents=True, exist_ok=True)
                np.save(cache, self.emb)

    def search(self, query: str, k: int = 50) -> list[tuple[int, float]]:
        q = self.model.encode([self.QUERY_PREFIX.get(self.model_name, "") + query], normalize_embeddings=True,
                              convert_to_numpy=True)[0]
        scores = self.emb @ q
        idx = np.argsort(-scores)[:k]
        return [(int(i), float(scores[i])) for i in idx]


@lru_cache(maxsize=2)
def _cross_encoder(name: str, device: str | None):
    from sentence_transformers import CrossEncoder
    return CrossEncoder(name, device=device, max_length=512)


def rrf(rankings: list[list[int]], k: int = 60) -> dict[int, float]:
    """Reciprocal Rank Fusion: robust to the very different score scales of BM25 and cosine."""
    fused: dict[int, float] = {}
    for ranking in rankings:
        for rank, idx in enumerate(ranking):
            fused[idx] = fused.get(idx, 0.0) + 1.0 / (k + rank + 1)
    return fused


class HybridRetriever:
    def __init__(self, chunks: list[Chunk], dense_model: str = "BAAI/bge-base-en-v1.5",
                 reranker: str | None = "BAAI/bge-reranker-base", cache_dir: Path | None = None,
                 device: str | None = None):
        self.chunks = chunks
        self.bm25 = BM25Retriever(chunks)
        self.dense = DenseRetriever(chunks, dense_model, cache_dir, device) if dense_model else None
        self.reranker_name, self.device = reranker, device
        # Index of regulation number -> chunk indices, for exact citation lookups.
        self.reg_index: dict[str, list[int]] = {}
        for i, c in enumerate(chunks):
            m = re.match(r"^(\d{1,4}[A-Z]{0,2})\.\s", c.section)
            if m:
                self.reg_index.setdefault(m.group(1).upper(), []).append(i)

    def search(self, query: str, k: int = 5, mode: str = "hybrid_rerank", candidates: int = 30) -> list[Hit]:
        """mode: bm25 | dense | hybrid | hybrid_rerank"""
        stages: dict[str, list[int]] = {}
        if mode in ("bm25", "hybrid", "hybrid_rerank"):
            stages["bm25"] = [i for i, _ in self.bm25.search(query, candidates)]
        if mode in ("dense", "hybrid", "hybrid_rerank") and self.dense:
            stages["dense"] = [i for i, _ in self.dense.search(query, candidates)]
        if mode in ("bm25", "dense"):
            ranking = stages[mode]
            scores = {i: 1.0 / (r + 1) for r, i in enumerate(ranking)}
        else:
            scores = rrf(list(stages.values()))
        # Exact legal references in the question are pinned to the top.
        pinned = [i for ref in _REG_REF.findall(query) for i in self.reg_index.get(ref.upper(), [])]
        order = list(dict.fromkeys(pinned + sorted(scores, key=scores.get, reverse=True)))[:candidates]

        if mode == "hybrid_rerank" and self.reranker_name and order:
            ce = _cross_encoder(self.reranker_name, self.device)
            ce_scores = ce.predict([(query, self.chunks[i].embed_text) for i in order], batch_size=16,
                                   show_progress_bar=False)
            ce_map = dict(zip(order, (float(s) for s in ce_scores), strict=True))
            for i in pinned:
                ce_map[i] = ce_map[i] + 100.0  # an explicitly cited regulation always survives reranking
            order = sorted(order, key=ce_map.get, reverse=True)
            scores = ce_map

        hits = []
        for i in order[:k]:
            src = {name: ranking.index(i) + 1 for name, ranking in stages.items() if i in ranking}
            if i in pinned:
                src["reference"] = 1
            hits.append(Hit(self.chunks[i], float(scores.get(i, 0.0)), src))
        return hits
