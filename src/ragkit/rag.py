"""Grounded answering: retrieve -> prompt with numbered sources -> validate citations."""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field

from .retrieval import Hit, HybridRetriever

SYSTEM_PROMPT = """You are a compliance assistant answering questions about {domain}.

Rules:
- Answer ONLY from the numbered sources provided. Do not use outside knowledge.
- Cite every factual sentence with the source number(s) in square brackets, e.g. [1] or [2][3].
- Quote exact obligations, numbers, time periods and penalties as written in the sources.
- If the sources do not contain the answer, reply exactly: "{refusal}" and nothing else.
- Be concise: a direct answer first, then key details as short bullet points if needed.
- This is information, not legal or medical advice."""

REFUSAL = "I can't find this in the provided documents."
_CITE = re.compile(r"\[(\d{1,2})\]")


@dataclass
class Answer:
    question: str
    answer: str
    citations: list[dict]  # sources actually cited
    sources: list[dict]  # all retrieved sources shown to the model
    grounded: bool  # has >= 1 valid citation (or is an explicit refusal)
    refused: bool
    invalid_citations: list[int] = field(default_factory=list)
    model: str = ""
    timings: dict = field(default_factory=dict)


def format_sources(hits: list[Hit], max_words: int = 400) -> str:
    blocks = []
    for n, h in enumerate(hits, 1):
        c = h.chunk
        text = " ".join(c.text.split()[:max_words])
        blocks.append(f"[{n}] {c.doc_title} | {c.path} | page {c.page}\n{text}")
    return "\n\n".join(blocks)


def source_meta(n: int, h: Hit) -> dict:
    c = h.chunk
    return {"n": n, "chunk_id": c.id, "doc_id": c.doc_id, "doc_title": c.doc_title, "section": c.section,
            "path": c.path, "page": c.page, "score": round(h.score, 4), "retrieval": h.sources}


def validate_citations(text: str, n_sources: int) -> tuple[str, list[int], list[int]]:
    """Return (cleaned text, valid cited numbers, invalid numbers). Invalid markers are removed."""
    cited = [int(m) for m in _CITE.findall(text)]
    invalid = sorted({c for c in cited if not 1 <= c <= n_sources})
    for c in invalid:
        text = text.replace(f"[{c}]", "")
    valid = sorted({c for c in cited if 1 <= c <= n_sources})
    return text.strip(), valid, invalid


class RAGPipeline:
    def __init__(self, retriever: HybridRetriever, llm, domain: str, k: int = 6, mode: str = "hybrid_rerank"):
        self.retriever, self.llm, self.domain, self.k, self.mode = retriever, llm, domain, k, mode

    def ask(self, question: str) -> Answer:
        t0 = time.perf_counter()
        hits = self.retriever.search(question, k=self.k, mode=self.mode)
        t1 = time.perf_counter()
        if not hits:  # nothing relevant retrieved: refuse without spending an LLM call
            return Answer(question, REFUSAL, [], [], grounded=True, refused=True, model="none",
                          timings={"retrieval_s": round(t1 - t0, 3), "generation_s": 0.0})
        system = SYSTEM_PROMPT.format(domain=self.domain, refusal=REFUSAL)
        user = f"Sources:\n\n{format_sources(hits)}\n\nQuestion: {question}"
        resp = self.llm.generate(system, user)
        t2 = time.perf_counter()
        text = re.sub(r"<think>.*?</think>", "", resp.text, flags=re.S).strip()  # reasoning models
        refused = REFUSAL.lower().rstrip(".") in text.lower()
        text, valid, invalid = validate_citations(text, len(hits))
        sources = [source_meta(n, h) for n, h in enumerate(hits, 1)]
        return Answer(
            question=question, answer=text, citations=[s for s in sources if s["n"] in valid], sources=sources,
            grounded=refused or bool(valid), refused=refused, invalid_citations=invalid, model=resp.model,
            timings={"retrieval_s": round(t1 - t0, 3), "generation_s": round(t2 - t1, 3)},
        )
