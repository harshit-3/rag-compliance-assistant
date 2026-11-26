"""Streamlit chat UI.  Run:  streamlit run app.py"""
from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from ragkit import service  # noqa: E402
from ragkit.llm import LLMError  # noqa: E402

st.set_page_config(page_title="Compliance RAG Assistant", layout="wide")

# Public demo (Streamlit Community Cloud mounts apps under /mount/src): no local Ollama, a small CPU, and a
# shared Gemini quota. So: Gemini only, a per-session question cap, and a lighter reranker (MiniLM matches
# bge-reranker-base on Recall@5 at ~6x lower CPU latency, see README). Locally everything stays on.
PUBLIC_DEMO = os.getenv("RAG_PUBLIC_DEMO", "1" if str(ROOT).startswith("/mount/src") else "0") == "1"
if PUBLIC_DEMO:
    os.environ.setdefault("RAG_BACKENDS", "gemini")
    os.environ.setdefault("RAG_MAX_QUESTIONS", "15")
    os.environ.setdefault("RAG_RERANKER", "cross-encoder/ms-marco-MiniLM-L-6-v2")
BACKENDS = [b.strip() for b in os.getenv("RAG_BACKENDS", "gemini,ollama:qwen3:8b").split(",") if b.strip()]
MAX_QUESTIONS = int(os.getenv("RAG_MAX_QUESTIONS", "0"))  # 0 = unlimited

EXAMPLES = {
    "mining": ["What has to be included in a mine safety management system?",
               "How long must ventilation testing records be kept for an underground mine?",
               "What does regulation 625B require?",
               "What is the workplace noise exposure standard?"],
    "healthcare": ["When should healthcare workers perform hand hygiene?",
                   "What PPE is needed for a patient with an airborne infection like measles?",
                   "How should used linen be handled to avoid spreading infection?",
                   "What should a hospital do to contain an outbreak?"],
}

with st.sidebar:
    st.header("Settings")
    corpus = st.radio("Knowledge base", list(service.CONFIG), format_func=lambda c: service.CONFIG[c]["title"])
    backend = st.radio("Model", BACKENDS,
                       format_func=lambda b: "Gemini (cloud)" if b == "gemini" else "Qwen3 8B (local, private)")
    mode = st.selectbox("Retrieval", ["hybrid_rerank", "hybrid", "dense", "bm25"])
    k = st.slider("Sources per answer", 2, 10, 6)
    st.divider()
    st.caption("Documents")
    for d in service.CONFIG[corpus]["documents"]:
        st.caption(f"• {d['title']} ({d['source']})")
    res = ROOT / "eval" / "results" / f"{corpus}.json"
    if res.exists():
        r = json.loads(res.read_text())
        if "retrieval" in r:
            st.caption(f"Eval: Recall@5 {r['retrieval']['hybrid_rerank']['recall@5']:.0%} on "
                       f"{r['retrieval']['hybrid_rerank']['n']} questions")

st.title("Compliance RAG Assistant")
st.caption("Every answer is grounded in the official documents with numbered citations. "
           "Information only, not legal or medical advice.")

if "history" not in st.session_state:
    st.session_state.history = []

cols = st.columns(len(EXAMPLES[corpus]))
clicked = None
for col, ex in zip(cols, EXAMPLES[corpus], strict=True):
    if col.button(ex, use_container_width=True):
        clicked = ex

question = st.chat_input("Ask about the regulations / guidelines…") or clicked

if PUBLIC_DEMO and "ready" not in st.session_state:
    with st.spinner("First visit after a restart: downloading the official PDFs and building the index (~1 min)…"):
        service.retriever(corpus)
    st.session_state.ready = True

for turn in st.session_state.history:
    with st.chat_message("user"):
        st.write(turn["q"])
    with st.chat_message("assistant"):
        st.markdown(turn["md"])

if question and MAX_QUESTIONS and len(st.session_state.history) >= MAX_QUESTIONS:
    st.info(f"This public demo is limited to {MAX_QUESTIONS} questions per session. "
            "Run it locally (see the GitHub README) for unlimited use and the private local model.")
    question = None

if question:
    with st.chat_message("user"):
        st.write(question)
    with st.chat_message("assistant"):
        with st.spinner("Retrieving and reading sources…"):
            try:
                a = service.pipeline(corpus, backend, k, mode).ask(question)
            except LLMError as e:
                st.error(f"Model unavailable: {e}")
                st.stop()
        answer_md = re.sub(r"\[(\d+)\]", r"**[\1]**", a.answer)
        st.markdown(answer_md)
        badge = "refused (not in documents)" if a.refused else ("grounded" if a.grounded else "⚠ no valid citations")
        st.caption(f"{a.model} · retrieval {a.timings['retrieval_s']:.2f}s · generation "
                   f"{a.timings['generation_s']:.1f}s · {badge}")
        cited = {c["n"] for c in a.citations}
        with st.expander(f"Sources ({len(cited)} cited of {len(a.sources)} retrieved)", expanded=not a.refused):
            chunk_text = {c.id: c.text for c in service.retriever(corpus).chunks}
            for s in a.sources:
                mark = "✅" if s["n"] in cited else "▫️"
                st.markdown(f"{mark} **[{s['n']}] {s['section']}**, page {s['page']}  \n"
                            f"<small>{s['doc_title']} › {s['path']} · matched by {', '.join(s['retrieval'])}</small>",
                            unsafe_allow_html=True)
                if s["n"] in cited:
                    st.text(chunk_text[s["chunk_id"]][:1200])
        st.session_state.history.append({"q": question, "md": answer_md})
