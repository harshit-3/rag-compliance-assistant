"""Evaluate retrieval and answer quality.

    python eval/run_eval.py --corpus mining --retrieval            # retrieval ablation only (fast, no LLM)
    python eval/run_eval.py --corpus mining --answers gemini ollama  # + generation & LLM-judge

Retrieval: Recall@{1,3,5,10} and MRR@10 per mode (bm25 / dense / hybrid / hybrid_rerank).
A hit = a retrieved chunk from a gold section (same document + section path, any part).

Answers (hybrid_rerank, k=6):
  gold_citation  - answer cites >= 1 chunk from a gold section
  correct        - LLM judge vs reference answer: 2 = fully, 1 = partially, 0 = wrong/missing
  faithful       - LLM judge: every claim is supported by the sources the answer cites
  false_refusal  - refused an in-scope question
  oos_refusal    - correctly refused an out-of-scope question
All LLM outputs are cached in eval/cache/, so reruns only pay for new work.
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ragkit.ingest import load_chunks  # noqa: E402
from ragkit.llm import GeminiLLM, LLMError, get_llm  # noqa: E402
from ragkit.rag import RAGPipeline  # noqa: E402
from ragkit.retrieval import HybridRetriever  # noqa: E402

DOMAIN = {"mining": "Western Australian mine safety regulations and codes of practice",
          "healthcare": "Australian infection prevention and control guidelines for healthcare"}
MODES = ["bm25", "dense", "hybrid", "hybrid_rerank"]
CACHE = ROOT / "eval" / "cache"

JUDGE_SYSTEM = """You grade answers from a document-grounded assistant. Return JSON only:
{"correct": 0|1|2, "faithful": true|false, "reason": "<one short sentence>"}
- correct: compare the ANSWER with the REFERENCE answer. 2 = contains the key facts of the reference,
  1 = partially correct or missing important details, 0 = wrong, irrelevant or a refusal.
- faithful: true if every factual claim in the ANSWER is supported by the CITED SOURCES text.
  Unsupported or contradicted claims -> false."""


def base(path: str) -> str:
    return re.sub(r" \(part \d+\)$", "", path)


def key(chunk) -> str:
    return f"{chunk.doc_id}::{base(chunk.path)}"


def load_questions(corpus: str) -> list[dict]:
    qs = []
    for f in [ROOT / "eval" / f"synthetic_{corpus}.jsonl", ROOT / "eval" / "handwritten.jsonl"]:
        if f.exists():
            qs += [q for q in map(json.loads, f.open(encoding="utf-8")) if q["corpus"] == corpus]
    return qs


def eval_retrieval(retriever: HybridRetriever, qs: list[dict]) -> dict:
    res = {}
    gold_qs = [q for q in qs if q["gold_sections"]]
    for mode in MODES:
        ranks, lat = [], []
        for q in gold_qs:
            t0 = time.perf_counter()
            hits = retriever.search(q["question"], k=10, mode=mode)
            lat.append(time.perf_counter() - t0)
            gold = set(q["gold_sections"])
            rank = next((i + 1 for i, h in enumerate(hits) if key(h.chunk) in gold), None)
            ranks.append(rank)
        res[mode] = {f"recall@{k}": round(sum(r is not None and r <= k for r in ranks) / len(ranks), 3)
                     for k in (1, 3, 5, 10)}
        res[mode]["mrr@10"] = round(sum(1 / r for r in ranks if r) / len(ranks), 3)
        res[mode]["latency_ms_p50"] = round(statistics.median(lat) * 1000, 1)
        res[mode]["n"] = len(ranks)
    return res


def _cache(name: str) -> tuple[dict, Path]:
    p = CACHE / name
    p.parent.mkdir(parents=True, exist_ok=True)
    data = {}
    if p.exists():
        for line in p.open(encoding="utf-8"):
            d = json.loads(line)
            data[d["qid"]] = d
    return data, p


def judge(judge_llm, q: dict, ans: dict) -> dict:
    cited = "\n\n".join(f"[{s['n']}] {s['text']}" for s in ans["cited_sources"]) or "(no sources cited)"
    user = (f"QUESTION: {q['question']}\nREFERENCE: {q.get('reference_answer', '')}\n\nANSWER: {ans['answer']}\n\n"
            f"CITED SOURCES:\n{cited}")
    try:
        return json.loads(judge_llm.generate(JUDGE_SYSTEM, user, 0.0, json_mode=True).text)
    except (LLMError, json.JSONDecodeError) as e:
        return {"correct": None, "faithful": None, "reason": f"judge error: {str(e)[:100]}"}


def eval_answers(retriever, qs, backend: str, corpus: str, judge_llm) -> dict:
    llm = get_llm(backend)
    rag = RAGPipeline(retriever, llm, DOMAIN[corpus], k=6)
    tag = backend.replace(":", "_")
    answers, apath = _cache(f"answers_{corpus}_{tag}.jsonl")
    judged, jpath = _cache(f"judge_{corpus}_{tag}.jsonl")
    for q in qs:
        if q["id"] not in answers:
            try:
                a = rag.ask(q["question"])
            except LLMError as e:
                print("  LLM error", q["id"], str(e)[:100])
                continue
            by_id = {c.id: c for c in retriever.chunks}
            cited_sources = [{"n": s["n"], "text": by_id[s["chunk_id"]].text} for s in a.citations]
            rec = {"qid": q["id"], "answer": a.answer, "refused": a.refused, "grounded": a.grounded,
                   "cited_keys": [f"{s['doc_id']}::{base(s['path'])}" for s in a.citations],
                   "cited_sources": cited_sources, "model": a.model, "timings": a.timings}
            answers[q["id"]] = rec
            with apath.open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        a = answers[q["id"]]
        if q["type"] != "out_of_scope" and not a["refused"] and q["id"] not in judged and judge_llm:
            j = {"qid": q["id"], **judge(judge_llm, q, a)}
            judged[q["id"]] = j
            with jpath.open("a", encoding="utf-8") as f:
                f.write(json.dumps(j, ensure_ascii=False) + "\n")

    ins = [q for q in qs if q["type"] != "out_of_scope" and q["id"] in answers]
    oos = [q for q in qs if q["type"] == "out_of_scope" and q["id"] in answers]
    js = [judged[q["id"]] for q in ins if q["id"] in judged and judged[q["id"]].get("correct") is not None]
    gen_lat = [answers[q["id"]]["timings"]["generation_s"] for q in ins + oos]
    return {
        "backend": backend, "model": next(iter(answers.values()))["model"] if answers else None,
        "n_in_scope": len(ins), "n_out_of_scope": len(oos),
        "gold_citation": round(sum(bool(set(answers[q["id"]]["cited_keys"]) & set(q["gold_sections"]))
                                   for q in ins) / max(len(ins), 1), 3),
        "grounded": round(sum(answers[q["id"]]["grounded"] for q in ins) / max(len(ins), 1), 3),
        "false_refusal": round(sum(answers[q["id"]]["refused"] for q in ins) / max(len(ins), 1), 3),
        "oos_refusal": round(sum(answers[q["id"]]["refused"] for q in oos) / max(len(oos), 1), 3),
        "correct_full": round(sum(j["correct"] == 2 for j in js) / max(len(js), 1), 3),
        "correct_any": round(sum(j["correct"] >= 1 for j in js) / max(len(js), 1), 3),
        "faithful": round(sum(bool(j["faithful"]) for j in js) / max(len(js), 1), 3),
        "n_judged": len(js),
        "generation_s_p50": round(statistics.median(gen_lat), 2) if gen_lat else None,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--retrieval", action="store_true")
    ap.add_argument("--answers", nargs="*", default=[])
    a = ap.parse_args()

    qs = load_questions(a.corpus)
    print(f"{a.corpus}: {len(qs)} questions "
          f"({sum(q['type'] == 'synthetic' for q in qs)} synthetic, {sum(q['type'] == 'handwritten' for q in qs)} "
          f"handwritten, {sum(q['type'] == 'out_of_scope' for q in qs)} out-of-scope)")
    retriever = HybridRetriever(load_chunks(ROOT / "data" / "chunks" / f"{a.corpus}.jsonl"),
                                cache_dir=ROOT / "data" / "index")
    out_path = ROOT / "eval" / "results" / f"{a.corpus}.json"
    out = json.loads(out_path.read_text()) if out_path.exists() else {}
    out["n_questions"] = len(qs)
    if a.retrieval:
        out["retrieval"] = eval_retrieval(retriever, qs)
        print(json.dumps(out["retrieval"], indent=1))
    if a.answers:
        judge_llm = GeminiLLM(models=("gemini-3.5-flash-lite", "gemini-3.1-flash-lite"))
        out.setdefault("answers", {})
        for backend in a.answers:
            out["answers"][backend] = eval_answers(retriever, qs, backend, a.corpus, judge_llm)
            print(json.dumps(out["answers"][backend], indent=1))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
