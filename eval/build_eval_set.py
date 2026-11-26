"""Build the evaluation set: synthetic questions (LLM-written, LLM-filtered) + hand-written +
out-of-scope questions. Every question carries its gold section(s).

    python eval/build_eval_set.py --corpus mining --n 50
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ragkit.ingest import load_chunks  # noqa: E402
from ragkit.llm import GeminiLLM  # noqa: E402

GEN_SYSTEM = """You write evaluation questions for a retrieval-augmented assistant used by {persona}.
Given a passage, write ONE realistic question that this passage answers, as that person would
actually ask it at work. Rules:
- Paraphrase: do NOT copy distinctive phrases or section titles from the passage.
- Do NOT mention regulation/section numbers, the document name, or "the passage".
- The question must be specific enough that this passage (not general knowledge) is needed.
- Also give a short reference answer (1-2 sentences) taken from the passage.
Return JSON: {{"question": "...", "answer": "..."}}"""

CHECK_SYSTEM = """You are a strict reviewer of evaluation questions. Given a passage and a question,
answer in JSON: {"answerable": true/false, "specific": true/false}
- answerable: the passage alone fully answers the question.
- specific: the question is not vague/generic and could not be answered without this kind of document."""

PERSONA = {"mining": "mine supervisors, safety officers and mine operators in Western Australia",
           "healthcare": "nurses, infection control practitioners and hospital managers in Australia"}


def base_section(path: str) -> str:
    return re.sub(r" \(part \d+\)$", "", path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()

    chunks = load_chunks(ROOT / "data" / "chunks" / f"{a.corpus}.jsonl")
    # Sample passages with real content (skip very short definitional fragments).
    pool = [c for c in chunks if len(c.text.split()) >= 60]
    random.Random(a.seed).shuffle(pool)
    llm = GeminiLLM()
    out_path = ROOT / "eval" / f"synthetic_{a.corpus}.jsonl"
    done = {json.loads(line)["chunk_id"] for line in out_path.open(encoding="utf-8")} if out_path.exists() else set()
    kept = len(done)
    with out_path.open("a", encoding="utf-8") as f:
        for c in pool:
            if kept >= a.n:
                break
            if c.id in done:
                continue
            passage = f"{c.path}\n{c.text}"
            try:
                q = json.loads(llm.generate(GEN_SYSTEM.format(persona=PERSONA[a.corpus]), passage, 0.7,
                                            json_mode=True).text)
                chk = json.loads(llm.generate(CHECK_SYSTEM, f"Passage:\n{passage}\n\nQuestion: {q['question']}", 0.0,
                                              json_mode=True).text)
            except Exception as e:  # noqa: BLE001 - keep going; log and skip this passage
                print("skip", c.id, str(e)[:120])
                continue
            ok = chk.get("answerable") and chk.get("specific")
            print(("KEEP " if ok else "drop ") + q["question"][:110])
            if not ok:
                continue
            f.write(json.dumps({"id": f"{a.corpus}-syn-{kept:03d}", "corpus": a.corpus, "type": "synthetic",
                                "question": q["question"], "reference_answer": q["answer"], "chunk_id": c.id,
                                "gold_sections": [f"{c.doc_id}::{base_section(c.path)}"]}, ensure_ascii=False) + "\n")
            f.flush()
            kept += 1
    print(f"{kept} synthetic questions in {out_path.name}")


if __name__ == "__main__":
    main()
