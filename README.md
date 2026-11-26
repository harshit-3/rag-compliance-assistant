# Compliance RAG Assistant: cited answers over mine-safety law & clinical guidelines

[![CI](https://github.com/harshit-3/rag-compliance-assistant/actions/workflows/ci.yml/badge.svg)](https://github.com/harshit-3/rag-compliance-assistant/actions)

A retrieval-augmented assistant that answers questions about **WA mine-safety regulations**
(750-page *Work Health and Safety (Mines) Regulations 2022* + a Safe Work Australia Code of Practice)
or **Australian infection-control guidelines** (318 pages), with **every claim cited to the exact
regulation or guideline section and page**. It refuses when the documents don't cover the question.

It runs on **Google Gemini** (cloud) or a **local Qwen3-8B via Ollama** (fully offline, for sites
that can't send documents to a third party), and it's **evaluated, not eyeballed**: 121 questions
with gold sections, a retrieval ablation and LLM-judged answer quality.

## Results

### Retrieval: each stage earns its place

| Retriever | Mining R@1 | Mining R@5 | Mining R@10 | Mining MRR | Health R@1 | Health R@5 | Health R@10 | Health MRR | Latency p50 |
|---|---|---|---|---|---|---|---|---|---|
| BM25 | 0.50 | 0.75 | 0.83 | 0.60 | 0.62 | 0.87 | 0.89 | 0.71 | 2 ms |
| Dense (bge-base) | 0.58 | 0.90 | 0.93 | 0.71 | 0.78 | 0.93 | 0.93 | 0.85 | 11 ms |
| Hybrid (RRF) | 0.67 | 0.88 | 0.92 | 0.74 | 0.78 | 0.93 | 0.96 | 0.83 | 14 ms |
| **Hybrid + reranker** | **0.73** | **0.93** | **1.00** | **0.81** | **0.76** | **0.98** | **1.00** | **0.85** | 300 ms |

*60 mining + 45 healthcare questions with gold sections. Latency measured on an RTX 4070 Laptop.*

On mining, the reranker lifts top-1 accuracy from 0.50 (BM25) / 0.58 (dense) to **0.73** and finds the gold section in the top 10 for **every** question. On healthcare, dense retrieval alone is already strong (long prose guidelines suit embeddings) and the reranker trades a little top-1 (0.78 → 0.76) for the best Recall@5 (0.98) and Recall@10 (1.00). Since 6 sources go to the LLM, Recall@5–10 is what matters.

**BM25 still earns its place** in legal text: for exact terms like "competent assessor" or
"principal mining hazard", dense embeddings drift to semantically similar but wrong regulations
(e.g. surveyor certificates), and fusion recovers them.

### Answers (hybrid + rerank, 6 sources, LLM-as-judge vs reference answers)

| Corpus | Model | Cites gold section | Fully correct | Faithful | Out-of-scope refused | False refusals | Generation p50 |
|---|---|---|---|---|---|---|---|
| Mining | Gemini 3.5 Flash-Lite | 90% | 96% | 96% | 8/8 | 5% | 2.7 s |
| Mining | Qwen3-8B (local, Ollama) | 88% | 95% | 93% | 8/8 | 3% | 8.5 s |
| Healthcare | Gemini 3.5 Flash-Lite | 89% | 98% | 100% | 8/8 | 4% | 2.8 s |
| Healthcare | Qwen3-8B (local, Ollama) | 87% | 96% | 89% | 8/8 | 0% | 14.9 s |

- **Cites gold section:** the answer cites at least one chunk from the section the question was written from.
- **Fully correct:** the judge scores the answer 2/2 against the reference answer.
- **Faithful:** every claim is supported by the sources the answer actually cites.
- **Refusal:** correct refusals on 16 deliberately out-of-scope questions (wages, royalties, drug
  doses…). Wrong refusals on in-scope questions are reported separately.

**The local 8B model is within 1–2 points of the cloud model on correctness.** For a mine operator
that can't send internal documents to a third-party API, that changes the deployment decision. The
gap is in **faithfulness on long clinical prose** (89% vs 100%): Qwen occasionally adds a detail that
isn't in the cited source. That argues for the citation validator plus a stricter prompt, or a larger
local model, before clinical use. It runs about 3–5× slower on a laptop GPU.

## How it works

```
PDF ─▶ bookmark-tree sectioning ─▶ clean chunks ─┬─▶ BM25 index ───────┐
       (regulation = chunk,        + hierarchy   └─▶ BGE embeddings ──┤ RRF fusion ─▶ cross-encoder ─▶ top-k
        headers/footers removed)     path + page        (GPU/CPU)      │              rerank
                                                                       │
question ─▶ "regulation 625B" reference resolver ──────────────────────┘ (pinned)
                                                                              ▼
              cited answer ◀── citation validator ◀── Gemini / Qwen3 (Ollama) ◀── numbered sources
```

**Structure-aware ingestion** ([`ingest.py`](src/ragkit/ingest.py)): uses each PDF's bookmark tree so
a chunk is a real unit, such as *"617. Managing risks to health and safety"* with its path
*Chapter 10 › Part 10.2 › Subdivision 1*, not an arbitrary 512-token window. It:
- strips running headers and footers (WA legislation `[PCO …]` headers, "34 of 318" page counters)
- keeps subregulation and list structure, so `(1)`, `(a)` and `Penalty:` labels survive
- does not split inline references ("subregulation (1) is…")
- skips "Not used" regulations and back matter

These details matter: an early version silently dropped every `(1)` and `Penalty:` label because
they "repeat on every page", which changes the meaning of the law.

**Retrieval** ([`retrieval.py`](src/ragkit/retrieval.py)):
- **Hybrid search:** BM25 and `BAAI/bge-base-en-v1.5` are fused with Reciprocal Rank Fusion, then
  reranked with `BAAI/bge-reranker-base`.
- **Contextual chunk headers:** the document title and section path are embedded with the text.
- **Reference resolver:** questions naming "reg 625B" or "regulation 617" pin that regulation.

**Grounded generation** ([`rag.py`](src/ragkit/rag.py)): the model sees numbered sources and must
cite them. A validator then:
- strips citation numbers that point to no source (hallucinated citations)
- flags answers that cite nothing
- detects refusals
- skips the LLM call entirely when retrieval returns nothing

**Resilience** ([`llm.py`](src/ragkit/llm.py)): the Gemini client retries 429/5xx errors with
backoff and falls back across models. This was built after the hosted Flash models returned sustained
`503 high demand` errors during development.

## Evaluation methodology

- **Synthetic questions (85).** Gemini writes a realistic question from a randomly sampled section,
  told to paraphrase and never quote section numbers or titles. A second strict LLM pass keeps only
  questions that the passage fully answers and that aren't generic. The gold label is the source section.
- **Hand-written questions (20).** These are written as a mine supervisor or nurse would ask them,
  with gold sections looked up manually. Some accept several valid sections.
- **Out-of-scope questions (16)** that sound plausible but aren't covered (FIFO wages, lithium
  royalties, amoxicillin dosing, AHPRA registration), to measure whether the system refuses.
- Retrieval hits are matched at **section level**, so any chunk of a split section counts.
- **Judge caveat:** the judge is `gemini-3.5-flash-lite`, the same family as one of the answer
  models, so self-preference bias is possible. Qwen's scores come from a different model family, and
  the retrieval and citation metrics are judge-free.
- Every LLM output is cached in `eval/cache/`, so results are reproducible and reruns are free.

## Quick start

```bash
pip install -r requirements.txt
cp .env.example .env                       # add GEMINI_API_KEY (or use Ollama only)
python scripts/download_corpora.py         # fetches PDFs from the official publishers
python scripts/build_index.py              # chunk + embed (~1 min on GPU)
streamlit run app.py                       # chat UI with citations
PYTHONPATH=src uvicorn ragkit.api:app      # REST API, docs at /docs
```

Local model: install [Ollama](https://ollama.com), run `ollama pull qwen3:8b`, then choose
"Qwen3 8B (local)" in the UI or pass `"backend": "ollama:qwen3:8b"`.

```bash
curl -X POST localhost:8000/ask -H "Content-Type: application/json" \
  -d '{"question": "How long must ventilation testing records be kept?", "corpus": "mining"}'
```

```json
{"answer": "Records of all monitoring and testing of the ventilation system for an underground mine must be kept for at least 7 years [1]. ...",
 "citations": [{"n": 1, "section": "653. Monitoring and testing of ventilation system", "page": 499,
                "path": "Chapter 10 — Mines > Part 10.2 — Managing risks > ...", "retrieval": {"bm25": 1, "dense": 1}}],
 "grounded": true, "refused": false, "model": "gemini-3.5-flash-lite", "timings": {"retrieval_s": 0.31, "generation_s": 2.7}}
```

**Reproduce the evaluation:**

```bash
python eval/build_eval_set.py --corpus mining --n 50          # synthetic questions (cached)
python eval/run_eval.py --corpus mining --retrieval --answers gemini ollama:qwen3:8b
```

**Add your own knowledge base:** add an entry to [`configs/corpora.yaml`](configs/corpora.yaml) with PDF URLs.
Any PDF with bookmarks gets structure-aware chunking automatically.

## Limitations

- The answers are information, not legal or medical advice. The regulations change, so the corpus is
  pinned to a version (WA Regs `02-b0-00`, in force from 11 Nov 2026) and should be re-ingested on
  amendment.
- Tables in the PDFs are flattened to text. Table-heavy questions (e.g. exposure-standard tables)
  would benefit from table-aware parsing.
- Evaluation questions are mostly synthetic. The hand-written and out-of-scope sets are small (36).
- There's no conversation memory: each question is answered independently.

## Sources

- *Work Health and Safety (Mines) Regulations 2022* (WA), Parliamentary Counsel's Office, legislation.wa.gov.au
- *Model Code of Practice: Managing the risks of plant in the workplace*, Safe Work Australia (CC BY 4.0)
- *Australian Guidelines for the Prevention and Control of Infection in Healthcare*, ACSQHC / NHMRC

Documents are downloaded from the publishers by `scripts/download_corpora.py` and are not
redistributed in this repository.
