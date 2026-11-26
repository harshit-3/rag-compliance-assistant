"""Chunk all corpora and pre-compute embeddings (and warm the reranker cache).

    python scripts/build_index.py
"""
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ragkit import service  # noqa: E402

for corpus in service.CONFIG:
    t0 = time.time()
    r = service.retriever(corpus)
    r.search("warm up", k=1)
    print(f"{corpus}: {len(r.chunks)} chunks indexed in {time.time() - t0:.1f}s")
