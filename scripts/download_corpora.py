"""Download the source PDFs listed in configs/corpora.yaml into data/raw/<corpus>/.

    python scripts/download_corpora.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ragkit import service  # noqa: E402

for corpus in service.CONFIG:
    service.download_pdfs(corpus)
    print(corpus, "ok")
