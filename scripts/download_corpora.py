"""Download the source PDFs listed in configs/corpora.yaml into data/raw/<corpus>/.

    python scripts/download_corpora.py
"""
from pathlib import Path

import requests
import yaml

ROOT = Path(__file__).resolve().parents[1]
cfg = yaml.safe_load((ROOT / "configs" / "corpora.yaml").read_text())
for corpus, spec in cfg.items():
    for doc in spec["documents"]:
        out = ROOT / "data" / "raw" / corpus / f"{doc['id']}.pdf"
        if out.exists():
            print("exists", out.name)
            continue
        out.parent.mkdir(parents=True, exist_ok=True)
        r = requests.get(doc["url"], timeout=300, headers={"User-Agent": "Mozilla/5.0 (research RAG demo)"})
        r.raise_for_status()
        if not r.content.startswith(b"%PDF"):
            raise ValueError(f"{doc['id']}: not a PDF (content-type {r.headers.get('content-type')})")
        out.write_bytes(r.content)
        print(f"{doc['id']}: {len(r.content) / 1e6:.1f} MB")
