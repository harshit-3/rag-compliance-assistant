"""PDF -> structure-aware chunks.

Uses each PDF's bookmark tree (table of contents) to split documents along their real
structure: a regulation, a guideline section, a code-of-practice clause. Every chunk keeps its
hierarchy path (e.g. "Chapter 10 > Part 10.2 > 617. Managing risks to health and safety") and
page number, so answers can cite an exact regulation instead of "page 436-ish".
"""
from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

import pymupdf

# Lines that are pure page furniture in WA legislation PDFs.
_RUNNING_HEADER_END = re.compile(r"^\[PCO [\w-]+\]$")
# Break before list items only at clause boundaries ("...: (a) ...", "...; (b) ..."), not inline
# references like "subregulation (1) is ...".
_LIST_MARKER = re.compile(r"(?:(?<=[.:;\u2014])|^)\s*(?=\((?:\d+[A-Z]?|[a-z]{1,2}|[ivx]{1,5}|[A-Z])\)\s)", re.M)
_LEGIS_FURNITURE = re.compile(r"^(Published on www\.legislation\.wa\.gov\.au|Official Version|As at \d{1,2} \w+ \d{4}|"
                              r"page [ivxlc\d]+|\[PCO [\w-]+\])$", re.I)
_SKIP_TITLE = re.compile(r"^\s*(\d+[A-Z]*\.\s*)?Not used\s*$|^table of contents$|^contents$|^sections$", re.I)


@dataclass
class Chunk:
    id: str
    corpus: str
    doc_id: str
    doc_title: str
    section: str  # leaf heading, e.g. "617. Managing risks to health and safety"
    path: str  # full hierarchy
    page: int  # 1-based PDF page where the section starts
    text: str  # body text (without the contextual header)

    @property
    def embed_text(self) -> str:
        """Text used for indexing: contextual header + body (improves retrieval of short sections)."""
        return f"{self.doc_title}\n{self.path}\n\n{self.text}"


def _clean_title(t: str) -> str:
    return re.sub(r"\s+", " ", t).strip()


def _page_lines(doc: pymupdf.Document, legislation_headers: bool) -> list[list[str]]:
    pages = []
    for page in doc:
        lines = [ln.strip() for ln in page.get_text().splitlines()]
        if legislation_headers:
            for i, ln in enumerate(lines[:25]):
                if _RUNNING_HEADER_END.match(ln):
                    lines = lines[i + 1:]
                    break
        pages.append([ln for ln in lines if ln])
    if legislation_headers:  # running header already removed precisely; keep "(1)", "Penalty:" etc.
        return [[ln for ln in p if not _LEGIS_FURNITURE.match(ln)] for p in pages]
    # Other PDFs: drop running titles/footers = lines repeated on >30% of pages. Never drop list
    # markers like "(a)" or labels ending in ":" - they repeat because the document is structured.
    def key(ln: str) -> str:  # "Guidelines ... 34 of 318" and "... 35 of 318" are the same footer
        return re.sub(r"\d+", "#", ln)

    counts = Counter(key(ln) for p in pages for ln in set(p))
    n = len(pages)
    boiler = {k for k, c in counts.items()
              if not re.fullmatch(r"[#.\s]+", k)  # bare section numbers like "3.1.1" are structure
              and ((c > 0.3 * n and 12 <= len(k) < 160 and not k.endswith(":"))
                   or (c > 0.8 * n and "#" in k))}  # page counters like "34 of 318"
    return [[ln for ln in p if key(ln) not in boiler and not re.fullmatch(r"\d{1,4}", ln)] for p in pages]


def _normalise(text: str) -> str:
    text = re.sub(r"[ \t ]+", " ", text.replace("\n", " ")).strip()
    return _LIST_MARKER.sub("\n", text).strip()  # (1) / (a) / (i) list items on their own lines


def _find_heading(lines: list[str], title: str, start: int) -> int:
    """Index of the line where `title` begins (titles may be split across lines in the PDF)."""
    words = title.split()
    if not words:
        return start
    head = " ".join(words[:4]).lower()
    for i in range(start, len(lines)):
        window = " ".join(lines[i:i + 4]).lower()
        window = re.sub(r"\s+", " ", window)
        if window.startswith(head) or (words[0].endswith(".") and lines[i] == words[0]
                                       and head.split(" ", 1)[-1] in window):
            return i
    return -1


def extract_sections(pdf_path: Path, doc_id: str, doc_title: str, corpus: str,
                     exclude_top: tuple[str, ...] = (), stop_at: str | None = None) -> list[Chunk]:
    doc = pymupdf.open(pdf_path)
    toc = [(lvl, _clean_title(t), p) for lvl, t, p in doc.get_toc() if p > 0]
    if stop_at:  # drop back matter (references, evidence annexes) from the first matching top-level entry
        cut = next((k for k, (lvl, t, _) in enumerate(toc) if lvl == 1 and t.lower().startswith(stop_at.lower())),
                   None)
        if cut is not None:
            toc = toc[:cut] + [(1, "__END__", toc[cut][2])]
    # WA legislation PDFs carry a "[PCO xx-xx-xx]" running header on every page.
    legislation = any("[PCO " in doc[i].get_text() for i in range(0, min(doc.page_count, 60), 10))
    pages = _page_lines(doc, legislation_headers=legislation)

    # Flatten pages into one stream with a page index per line.
    stream, page_of = [], []
    for pno, lines in enumerate(pages):
        stream.extend(lines)
        page_of.extend([pno] * len(lines))
    page_start = {}
    for i, p in enumerate(page_of):
        page_start.setdefault(p, i)

    # Locate each TOC heading in the stream (monotonic search).
    anchors, cursor, stack, skipping = [], 0, [], False
    for lvl, title, p in toc:
        stack = stack[:lvl - 1] + [title]
        if lvl == 1:
            skipping = any(title.lower().startswith(x.lower()) for x in exclude_top)
        pos_start = max(cursor, page_start.get(p - 1, cursor))
        idx = _find_heading(stream, title, pos_start)
        if idx < 0:
            idx = pos_start
        anchors.append((idx, lvl, title, " > ".join(stack), p, skipping))
        cursor = idx
    anchors.append((len(stream), 0, "", "", doc.page_count, True))

    chunks = []
    for (i, _lvl, title, path, p, skip), (j, *_rest) in zip(anchors, anchors[1:], strict=False):
        if skip or title == "__END__" or _SKIP_TITLE.match(title):
            continue
        body_lines = stream[i:j]
        # Remove the heading lines themselves from the body.
        consumed, joined = 0, ""
        for k, ln in enumerate(body_lines[:6]):
            joined = (joined + " " + ln).strip()
            if re.sub(r"\s+", " ", joined).lower() == title.lower():
                consumed = k + 1
                break
        body = _normalise("\n".join(body_lines[consumed:]))
        if len(body.split()) < 8:  # heading-only nodes (e.g. "Part 10.2 — Managing risks")
            continue
        chunks.append(Chunk(id="", corpus=corpus, doc_id=doc_id, doc_title=doc_title, section=title,
                            path=path, page=p, text=body))
    return chunks


def split_long(chunk: Chunk, max_words: int = 350, overlap: int = 50) -> list[Chunk]:
    """Split oversized sections on line (list-item) boundaries with overlap."""
    words = chunk.text.split()
    if len(words) <= max_words:
        return [chunk]
    lines = chunk.text.split("\n")
    parts, cur = [], []
    for ln in lines:
        if cur and len(" ".join(cur).split()) + len(ln.split()) > max_words:
            parts.append(cur)
            tail = " ".join(cur).split()[-overlap:]
            cur = [" ".join(tail)]
        cur.append(ln)
    if cur:
        parts.append(cur)
    out = []
    for n, part in enumerate(parts):
        text = "\n".join(part)
        # A single huge line (no list structure): hard-wrap by words.
        ws = text.split()
        for s in range(0, max(len(ws), 1), max_words - overlap):
            piece = " ".join(ws[s:s + max_words])
            if piece:
                suffix = f" (part {n + 1})" if len(parts) > 1 else ""
                out.append(Chunk(**{**asdict(chunk), "section": chunk.section + suffix, "text": piece}))
            if s + max_words >= len(ws):
                break
    return out


def build_corpus(corpus: str, spec: dict, raw_dir: Path) -> list[Chunk]:
    chunks: list[Chunk] = []
    for d in spec["documents"]:
        secs = extract_sections(raw_dir / corpus / f"{d['id']}.pdf", d["id"], d["title"], corpus,
                                tuple(d.get("exclude_sections", [])), d.get("stop_at"))
        for c in secs:
            chunks.extend(split_long(c))
    for n, c in enumerate(chunks):
        c.id = f"{c.doc_id}#{n:05d}"
    return chunks


def save_chunks(chunks: list[Chunk], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(asdict(c), ensure_ascii=False) + "\n")


def load_chunks(path: Path) -> list[Chunk]:
    with path.open(encoding="utf-8") as f:
        return [Chunk(**json.loads(line)) for line in f]
