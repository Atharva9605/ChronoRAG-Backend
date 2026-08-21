import hashlib
import re
from pathlib import Path

import fitz  # pymupdf

from .config import settings


def extract_pages(pdf_path: Path) -> list[str]:
    """Return page text, index 0 = page 1."""
    doc = fitz.open(pdf_path)
    try:
        return [page.get_text("text") or "" for page in doc]
    finally:
        doc.close()


def doc_id_for(pdf_path: Path) -> str:
    h = hashlib.sha1(pdf_path.read_bytes()).hexdigest()[:16]
    return f"doc_{h}"


# ------------------------------------------------------------
# PART A chunking: character windows, page-boundary agnostic.
# This is the honest naive baseline — it does NOT know about pages.
# ------------------------------------------------------------
def naive_chunks(pages: list[str]) -> list[dict]:
    size = settings.naive_chunk_chars
    overlap = settings.naive_chunk_overlap

    # Build one long string, remembering where each page starts
    offsets: list[tuple[int, int]] = []   # (char_offset, page_no)
    buf: list[str] = []
    cursor = 0
    for i, text in enumerate(pages, start=1):
        offsets.append((cursor, i))
        buf.append(text)
        cursor += len(text) + 1
    full = "\n".join(buf)

    def page_at(pos: int) -> int:
        lo, hi, ans = 0, len(offsets) - 1, 1
        while lo <= hi:
            mid = (lo + hi) // 2
            if offsets[mid][0] <= pos:
                ans = offsets[mid][1]
                lo = mid + 1
            else:
                hi = mid - 1
        return ans

    chunks: list[dict] = []
    start = 0
    idx = 0
    step = max(1, size - overlap)
    while start < len(full):
        end = min(start + size, len(full))
        content = full[start:end].strip()
        if content:
            chunks.append({
                "chunk_index": idx,
                "page_start": page_at(start),
                "page_end": page_at(max(start, end - 1)),
                "content": content,
            })
            idx += 1
        if end == len(full):
            break
        start += step
    return chunks


# ------------------------------------------------------------
# PART B windowing: page-preserving sliding windows with
# explicit [PAGE N] markers injected into the text stream.
# ------------------------------------------------------------
def sliding_windows(pages: list[str]) -> list[dict]:
    size = settings.window_size
    overlap = settings.window_overlap
    windows: list[dict] = []
    start = 0
    n = len(pages)
    while start < n:
        end = min(start + size, n)
        body = "\n".join(
            f"[PAGE {start + i + 1}]\n{pages[start + i]}"
            for i in range(end - start)
        )
        windows.append({
            "id": f"win_{start + 1}_{end}",
            "start": start + 1,
            "end": end,
            "text": body.strip(),
        })
        if end == n:
            break
        start += max(1, size - overlap)
    return windows


PAGE_RE = re.compile(r"\[PAGE\s+([0-9,\s]+)\]")


def parse_pages_from_bullet(line: str) -> list[int]:
    """Extract page numbers from a '[PAGE 8, 9] text' observation bullet."""
    m = PAGE_RE.search(line)
    if not m:
        return []
    return sorted({int(x) for x in re.findall(r"\d+", m.group(1))})
