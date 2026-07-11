"""
Document loader — PDF and TXT files → word-chunked dicts.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

from config import settings


def _words(text: str) -> list[str]:
    return text.split()


def _chunk_words(
    words: list[str],
    size: int = settings.CHUNK_SIZE_WORDS,
    overlap: int = settings.CHUNK_OVERLAP_WORDS,
) -> Iterator[str]:
    if not words:
        return
    step = max(1, size - overlap)
    i = 0
    while i < len(words):
        yield " ".join(words[i : i + size])
        i += step


class DocumentLoader:
    """Load PDF or TXT files and yield chunks with metadata."""

    def load(self, file_path: str | Path) -> list[dict]:
        """
        Returns list of dicts:
            {text, source_file, chunk_index, page_num (PDF only)}
        """
        path = Path(file_path)
        suffix = path.suffix.lower()

        if suffix == ".pdf":
            return self._load_pdf(path)
        elif suffix in (".txt", ".md"):
            return self._load_txt(path)
        else:
            raise ValueError(f"Unsupported file type: {suffix}")

    def _load_pdf(self, path: Path) -> list[dict]:
        import fitz  # PyMuPDF

        doc = fitz.open(str(path))
        chunks = []
        idx = 0
        for page_num, page in enumerate(doc):
            text = page.get_text("text").strip()
            if not text:
                continue
            words = _words(text)
            for chunk_text in _chunk_words(words):
                if chunk_text.strip():
                    chunks.append({
                        "text": chunk_text,
                        "source_file": path.name,
                        "chunk_index": idx,
                        "page_num": page_num + 1,
                    })
                    idx += 1
        doc.close()
        return chunks

    def _load_txt(self, path: Path) -> list[dict]:
        text = path.read_text(encoding="utf-8", errors="replace")
        words = _words(text)
        chunks = []
        for idx, chunk_text in enumerate(_chunk_words(words)):
            if chunk_text.strip():
                chunks.append({
                    "text": chunk_text,
                    "source_file": path.name,
                    "chunk_index": idx,
                    "page_num": None,
                })
        return chunks
