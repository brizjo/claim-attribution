"""
Output store — persists every intermediate result to JSONL on disk.

Files are append-only (one JSON object per line).  Each record carries a
timestamp and is indexed by source_id + extractor so results can be traced
back to their origin.

Produced files (under settings.OUTPUT_DIR):

    coref_resolved.jsonl        original + resolved text per passage
    triples_extracted.jsonl     each (S, P, O) + claim_span + source_id
    attribution_results.jsonl   claim attribution outcomes
    ingest_reports.jsonl        per-question ingestion summaries
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from config import settings

logger = logging.getLogger(__name__)


def _ensure_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _append(path: Path, record: dict) -> None:
    """Append a single JSON line to *path*, creating file if needed."""
    _ensure_dir(path)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def _timestamp() -> str:
    return datetime.now().isoformat(timespec="seconds")


# ── Public API ────────────────────────────────────────────────────────


_BASE = Path(settings.OUTPUT_DIR)


def save_coref(
    source_id: str,
    sample_id: str,
    title: str,
    chunk_index: int,
    original_text: str,
    resolved_text: str,
) -> None:
    """Persist one coref-resolution result."""
    _append(_BASE / "coref_resolved.jsonl", {
        "source_id": source_id,
        "sample_id": sample_id,
        "title": title,
        "chunk_index": chunk_index,
        "original_text": original_text,
        "resolved_text": resolved_text,
        "timestamp": _timestamp(),
    })


def save_triple(
    source_id: str,
    sample_id: str,
    extractor: str,
    subject: str,
    predicate: str,
    obj: str,
    claim_span: str,
    chunk_text: str,
    title: str,
    chunk_index: int,
) -> None:
    """Persist one extracted triple."""
    _append(_BASE / "triples_extracted.jsonl", {
        "source_id": source_id,
        "sample_id": sample_id,
        "extractor": extractor,
        "subject": subject,
        "predicate": predicate,
        "object": obj,
        "claim_span": claim_span,
        "chunk_text": chunk_text,
        "title": title,
        "chunk_index": chunk_index,
        "timestamp": _timestamp(),
    })


def save_triples_batch(
    triples: list[dict],
    sample_id: str,
    extractor: str,
) -> None:
    """Persist a batch of extracted triples at once."""
    path = _BASE / "triples_extracted.jsonl"
    _ensure_dir(path)
    ts = _timestamp()
    with path.open("a", encoding="utf-8") as fh:
        for t in triples:
            record = {
                "source_id": t.get("source_id", ""),
                "sample_id": sample_id,
                "extractor": extractor,
                "subject": t.get("subject", ""),
                "predicate": t.get("predicate", ""),
                "object": t.get("obj", t.get("object", "")),
                "claim_span": t.get("claim_span", ""),
                "chunk_text": t.get("chunk_text", ""),
                "title": t.get("title", t.get("source_file", "")),
                "chunk_index": t.get("chunk_index", 0),
                "timestamp": ts,
            }
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def save_attribution(
    claim: str,
    subject: str,
    predicate: str,
    obj: str,
    match_type: str,
    similarity: float,
    source_chunk: str,
    source_id: str,
    extractor: str,
    is_question: bool = False,
) -> None:
    """Persist one attribution/verification result."""
    _append(_BASE / "attribution_results.jsonl", {
        "claim": claim,
        "parsed_triple": {"subject": subject, "predicate": predicate, "object": obj},
        "match_type": match_type,
        "similarity": similarity,
        "source_chunk": source_chunk,
        "source_id": source_id,
        "extractor": extractor,
        "is_question": is_question,
        "timestamp": _timestamp(),
    })


def save_ingest_report(
    sample_id: str,
    question: str,
    extractor: str,
    total_triples: int,
    docs_processed: int,
    docs_skipped: int,
    zero_triple_docs: int,
    errors: list[str],
) -> None:
    """Persist one per-question ingestion report."""
    _append(_BASE / "ingest_reports.jsonl", {
        "sample_id": sample_id,
        "question": question,
        "extractor": extractor,
        "total_triples": total_triples,
        "docs_processed": docs_processed,
        "docs_skipped": docs_skipped,
        "zero_triple_docs": zero_triple_docs,
        "errors": errors,
        "timestamp": _timestamp(),
    })


# ── Read helpers (for loading results back) ───────────────────────────


def load_jsonl(filename: str) -> list[dict]:
    """Read all records from a JSONL file under OUTPUT_DIR."""
    path = _BASE / filename
    if not path.is_file():
        return []
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                logger.warning("Skipping malformed JSONL line in %s", filename)
    return records
