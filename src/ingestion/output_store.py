"""
Output store — persists every intermediate result to JSONL on disk.

Files are append-only (one JSON object per line).  Each record carries a
timestamp and is indexed by source_id + extractor so results can be traced
back to their origin.

Produced files (under settings.OUTPUT_DIR):

    coref_resolved.jsonl        original + resolved text per passage
    triples_extracted.jsonl     each (S, P, O) + claim_span + source_id
    triples_discarded.jsonl     triple bocciate dai guardrail della pipeline
                                principale (+ discard_reason, stage)
    generated_answers.jsonl     risposte del generatore grounded (domanda,
                                risposta, modello, source_id dei passaggi)
    attribution_results.jsonl   claim attribution outcomes
    ingest_reports.jsonl        per-question ingestion summaries
    canonicalization.jsonl      una riga per menzione: forma originale -> forma
                                canonica, stadio (1-4), confidenza, external_id

Pipeline ibrida REBEL+DeepSeek (nessun campo `extractor`: la pipeline è una
sola, `origin` dice solo quale modello ha proposto la tripla):

    triples_hybrid.jsonl            triple sopravvissute ai guardrail
    triples_hybrid_discarded.jsonl  triple scartate + motivo del guardrail
    hybrid_runs.jsonl               statistiche per run (variante A / B)
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


def save_generated_answer(
    sample_id: str,
    question: str,
    answer: str,
    model: str,
    seconds: float,
    passages: list[dict],
) -> None:
    """
    Persiste una risposta del generatore grounded (studio <claim, context>,
    stadio 1).  Dei passaggi si salvano source_id e title: il testo integrale
    e' recuperabile dal corpus per source_id e sta gia' in coref_resolved.jsonl.
    """
    _append(_BASE / "generated_answers.jsonl", {
        "sample_id": sample_id,
        "question": question,
        "answer": answer,
        "model": model,
        "seconds": round(seconds, 3),
        "passages": [
            {"source_id": p.get("source_id", ""),
             "title": p.get("title", p.get("source_file", ""))}
            for p in passages
        ],
        "timestamp": _timestamp(),
    })


def save_discarded_triples(records: list[dict]) -> int:
    """
    Persiste le triple scartate dai guardrail della pipeline PRINCIPALE
    (una riga per tripla, con `discard_reason` e `stage`: "extract" se
    bocciata alla prima passata, "repair" se bocciata anche dopo la
    riparazione DeepSeek).  File separato da `triples_hybrid_discarded.jsonl`,
    che appartiene agli esperimenti.
    """
    if not records:
        return 0
    path = _BASE / "triples_discarded.jsonl"
    _ensure_dir(path)
    ts = _timestamp()
    with path.open("a", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps({**record, "timestamp": ts}, ensure_ascii=False) + "\n")
    return len(records)


# ── Canonicalizzazione delle entita' (fase pre-write) ─────────────────


def save_canonicalization(records: list[dict]) -> int:
    """
    Persiste l'esito della cascata di canonicalizzazione: una riga per
    menzione.  E' il deliverable dell'analisi (quante menzioni si chiudono a
    ciascuno stadio), non un log accessorio — vedi
    `scripts/analyze_canonicalization.py`.
    """
    if not records:
        return 0
    path = _BASE / "canonicalization.jsonl"
    _ensure_dir(path)
    ts = _timestamp()
    with path.open("a", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps({**record, "timestamp": ts}, ensure_ascii=False) + "\n")
    return len(records)


# ── Pipeline ibrida (REBEL + DeepSeek) ────────────────────────────────


def _hybrid_record(
    t, passage, sample_id: str, variant: str, ts: str, with_reason: bool = False,
) -> dict:
    """Un record JSONL da una `HybridTriple` + il suo `PassageResult`."""
    record = {
        "sample_id": sample_id,
        "variant": variant,
        "source_id": passage.source_id,
        "title": passage.title,
        "chunk_index": passage.chunk_index,
        "sentence_index": t.sentence_index,
        "subject": t.subject,
        "predicate": t.predicate,
        "object": t.obj,
        "claim_span": t.claim_span,
        "sentence": t.sentence,
        "chunk_text": passage.original_text,
        "origin": t.origin,
        "rebel_predicate": t.rebel_predicate,
        "timestamp": ts,
    }
    if with_reason:
        record["discard_reason"] = t.reason
    return record


def save_hybrid_report(report) -> dict[str, int]:
    """
    Persiste un `RunReport` della pipeline ibrida: triple sopravvissute,
    triple scartate (con motivo) e statistiche di run.

    Ritorna i conteggi scritti — utili per la conferma in UI.
    """
    ts = _timestamp()
    survived_path = _BASE / "triples_hybrid.jsonl"
    discarded_path = _BASE / "triples_hybrid_discarded.jsonl"
    _ensure_dir(survived_path)

    n_survived = n_discarded = 0
    with survived_path.open("a", encoding="utf-8") as ok_fh, \
            discarded_path.open("a", encoding="utf-8") as ko_fh:
        for passage in report.passages:
            for t in passage.survived:
                ok_fh.write(json.dumps(
                    _hybrid_record(t, passage, report.sample_id, report.variant, ts),
                    ensure_ascii=False) + "\n")
                n_survived += 1
            for t in passage.discarded:
                ko_fh.write(json.dumps(
                    _hybrid_record(t, passage, report.sample_id, report.variant, ts,
                                   with_reason=True),
                    ensure_ascii=False) + "\n")
                n_discarded += 1

    _append(_BASE / "hybrid_runs.jsonl", {
        "sample_id": report.sample_id,
        "question": report.question,
        "variant": report.variant,
        "passages": len(report.passages),
        "produced": report.produced,
        "survived": report.survived,
        "discarded": report.produced - report.survived,
        "rebel_produced": report.rebel_produced,
        "rebel_matched": report.rebel_matched,
        "rebel_kept": report.rebel_kept,
        "vocabulary_size": len(report.vocabulary),
        "rebel_rejected": report.rebel_rejected,
        "discard_reasons": dict(report.discard_reasons),
        "llm_calls": report.llm_calls,
        "seconds": round(report.seconds, 3),
        "errors": report.errors,
        "timestamp": ts,
    })

    return {"survived": n_survived, "discarded": n_discarded}


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
