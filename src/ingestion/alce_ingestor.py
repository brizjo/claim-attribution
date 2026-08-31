"""
ALCE Ingestor — orchestrates the pipeline on ALCE passages.

For each passage:
    skip-check -> coref -> triple extraction -> anchor claim_span on ORIGINAL
    text -> (optionally) embed predicates -> MERGE into Neo4j -> registry

Two-phase architecture (2026-08-25 refactor):
    1. extract_doc / extract_entry  — coref + extraction, produces DocResult
       with triples but does NOT touch Neo4j.  Output saved to JSONL.
    2. write_doc / write_entry      — takes a DocResult, embeds predicates,
       writes to Neo4j, marks registry.

The legacy ingest_doc / ingest_entry methods remain as convenience wrappers
that call both phases sequentially.

Skip logic (two levels, because neither alone is sufficient):
  * Neo4j (is_source_ingested) — misses docs with zero triples;
  * ProcessedRegistry — local file, unaware if graph was cleared.
A doc is skipped if it is processed in AT LEAST one of the two (skip_existing).
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Callable, Optional, Protocol

from config import settings
from src.graph.neo4j_client import Neo4jClient
from src.ingestion.alce_loader import AlceEntry
from src.ingestion.coref_resolver import CoreferenceResolver
from src.ingestion.graph_writer import GraphWriter
from src.ingestion.output_store import save_coref, save_triples_batch
from src.ingestion.processed_registry import ProcessedRegistry
from src.ingestion.span_matcher import best_span
from src.ingestion.triple_extractor import Triple

logger = logging.getLogger(__name__)


class Extractor(Protocol):
    """Common interface for TripleExtractor (REBEL) and DeepSeekExtractor."""

    name: str

    def extract(self, chunks: list[dict]) -> list[Triple]: ...


@dataclass
class DocResult:
    """Outcome of a single passage — feeds the comparison view in the UI."""

    source_id: str
    title: str
    chunk_index: int
    original_text: str
    sample_id: str = ""
    resolved_text: str = ""
    triples: list[Triple] = field(default_factory=list)
    written: int = 0
    skipped: bool = False
    error: str = ""
    seconds: float = 0.0
    extract_seconds: float = 0.0  # extractor.extract() only, no coref/embed/write

    @property
    def zero_triples(self) -> bool:
        return not self.skipped and not self.error and not self.triples


@dataclass
class IngestReport:
    extractor: str
    sample_id: str
    question: str
    docs: list[DocResult] = field(default_factory=list)

    @property
    def total_triples(self) -> int:
        return sum(d.written for d in self.docs)

    @property
    def total_extracted(self) -> int:
        return sum(len(d.triples) for d in self.docs)

    @property
    def processed(self) -> list[DocResult]:
        return [d for d in self.docs if not d.skipped]

    @property
    def zero_triple_docs(self) -> list[DocResult]:
        return [d for d in self.docs if d.zero_triples]


class AlceIngestor:
    """Ingests an ALCE entry into the graph with a specific extractor."""

    def __init__(
        self,
        client: Optional[Neo4jClient],
        extractor: Extractor,
        resolver: Optional[CoreferenceResolver] = None,
        registry: Optional[ProcessedRegistry] = None,
        use_coref: bool = True,
    ):
        self._client = client
        self._extractor = extractor
        self._resolver = resolver or CoreferenceResolver()
        self._registry = registry or ProcessedRegistry()
        self._writer = GraphWriter(client=client) if client else None
        self._use_coref = use_coref

    @property
    def extractor_name(self) -> str:
        return self._extractor.name

    # ────────────────────────────────────────────────────────────────
    # Skip
    # ────────────────────────────────────────────────────────────────

    def is_processed(self, source_id: str) -> bool:
        if self._registry.is_processed(source_id, self.extractor_name):
            return True
        if self._client:
            return self._client.is_source_ingested(source_id, self.extractor_name)
        return False

    def processed_ids(self) -> set[str]:
        """Union of graph + registry — used for the status indicator in UI."""
        ids = self._registry.processed_ids(self.extractor_name)
        if self._client:
            try:
                ids |= self._client.ingested_source_ids(self.extractor_name)
            except Exception as exc:
                logger.warning("Neo4j unreachable for ingest status: %s", exc)
        return ids

    # ────────────────────────────────────────────────────────────────
    # Phase 1: Extract (coref + triple extraction, NO Neo4j)
    # ────────────────────────────────────────────────────────────────

    def extract_entry(
        self,
        entry: AlceEntry,
        skip_existing: bool = False,
        progress: Optional[Callable[[str], None]] = None,
    ) -> IngestReport:
        """Extract triples from all passages of an ALCE question.
        Does NOT write to Neo4j."""
        report = IngestReport(
            extractor=self.extractor_name,
            sample_id=entry.sample_id,
            question=entry.question,
        )

        for chunk in entry.docs():
            report.docs.append(
                self.extract_doc(
                    chunk,
                    sample_id=entry.sample_id,
                    skip_existing=skip_existing,
                    progress=progress,
                )
            )

        # Save all extracted triples to JSONL
        for doc in report.docs:
            if doc.triples:
                triples_dicts = [
                    {
                        "source_id": t.source_id,
                        "subject": t.subject,
                        "predicate": t.predicate,
                        "obj": t.obj,
                        "claim_span": t.claim_span,
                        "chunk_text": t.chunk_text,
                        "source_file": t.source_file,
                        "title": doc.title,
                        "chunk_index": t.chunk_index,
                    }
                    for t in doc.triples
                ]
                save_triples_batch(triples_dicts, entry.sample_id, self.extractor_name)

        return report

    def extract_doc(
        self,
        chunk: dict,
        sample_id: str = "",
        skip_existing: bool = False,
        progress: Optional[Callable[[str], None]] = None,
    ) -> DocResult:
        """Coref + triple extraction on a single passage. No Neo4j write."""
        source_id = str(chunk.get("source_id", ""))
        original = chunk.get("text", "")
        result = DocResult(
            source_id=source_id,
            title=chunk.get("title", chunk.get("source_file", "")),
            chunk_index=chunk.get("chunk_index", 0),
            original_text=original,
            sample_id=sample_id,
        )

        if skip_existing and self.is_processed(source_id):
            result.skipped = True
            return result

        if not original.strip():
            result.error = "empty passage"
            return result

        t0 = time.time()
        try:
            # 1. Coref on original text.
            if progress:
                progress(f"{self.extractor_name}: coreference on {source_id}...")
            resolved = self._resolver.resolve(original) if self._use_coref else original
            result.resolved_text = resolved

            # Save coref result to JSONL
            save_coref(
                source_id=source_id,
                sample_id=sample_id,
                title=result.title,
                chunk_index=result.chunk_index,
                original_text=original,
                resolved_text=resolved,
            )

            # 2. Extract triples from resolved text.
            if progress:
                progress(f"{self.extractor_name}: extracting {source_id}...")
            t_extract = time.time()
            raw_triples = self._extractor.extract([{**chunk, "text": resolved}])
            result.extract_seconds = time.time() - t_extract

            # 3. chunk_text = ORIGINAL text (evidence must be verbatim);
            #    claim_span anchored to original.
            triples = [
                t._replace(
                    chunk_text=original,
                    claim_span=best_span(original, t.subject, t.obj, t.claim_span),
                )
                for t in raw_triples
            ]
            result.triples = triples

        except Exception as exc:
            result.error = f"{type(exc).__name__}: {exc}"
            logger.error("Extraction failed on source_id=%s: %s", source_id, exc)

        result.seconds = time.time() - t0
        return result

    # ────────────────────────────────────────────────────────────────
    # Phase 2: Write to Neo4j (takes already-extracted DocResult)
    # ────────────────────────────────────────────────────────────────

    def write_entry(
        self,
        report: IngestReport,
        force: bool = False,
        progress: Optional[Callable[[str], None]] = None,
    ) -> IngestReport:
        """Write all extracted triples from a report to Neo4j."""
        for doc in report.docs:
            if doc.skipped or doc.error or not doc.triples:
                continue
            doc.written = self.write_doc(doc, force=force, progress=progress)

        return report

    def write_doc(
        self,
        result: DocResult,
        force: bool = False,
        progress: Optional[Callable[[str], None]] = None,
    ) -> int:
        """Write a DocResult's triples to Neo4j. Returns count written."""
        if not self._writer or not self._client:
            logger.warning("No Neo4j client — cannot write triples")
            return 0

        source_id = result.source_id

        if force:
            self._client.delete_by_source(source_id, self.extractor_name)

        if not result.triples:
            return 0

        try:
            if progress:
                progress(f"Writing {len(result.triples)} triples for {source_id}...")
            written = self._writer.write_triples(result.triples)
        except Exception as exc:
            logger.error("Write failed on source_id=%s: %s", source_id, exc)
            return 0

        # Mark registry — including docs with 0 triples.
        self._registry.mark(source_id, self.extractor_name, written)
        if written == 0:
            logger.warning(
                "ZERO TRIPLES written — extractor=%s source_id=%s title=%r",
                self.extractor_name, source_id, result.title,
            )

        return written

    # ────────────────────────────────────────────────────────────────
    # Legacy convenience wrappers (extract + write in one call)
    # ────────────────────────────────────────────────────────────────

    def ingest_entry(
        self,
        entry: AlceEntry,
        skip_existing: bool = True,
        force: bool = False,
        progress: Optional[Callable[[str], None]] = None,
    ) -> IngestReport:
        """Full pipeline: extract + write for all passages of an ALCE question."""
        report = self.extract_entry(entry, skip_existing=skip_existing, progress=progress)
        self.write_entry(report, force=force, progress=progress)
        return report

    def ingest_doc(
        self,
        chunk: dict,
        skip_existing: bool = True,
        force: bool = False,
        progress: Optional[Callable[[str], None]] = None,
    ) -> DocResult:
        """Full pipeline on a single passage. Legacy compatibility."""
        result = self.extract_doc(chunk, skip_existing=skip_existing, progress=progress)
        if not result.skipped and not result.error and result.triples:
            result.written = self.write_doc(result, force=force, progress=progress)
        return result


def build_extractor(name: str) -> Extractor:
    """Factory for extractor name ("rebel" | "deepseek")."""
    if name == settings.EXTRACTOR_REBEL:
        from src.ingestion.triple_extractor import TripleExtractor
        return TripleExtractor()
    if name == settings.EXTRACTOR_DEEPSEEK:
        from src.ingestion.deepseek_extractor import DeepSeekExtractor
        return DeepSeekExtractor()
    raise ValueError(
        f"Unknown extractor: {name!r} "
        f"(available: {settings.AVAILABLE_EXTRACTORS})"
    )
