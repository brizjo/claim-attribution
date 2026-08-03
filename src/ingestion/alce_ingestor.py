"""
Ingestor ALCE — orchestra la pipeline esistente sui passaggi ALCE.

Per ogni passaggio:
    skip-check → coref → estrazione triple → ancoraggio claim_span sul testo
    ORIGINALE → embedding predicato → MERGE in Neo4j → registro

Cambia solo la SORGENTE rispetto alla pipeline precedente: coref, REBEL,
embedding e writer sono gli stessi moduli.

Skip a due livelli, perché nessuno dei due basta da solo:
  * Neo4j (`is_source_ingested`) — non vede i doc a zero triple;
  * ProcessedRegistry — file locale, non sa se il grafo è stato svuotato.
Un doc è saltato se risulta processato in ALMENO uno dei due (`skip_existing`).
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
from src.ingestion.processed_registry import ProcessedRegistry
from src.ingestion.span_matcher import best_span
from src.ingestion.triple_extractor import Triple

logger = logging.getLogger(__name__)


class Extractor(Protocol):
    """Interfaccia comune a TripleExtractor (REBEL) e DeepSeekExtractor."""

    name: str

    def extract(self, chunks: list[dict]) -> list[Triple]: ...


@dataclass
class DocResult:
    """Esito di un singolo passaggio — alimenta la vista di confronto UI."""

    source_id: str
    title: str
    chunk_index: int
    original_text: str
    resolved_text: str = ""
    triples: list[Triple] = field(default_factory=list)
    written: int = 0
    skipped: bool = False
    error: str = ""
    seconds: float = 0.0

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
    def processed(self) -> list[DocResult]:
        return [d for d in self.docs if not d.skipped]

    @property
    def zero_triple_docs(self) -> list[DocResult]:
        return [d for d in self.docs if d.zero_triples]


class AlceIngestor:
    """Ingerisce una entry ALCE nel grafo con un estrattore specifico."""

    def __init__(
        self,
        client: Neo4jClient,
        extractor: Extractor,
        resolver: Optional[CoreferenceResolver] = None,
        registry: Optional[ProcessedRegistry] = None,
        use_coref: bool = True,
    ):
        self._client = client
        self._extractor = extractor
        self._resolver = resolver or CoreferenceResolver()
        self._registry = registry or ProcessedRegistry()
        self._writer = GraphWriter(client=client)
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
        return self._client.is_source_ingested(source_id, self.extractor_name)

    def processed_ids(self) -> set[str]:
        """Union grafo + registro — indicatore ✓/○ nella UI."""
        ids = self._registry.processed_ids(self.extractor_name)
        try:
            ids |= self._client.ingested_source_ids(self.extractor_name)
        except Exception as exc:
            logger.warning("Neo4j non raggiungibile per lo stato ingest: %s", exc)
        return ids

    # ────────────────────────────────────────────────────────────────
    # Ingest
    # ────────────────────────────────────────────────────────────────

    def ingest_entry(
        self,
        entry: AlceEntry,
        skip_existing: bool = True,
        force: bool = False,
        progress: Optional[Callable[[str], None]] = None,
    ) -> IngestReport:
        """Ingerisce i primi N passaggi di una domanda ALCE."""
        report = IngestReport(
            extractor=self.extractor_name,
            sample_id=entry.sample_id,
            question=entry.question,
        )

        for chunk in entry.docs():
            report.docs.append(
                self.ingest_doc(chunk, skip_existing=skip_existing, force=force, progress=progress)
            )

        return report

    def ingest_doc(
        self,
        chunk: dict,
        skip_existing: bool = True,
        force: bool = False,
        progress: Optional[Callable[[str], None]] = None,
    ) -> DocResult:
        """Pipeline completa su un passaggio. Non solleva: l'errore va nel result."""
        source_id = str(chunk.get("source_id", ""))
        original = chunk.get("text", "")
        result = DocResult(
            source_id=source_id,
            title=chunk.get("title", chunk.get("source_file", "")),
            chunk_index=chunk.get("chunk_index", 0),
            original_text=original,
        )

        if force:
            self._client.delete_by_source(source_id, self.extractor_name)
        elif skip_existing and self.is_processed(source_id):
            result.skipped = True
            return result

        if not original.strip():
            result.error = "passaggio vuoto"
            return result

        t0 = time.time()
        try:
            # 1. Coref sul testo originale (il testo salvato resta l'originale).
            resolved = self._resolver.resolve(original) if self._use_coref else original
            result.resolved_text = resolved

            # 2. Estrazione sul testo risolto.
            if progress:
                progress(f"{self.extractor_name}: estrazione {source_id}...")
            raw_triples = self._extractor.extract([{**chunk, "text": resolved}])

            # 3. chunk_text = testo ORIGINALE (l'evidenza deve essere verbatim);
            #    claim_span ancorato all'originale — per REBEL non esiste, per
            #    DeepSeek è calcolato sul testo risolto: in entrambi i casi va
            #    ri-ancorato qui.
            triples = [
                t._replace(
                    chunk_text=original,
                    claim_span=best_span(original, t.subject, t.obj, t.claim_span),
                )
                for t in raw_triples
            ]
            result.triples = triples

            # 4. Embedding predicato + MERGE.
            result.written = self._writer.write_triples(triples) if triples else 0

        except Exception as exc:
            result.error = f"{type(exc).__name__}: {exc}"
            logger.error("Ingest fallito su source_id=%s: %s", source_id, exc)
            result.seconds = time.time() - t0
            return result

        result.seconds = time.time() - t0

        # 5. Registro — anche (soprattutto) con 0 triple.
        self._registry.mark(source_id, self.extractor_name, result.written)
        if result.written == 0:
            logger.warning(
                "ZERO TRIPLE — extractor=%s source_id=%s title=%r",
                self.extractor_name, source_id, result.title,
            )

        return result


def build_extractor(name: str) -> Extractor:
    """Factory per nome estrattore ("rebel" | "deepseek")."""
    if name == settings.EXTRACTOR_REBEL:
        from src.ingestion.triple_extractor import TripleExtractor
        return TripleExtractor()
    if name == settings.EXTRACTOR_DEEPSEEK:
        from src.ingestion.deepseek_extractor import DeepSeekExtractor
        return DeepSeekExtractor()
    raise ValueError(
        f"Estrattore sconosciuto: {name!r} "
        f"(disponibili: {settings.AVAILABLE_EXTRACTORS})"
    )
