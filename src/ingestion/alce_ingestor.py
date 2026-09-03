"""
ALCE Ingestor — orchestrates the pipeline on ALCE passages.

For each passage:
    skip-check -> coref -> sentence split -> triple extraction -> GUARDRAILS
    (+ un round di repair DeepSeek sulle bocciate) -> claim_span dalla frase
    ORIGINALE allineata -> canonicalize -> MERGE into Neo4j -> registry

Three-phase architecture (canonicalization added 2026-09-03):
    1. extract_doc / extract_entry  — coref + extraction, produces DocResult
       with triples but does NOT touch Neo4j.  Output saved to JSONL.
    2. canonicalize_entry           — unifica i nodi dentro lo scope scelto e
       calcola `predicate_embedding` (un batch per domanda).  Ancora niente I/O.
    3. write_doc / write_entry      — prende triple gia' canonicalizzate e gia'
       embeddate, scrive su Neo4j, marca il registry.

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
from src.ingestion import guardrails, triple_repair
from src.ingestion.alce_loader import AlceEntry
from src.ingestion.coref_resolver import CoreferenceResolver
from src.ingestion.entity_canonicalizer import (
    CanonicalizationResult,
    EntityCanonicalizer,
)
from src.ingestion.graph_writer import GraphWriter
from src.ingestion.output_store import (
    save_coref,
    save_discarded_triples,
    save_triples_batch,
)
from src.ingestion.processed_registry import ProcessedRegistry
from src.ingestion.span_matcher import align_sentences, best_span
from src.ingestion.triple_extractor import Triple

logger = logging.getLogger(__name__)


class Extractor(Protocol):
    """Interfaccia dell'estrattore di triple della pipeline (DeepSeek).

    Resta un Protocol e non un tipo concreto perche' gli esperimenti iniettano
    altri estrattori (REBEL, ibrido) sugli stessi stadi a valle."""

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
    # Guardrail in pipeline (2026-09-03): triple bocciate (dict con
    # discard_reason + stage "extract"|"repair") e quante sono state
    # recuperate dal round di riparazione DeepSeek.
    discarded: list[dict] = field(default_factory=list)
    repaired: int = 0
    # La coref e' fallita su QUESTO passaggio: si e' proseguito col testo
    # originale e i guardrail hanno fatto da rete (unresolved_reference).
    coref_failed: bool = False

    @property
    def zero_triples(self) -> bool:
        return not self.skipped and not self.error and not self.triples


@dataclass
class IngestReport:
    extractor: str
    sample_id: str
    question: str
    docs: list[DocResult] = field(default_factory=list)
    # Riepilogo della fase di canonicalizzazione (vuoto finche' non gira).
    canonicalization: dict = field(default_factory=dict)

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

    @property
    def total_discarded(self) -> int:
        return sum(len(d.discarded) for d in self.docs)

    @property
    def total_repaired(self) -> int:
        return sum(d.repaired for d in self.docs)


class AlceIngestor:
    """Ingests an ALCE entry into the graph with a specific extractor."""

    def __init__(
        self,
        client: Optional[Neo4jClient],
        extractor: Extractor,
        resolver: Optional[CoreferenceResolver] = None,
        registry: Optional[ProcessedRegistry] = None,
        use_coref: bool = True,
        canonicalizer: Optional[EntityCanonicalizer] = None,
        anchorer: Optional[guardrails.EntityAnchorer] = None,
    ):
        self._client = client
        self._extractor = extractor
        self._resolver = resolver or CoreferenceResolver()
        self._registry = registry or ProcessedRegistry()
        # Il canonicalizer non carica nulla finche' non serve (encoder lazy).
        self._canonicalizer = canonicalizer or EntityCanonicalizer()
        self._writer = GraphWriter(client=client) if client else None
        self._use_coref = use_coref
        self._nlp = None  # lazy-loaded spaCy model for sentence splitting
        self._anchorer = anchorer  # guardrail generic_node (spaCy NER/POS)

    def _get_nlp(self):
        """Lazily load the spaCy model (reused across calls)."""
        if self._nlp is None:
            import spacy
            try:
                self._nlp = spacy.load(settings.SPACY_MODEL)
            except OSError:
                import spacy
                self._nlp = spacy.blank("en")
                self._nlp.add_pipe("sentencizer")
                logger.warning(
                    "spaCy model %s not found — using blank sentencizer",
                    settings.SPACY_MODEL,
                )
        return self._nlp

    def _get_anchorer(self) -> guardrails.EntityAnchorer:
        """Anchorer per il guardrail `generic_node`.

        Riusa lo spaCy dell'ingestor se ha la NER; il fallback "blank
        sentencizer" di `_get_nlp` NON basta (senza NER ogni entita'
        risulterebbe generica) — in quel caso si usa `default_anchorer`,
        che fallisce rumorosamente se il modello manca."""
        if self._anchorer is None:
            nlp = self._get_nlp()
            if "ner" in getattr(nlp, "pipe_names", []):
                self._anchorer = guardrails.EntityAnchorer(nlp=nlp)
            else:
                self._anchorer = guardrails.default_anchorer()
        return self._anchorer

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
            # 1. Coref on original text.  GUARDRAIL (2026-09-03): un bug di
            #    fastcoref su UN passaggio non deve piu' uccidere il passaggio
            #    intero — si prosegue col testo originale, si marca
            #    `coref_failed` e i guardrail a valle (`unresolved_reference`)
            #    scartano i deittici rimasti.  Il check globale "fastcoref non
            #    carica affatto" resta nel runner batch (exit 2).
            if progress:
                progress(f"{self.extractor_name}: coreference on {source_id}...")
            resolved = original
            if self._use_coref:
                try:
                    resolved = self._resolver.resolve(original)
                except Exception as exc:
                    result.coref_failed = True
                    logger.error(
                        "Coref FALLITA su source_id=%s (%s: %s) — si prosegue "
                        "col testo ORIGINALE; i guardrail scartano i "
                        "riferimenti irrisolti.",
                        source_id, type(exc).__name__, exc,
                    )
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

            # 2. Sentence-tokenize resolved text, then extract triples.
            if progress:
                progress(f"{self.extractor_name}: extracting {source_id}...")
            t_extract = time.time()
            nlp = self._get_nlp()
            sents = [s.text.strip() for s in nlp(resolved).sents
                     if s.text.strip()]
            if not sents:
                sents = [resolved]
            sent_chunks = [{**chunk, "text": s} for s in sents]
            raw_triples = self._extractor.extract(sent_chunks)
            result.extract_seconds = time.time() - t_extract

            # 3. Guardrail + repair round (una sola ripetizione, mai un loop).
            if progress and raw_triples:
                progress(f"{self.extractor_name}: guardrails on {source_id}...")
            kept = self._guard_and_repair(result, chunk, raw_triples)

            # 4. chunk_text = ORIGINAL text (evidence must be verbatim);
            #    claim_span = frase ORIGINALE allineata alla frase risolta
            #    da cui la tripla e' uscita (best_span solo come fallback).
            orig_sents = [s.text.strip() for s in nlp(original).sents
                          if s.text.strip()]
            sent_map = {res: orig
                        for orig, res in align_sentences(orig_sents, sents)}
            result.triples = [
                t._replace(
                    chunk_text=original,
                    claim_span=(sent_map.get(t.chunk_text)
                                or best_span(original, t.subject, t.obj,
                                             t.claim_span)),
                )
                for t in kept
            ]

        except Exception as exc:
            result.error = f"{type(exc).__name__}: {exc}"
            logger.error("Extraction failed on source_id=%s: %s", source_id, exc)

        result.seconds = time.time() - t0
        return result

    # ────────────────────────────────────────────────────────────────
    # Guardrail + repair (dentro extract_doc, prima dello span)
    # ────────────────────────────────────────────────────────────────

    def _discard_record(  # noqa: PLR0913
        self, result: DocResult, subject: str, predicate: str, obj: str,
        sentence: str, reason: str, stage: str,
    ) -> dict:
        return {
            "sample_id": result.sample_id,
            "source_id": result.source_id,
            "title": result.title,
            "chunk_index": result.chunk_index,
            "extractor": self.extractor_name,
            "sentence": sentence,
            "subject": subject,
            "predicate": predicate,
            "object": obj,
            "discard_reason": reason,
            "stage": stage,
        }

    def _guard_and_repair(
        self, result: DocResult, chunk: dict, raw_triples: list[Triple],
    ) -> list[Triple]:
        """
        Guardrail sulle triple grezze + UN round di riparazione DeepSeek.

        Portato dagli esperimenti alla pipeline principale (2026-09-03): le
        classi di corruzione osservate (subject=object, nodi generici,
        deittici irrisolti, entita' inventate) finivano dritte in Neo4j.
        Le bocciate NON si buttano subito: tornano a DeepSeek — una chiamata
        per frase fallita, col motivo dello scarto — e le riparate ripassano
        gli STESSI guardrail.  Chi fallisce due volte muore, loggato in
        `data/outputs/triples_discarded.jsonl`.

        La frase di verifica e' `t.chunk_text`: la frase coref-RISOLTA data
        all'estrattore (lo span verbatim sull'originale si assegna dopo).
        """
        anchorer = self._get_anchorer()
        title = chunk.get("title", chunk.get("source_file", ""))
        kept: list[Triple] = []
        failed: dict[str, list[tuple[Triple, str]]] = {}
        for t in raw_triples:
            sentence = t.chunk_text
            verdict = guardrails.check(
                t.subject, t.predicate, t.obj, sentence,
                anchorer=anchorer, title=title)
            if verdict.ok:
                kept.append(t)
            else:
                failed.setdefault(sentence, []).append((t, verdict.reason))

        records: list[dict] = []
        # Il Protocol `Extractor` non impone un client LLM: gli stub dei test
        # e gli estrattori sperimentali senza `.client` saltano il repair.
        repair_client = getattr(self._extractor, "client", None)

        for sentence, fails in failed.items():
            rejected = [
                {"subject": t.subject, "predicate": t.predicate,
                 "obj": t.obj, "reason": reason}
                for t, reason in fails
            ]
            for t, reason in fails:
                records.append(self._discard_record(
                    result, t.subject, t.predicate, t.obj, sentence,
                    reason, "extract"))
            if repair_client is None:
                continue
            for item in triple_repair.repair_sentence(
                    repair_client, sentence, title, rejected):
                verdict = guardrails.check(
                    item["subject"], item["predicate"], item["obj"],
                    sentence, anchorer=anchorer, title=title)
                if verdict.ok:
                    result.repaired += 1
                    kept.append(Triple(
                        subject=item["subject"],
                        predicate=item["predicate"],
                        obj=item["obj"],
                        chunk_text=sentence,
                        source_file=chunk.get("source_file", ""),
                        chunk_index=chunk.get("chunk_index", 0),
                        source_id=result.source_id,
                        extractor=self.extractor_name,
                        claim_span=item.get("claim_span", ""),
                    ))
                else:
                    records.append(self._discard_record(
                        result, item["subject"], item["predicate"],
                        item["obj"], sentence, verdict.reason, "repair"))

        # Dedup su (S, P, O): il repair puo' ri-emettere una tripla gia' tenuta
        # e la stessa tripla puo' uscire da due frasi dello stesso passaggio.
        deduped: list[Triple] = []
        seen: set[tuple[str, str, str]] = set()
        for t in kept:
            key = (t.subject.lower(), t.predicate.lower(), t.obj.lower())
            if key in seen:
                records.append(self._discard_record(
                    result, t.subject, t.predicate, t.obj, t.chunk_text,
                    "duplicate", "dedup"))
                continue
            seen.add(key)
            deduped.append(t)

        result.discarded = records
        if records:
            save_discarded_triples(records)
        return deduped

    # ────────────────────────────────────────────────────────────────
    # Phase 2: Canonicalize (nodi unificati + predicate_embedding, NO Neo4j)
    # ────────────────────────────────────────────────────────────────

    def canonicalize_entry(
        self,
        report: IngestReport,
        progress: Optional[Callable[[str], None]] = None,
    ) -> CanonicalizationResult:
        """
        Unifica le menzioni delle triple estratte e calcola gli embedding dei
        predicati.  Va eseguita FRA extract_entry e write_entry: e' il solo
        momento in cui tutte le triple della domanda sono in memoria e nessuna
        e' ancora scritta.
        """
        result = self._canonicalizer.canonicalize(report, progress=progress)
        report.canonicalization = result.summary()
        logger.info("Canonicalization: %s", report.canonicalization)
        return result

    # ────────────────────────────────────────────────────────────────
    # Phase 3: Write to Neo4j (takes already-canonicalized DocResult)
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
        """Full pipeline: extract -> canonicalize -> write."""
        report = self.extract_entry(entry, skip_existing=skip_existing, progress=progress)
        self.canonicalize_entry(report, progress=progress)
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


def build_extractor(name: str = settings.EXTRACTOR_DEEPSEEK) -> Extractor:
    """
    Factory dell'estrattore della pipeline.  Un solo estrattore: DeepSeek.

    REBEL e' stato tolto dalla pipeline principale (2026-09-03): sui 50
    passaggi misurati confermava l'11-27% delle triple e la variante ancorata
    al suo vocabolario produceva MENO triple e piu' povere (vedi
    `tasks/todo.md`, fase ibrida).  Resta disponibile solo negli esperimenti.
    """
    if name == settings.EXTRACTOR_DEEPSEEK:
        from src.ingestion.deepseek_extractor import DeepSeekExtractor
        return DeepSeekExtractor()
    if name == settings.EXTRACTOR_REBEL:
        raise ValueError(
            "REBEL non fa piu' parte della pipeline principale: l'unico "
            "estrattore e' DeepSeek.  Per confrontarli usa gli esperimenti "
            "(SHOW_EXPERIMENTS=1) o scripts/run_hybrid_experiment.py."
        )
    raise ValueError(
        f"Unknown extractor: {name!r} "
        f"(available: {settings.AVAILABLE_EXTRACTORS})"
    )
