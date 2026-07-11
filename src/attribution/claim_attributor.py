"""
Claim attributor — verifies a natural-language claim or answers a question
against the Neo4j knowledge graph.

Two routes:

  1. Claim path (declarative input)
       parse with mREBEL → exact match → semantic predicate-cosine fallback

  2. Question path (interrogative input)
       parse with Ollama into partial triple (?, P, O) / (S, ?, O) / (S, P, ?)
       → Neo4j pattern query → rank candidates by predicate cosine similarity
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from config import settings
from src.graph.neo4j_client import Neo4jClient
from src.ingestion.triple_extractor import TripleExtractor
from src.attribution.question_parser import QuestionParser, QuerySpec


@dataclass
class AttributionResult:
    claim: str
    # Parsed/resolved triple
    subject: str = ""
    predicate: str = ""
    obj: str = ""
    # Outcome
    match_type: str = "not_found"     # "exact" | "semantic" | "not_found" | "parse_error"
    similarity: float = 0.0
    source_chunk: str = ""
    source_file: str = ""
    chunk_index: int = 0
    verified: bool = False
    # Question-mode metadata
    is_question: bool = False
    answer_field: str = ""             # which slot was the answer ("subject"/"predicate"/"object")


_QUESTION_STARTERS = (
    "chi ", "cosa ", "che cosa ", "che ", "quale ", "quali ",
    "quando ", "dove ", "come ", "perché ", "perche ", "quanto ",
    "who ", "what ", "where ", "when ", "why ", "how ", "which ",
)


class ClaimAttributor:
    """Routes input to claim or question flow, queries Neo4j, returns evidence."""

    def __init__(
        self,
        client: Neo4jClient,
        semantic_threshold: float = settings.SEMANTIC_THRESHOLD,
        embedding_model: str = settings.PREDICATE_EMBEDDING_MODEL,
    ):
        self._client = client
        self._threshold = semantic_threshold
        self._model_name = embedding_model
        self._extractor = TripleExtractor()
        self._qparser: Optional[QuestionParser] = None
        self._encoder = None

    # ────────────────────────────────────────────────────────────────
    # Lazy loaders
    # ────────────────────────────────────────────────────────────────

    def _load_encoder(self) -> None:
        if self._encoder is not None:
            return
        from sentence_transformers import SentenceTransformer
        self._encoder = SentenceTransformer(self._model_name)

    def _embed(self, text: str) -> list[float]:
        self._load_encoder()
        return self._encoder.encode([text], show_progress_bar=False)[0].tolist()

    def _get_qparser(self) -> QuestionParser:
        if self._qparser is None:
            self._qparser = QuestionParser()
        return self._qparser

    @staticmethod
    def _looks_like_question(text: str) -> bool:
        lowered = text.lower().strip()
        if lowered.endswith("?"):
            return True
        return lowered.startswith(_QUESTION_STARTERS)

    # ────────────────────────────────────────────────────────────────
    # Public API
    # ────────────────────────────────────────────────────────────────

    def attribute(self, claim: str) -> AttributionResult:
        """Detect input type and route accordingly."""
        text = claim.strip()
        if self._looks_like_question(text):
            return self._attribute_question(text)
        return self._attribute_claim(text)

    # ────────────────────────────────────────────────────────────────
    # Claim flow (declarative)
    # ────────────────────────────────────────────────────────────────

    def _attribute_claim(self, claim: str) -> AttributionResult:
        dummy_chunk = {"text": claim, "source_file": "claim", "chunk_index": 0}
        try:
            triples = self._extractor.extract([dummy_chunk])
        except Exception as e:
            return AttributionResult(
                claim=claim,
                match_type="parse_error",
                source_chunk=f"Estrattore REBEL ha fallito: {e}",
            )

        if not triples:
            return AttributionResult(
                claim=claim,
                match_type="parse_error",
                source_chunk=(
                    "REBEL non ha estratto alcuna tripla dal testo. "
                    "Verifica che il claim sia dichiarativo e contenga "
                    "soggetto, predicato e oggetto espliciti."
                ),
            )

        t = triples[0]
        result = AttributionResult(
            claim=claim,
            subject=t.subject,
            predicate=t.predicate,
            obj=t.obj,
        )

        # Stage 1: exact match
        exact = self._client.exact_match(t.subject, t.predicate, t.obj)
        if exact:
            result.match_type = "exact"
            result.similarity = 1.0
            result.source_chunk = exact["chunk_text"]
            result.source_file = exact["source_file"]
            result.chunk_index = exact.get("chunk_index", 0)
            result.verified = True
            return result

        # Stage 2: semantic fallback (predicate cosine, S/O fixed)
        pred_emb = self._embed(t.predicate)
        candidates = self._client.semantic_fallback(t.subject, t.obj, pred_emb)

        if candidates:
            best = candidates[0]
            sim = best["similarity"]
            if sim >= self._threshold:
                result.match_type = "semantic"
                result.similarity = sim
                result.source_chunk = best["chunk_text"]
                result.source_file = best["source_file"]
                result.chunk_index = best.get("chunk_index", 0)
                result.verified = True
            else:
                result.match_type = "not_found"
                result.similarity = sim

        return result

    # ────────────────────────────────────────────────────────────────
    # Question flow (interrogative)
    # ────────────────────────────────────────────────────────────────

    def _attribute_question(self, question: str) -> AttributionResult:
        spec = self._get_qparser().parse(question)

        if spec.known_count == 0:
            return AttributionResult(
                claim=question,
                is_question=True,
                match_type="parse_error",
                source_chunk=(
                    "Impossibile estrarre una tripla parziale dalla domanda. "
                    "Riformula con entità più esplicite "
                    "(es. 'Chi è il protagonista di Monster?')."
                ),
            )

        pred_emb = self._embed(spec.predicate) if spec.predicate else None

        candidates = self._client.query_partial(
            subject=spec.subject,
            obj=spec.object,
            predicate_embedding=pred_emb,
            top_k=5,
        )

        answer_field = spec.unknown_field() or ""

        if not candidates:
            return AttributionResult(
                claim=question,
                is_question=True,
                subject=spec.subject or "",
                predicate=spec.predicate or "",
                obj=spec.object or "",
                match_type="not_found",
                answer_field=answer_field,
            )

        best = candidates[0]
        sim = best.get("similarity", 0.0)

        # Without predicate embedding we can't grade similarity — accept the
        # graph match as semantic at unit confidence.
        if pred_emb is None:
            match_type = "semantic"
            verified = True
            similarity = 1.0
        elif sim >= self._threshold:
            match_type = "semantic"
            verified = True
            similarity = sim
        else:
            match_type = "not_found"
            verified = False
            similarity = sim

        return AttributionResult(
            claim=question,
            is_question=True,
            subject=best["subject"],
            predicate=best["predicate"],
            obj=best["object"],
            match_type=match_type,
            similarity=similarity,
            source_chunk=best["chunk_text"],
            source_file=best["source_file"],
            chunk_index=best.get("chunk_index", 0),
            verified=verified,
            answer_field=answer_field,
        )
