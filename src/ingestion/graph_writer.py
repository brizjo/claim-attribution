"""
Graph writer — scrive su Neo4j triple gia' canonicalizzate.

Dal 2026-09-03 l'embedding dei predicati lo calcola la fase di
canonicalizzazione (`src/ingestion/entity_canonicalizer.py`): un solo batch per
domanda invece di uno per write.  Qui resta solo un fallback per le triple che
arrivano senza `predicate_embedding` (percorsi legacy, UI passo-passo).
"""

from __future__ import annotations

from typing import Callable, Optional

from config import settings
from src.graph.neo4j_client import Neo4jClient
from src.ingestion.triple_extractor import Triple


class GraphWriter:
    """Writes triples to Neo4j; embeds predicates solo se non gia' embeddati."""

    def __init__(
        self,
        client: Neo4jClient,
        embedding_model: str = settings.PREDICATE_EMBEDDING_MODEL,
    ):
        self._client = client
        self._model_name = embedding_model
        self._encoder = None

    def _load_encoder(self) -> None:
        if self._encoder is not None:
            return
        from sentence_transformers import SentenceTransformer
        self._encoder = SentenceTransformer(self._model_name)

    def write_triples(
        self,
        triples: list[Triple],
        progress_callback: Optional[Callable[[str], None]] = None,
    ) -> int:
        """
        Write triples to Neo4j.  Does NOT create a Document node — call
        finalize_document() once after all chunks.

        `predicate_embedding` arriva dalla canonicalizzazione; viene calcolato
        qui solo per le triple che ne sono prive.
        """
        if not triples:
            return 0

        missing = sorted({t.predicate for t in triples if not t.predicate_embedding})
        table: dict[str, list[float]] = {}
        if missing:
            self._load_encoder()
            for predicate, emb in zip(
                missing, self._encoder.encode(missing, show_progress_bar=False)
            ):
                table[predicate] = [float(x) for x in emb]

        triples_dicts = []
        for triple in triples:
            embedding = (list(triple.predicate_embedding)
                         or table.get(triple.predicate, []))
            triples_dicts.append({
                "subject": triple.subject,
                "predicate": triple.predicate,
                "obj": triple.obj,
                "chunk_text": triple.chunk_text,
                "claim_span": triple.claim_span,
                "source_file": triple.source_file,
                # source_id fa parte della chiave dell'arco: senza fallback
                # triple identiche da passaggi diversi collasserebbero.
                "source_id": triple.source_id or f"{triple.source_file}#{triple.chunk_index}",
                "extractor": triple.extractor,
                "chunk_index": triple.chunk_index,
                "predicate_embedding": embedding,
                # Provenienza verbatim: la forma canonica sta sul nodo, la
                # menzione com'era nel testo resta sull'arco.
                "subject_surface": triple.subject_surface or triple.subject,
                "object_surface": triple.object_surface or triple.obj,
                "subject_external_id": triple.subject_external_id,
                "object_external_id": triple.object_external_id,
            })

        return self._client.batch_write_triples(triples_dicts, progress_callback)

    def finalize_document(
        self,
        source_file: str,
        num_chunks: int,
        num_triples: int,
        status: str = "done",
    ) -> None:
        """Create/update the Document node after streaming ingest completes."""
        if not source_file:
            return
        self._client.create_document_node(
            name=source_file,
            num_chunks=num_chunks,
            num_triples=num_triples,
            status=status,
        )

    # Legacy batch API — preserved for compatibility, delegates to streaming.
    def write(
        self,
        triples: list[Triple],
        source_file: str = "",
        num_chunks: int = 0,
        progress_callback: Optional[Callable[[str], None]] = None,
    ) -> int:
        try:
            written = self.write_triples(triples, progress_callback)
            self.finalize_document(source_file, num_chunks, written, "done")
            return written
        except Exception:
            self.finalize_document(source_file, num_chunks, 0, "error")
            raise
