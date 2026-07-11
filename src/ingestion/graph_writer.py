"""
Graph writer — takes extracted triples, computes predicate embeddings,
writes to Neo4j.
"""

from __future__ import annotations

from typing import Callable, Optional

from config import settings
from src.graph.neo4j_client import Neo4jClient
from src.ingestion.triple_extractor import Triple


class GraphWriter:
    """Embeds predicate strings and writes triples to Neo4j."""

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
        Embed predicates and write triples to Neo4j.  Does NOT create a
        Document node — call finalize_document() once after all chunks.
        """
        if not triples:
            return 0

        self._load_encoder()

        predicates = [t.predicate for t in triples]
        embeddings = self._encoder.encode(predicates, show_progress_bar=False)

        triples_dicts = []
        for triple, emb in zip(triples, embeddings):
            triples_dicts.append({
                "subject": triple.subject,
                "predicate": triple.predicate,
                "obj": triple.obj,
                "chunk_text": triple.chunk_text,
                "source_file": triple.source_file,
                "chunk_index": triple.chunk_index,
                "predicate_embedding": emb.tolist(),
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
