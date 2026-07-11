"""
Neo4j client — connection, schema setup, triple CRUD, attribution queries.
"""

from __future__ import annotations

import numpy as np
from typing import Optional, Callable

from config import settings


class Neo4jClient:
    """
    Wraps the neo4j driver.

    Schema:
        (:Entity {name, normalized_name})
        -[:RELATES_TO {predicate, chunk_text, source_file, chunk_index,
                        predicate_embedding}]->
    """

    def __init__(
        self,
        uri: str = settings.NEO4J_URI,
        user: str = settings.NEO4J_USER,
        password: str = settings.NEO4J_PASSWORD,
        database: str = settings.NEO4J_DATABASE,
    ):
        from neo4j import GraphDatabase  # lazy import
        self._driver = GraphDatabase.driver(uri, auth=(user, password))
        self._database = database
        self._schema_ready = False

    def _session(self):
        """All queries must use this so they hit the configured database."""
        return self._driver.session(database=self._database)

    @property
    def database(self) -> str:
        return self._database

    # ────────────────────────────────────────────────────────────────
    # Schema
    # ────────────────────────────────────────────────────────────────

    def _ensure_schema(self) -> None:
        if self._schema_ready:
            return
        with self._session() as s:
            s.run(
                "CREATE CONSTRAINT entity_unique IF NOT EXISTS "
                "FOR (e:Entity) REQUIRE e.normalized_name IS UNIQUE"
            )
            s.run(
                "CREATE CONSTRAINT document_unique IF NOT EXISTS "
                "FOR (d:Document) REQUIRE d.name IS UNIQUE"
            )
        self._schema_ready = True

    def close(self) -> None:
        self._driver.close()

    # ────────────────────────────────────────────────────────────────
    # Write
    # ────────────────────────────────────────────────────────────────

    def batch_write_triples(
        self,
        triples: list[dict],
        progress_callback: Optional[Callable[[str], None]] = None,
    ) -> int:
        """Write all triples in a single transaction (much faster)."""
        self._ensure_schema()
        if not triples:
            return 0

        with self._session() as s:
            tx = s.begin_transaction()
            try:
                for i, t in enumerate(triples):
                    s_norm = t["subject"].lower().strip()
                    o_norm = t["obj"].lower().strip()
                    tx.run(
                        """
                        MERGE (sub:Entity {normalized_name: $s_norm})
                          ON CREATE SET sub.name = $s_name
                        MERGE (obj:Entity {normalized_name: $o_norm})
                          ON CREATE SET obj.name = $o_name
                        CREATE (sub)-[:RELATES_TO {
                            predicate: $pred,
                            chunk_text: $chunk_text,
                            source_file: $source_file,
                            chunk_index: $chunk_index,
                            predicate_embedding: $pred_emb
                        }]->(obj)
                        """,
                        s_norm=s_norm,
                        s_name=t["subject"],
                        o_norm=o_norm,
                        o_name=t["obj"],
                        pred=t["predicate"],
                        chunk_text=t["chunk_text"],
                        source_file=t["source_file"],
                        chunk_index=t["chunk_index"],
                        pred_emb=t["predicate_embedding"],
                    )
                    if progress_callback and (i + 1) % 10 == 0:
                        progress_callback(f"Indexed {i + 1}/{len(triples)} triples...")
                tx.commit()  # explicit — never trust driver auto-commit on exit
            except Exception:
                tx.rollback()
                raise

        return len(triples)

    def upsert_triple(  # noqa: PLR0913
        self,
        subject: str,
        predicate: str,
        obj: str,
        chunk_text: str,
        source_file: str,
        chunk_index: int,
        predicate_embedding: list[float],
    ) -> None:
        self._ensure_schema()
        s_norm = subject.lower().strip()
        o_norm = obj.lower().strip()
        with self._session() as s:
            s.run(
                """
                MERGE (sub:Entity {normalized_name: $s_norm})
                  ON CREATE SET sub.name = $s_name
                MERGE (obj:Entity {normalized_name: $o_norm})
                  ON CREATE SET obj.name = $o_name
                CREATE (sub)-[:RELATES_TO {
                    predicate: $pred,
                    chunk_text: $chunk_text,
                    source_file: $source_file,
                    chunk_index: $chunk_index,
                    predicate_embedding: $pred_emb
                }]->(obj)
                """,
                s_norm=s_norm,
                s_name=subject,
                o_norm=o_norm,
                o_name=obj,
                pred=predicate,
                chunk_text=chunk_text,
                source_file=source_file,
                chunk_index=chunk_index,
                pred_emb=predicate_embedding,
            )

    def clear_graph(self) -> None:
        with self._session() as s:
            s.run("MATCH (n) DETACH DELETE n")

    def create_document_node(
        self,
        name: str,
        num_chunks: int,
        num_triples: int,
        status: str,
    ) -> None:
        """Create or update Document node with ingestion metadata."""
        from datetime import datetime
        self._ensure_schema()
        ingested_at = datetime.utcnow().isoformat()
        with self._session() as s:
            s.run(
                """
                MERGE (d:Document {name: $name})
                SET d.ingested_at = $ingested_at,
                    d.num_chunks = $num_chunks,
                    d.num_triples = $num_triples,
                    d.status = $status
                """,
                name=name,
                ingested_at=ingested_at,
                num_chunks=num_chunks,
                num_triples=num_triples,
                status=status,
            )

    def get_documents(self) -> list[dict]:
        """Fetch all Document nodes, ordered by most recent first."""
        with self._session() as s:
            result = s.run(
                """
                MATCH (d:Document)
                RETURN d.name AS name,
                       d.ingested_at AS ingested_at,
                       d.num_chunks AS num_chunks,
                       d.num_triples AS num_triples,
                       d.status AS status
                ORDER BY d.ingested_at DESC
                """
            )
            return [dict(r) for r in result]

    def get_all_entity_names(self) -> list[dict]:
        """Fetch all Entity nodes with their names and normalized_names."""
        with self._session() as s:
            result = s.run(
                """
                MATCH (e:Entity)
                RETURN e.name AS name,
                       e.normalized_name AS normalized_name
                """
            )
            return [dict(r) for r in result]

    def merge_entity_into_canonical(
        self,
        duplicate_normalized: str,
        canonical_name: str,
    ) -> None:
        """Merge a duplicate entity into the canonical entity, preserving all relationships."""
        with self._session() as s:
            with s.begin_transaction() as tx:
                # Ensure canonical node exists
                tx.run(
                    """
                    MERGE (canon:Entity {normalized_name: $canon_norm})
                      ON CREATE SET canon.name = $canon_name
                    """,
                    canon_norm=canonical_name.lower().strip(),
                    canon_name=canonical_name,
                )
                # Copy outgoing relationships from duplicate to canonical
                tx.run(
                    """
                    MATCH (dup:Entity {normalized_name: $dup_norm})
                    MATCH (canon:Entity {normalized_name: $canon_norm})
                    MATCH (dup)-[r:RELATES_TO]->(target)
                    WHERE target <> canon
                    CREATE (canon)-[:RELATES_TO {
                        predicate: r.predicate,
                        chunk_text: r.chunk_text,
                        source_file: r.source_file,
                        chunk_index: r.chunk_index,
                        predicate_embedding: r.predicate_embedding
                    }]->(target)
                    """,
                    dup_norm=duplicate_normalized,
                    canon_norm=canonical_name.lower().strip(),
                )
                # Copy incoming relationships to canonical
                tx.run(
                    """
                    MATCH (dup:Entity {normalized_name: $dup_norm})
                    MATCH (canon:Entity {normalized_name: $canon_norm})
                    MATCH (source)-[r:RELATES_TO]->(dup)
                    WHERE source <> canon
                    CREATE (source)-[:RELATES_TO {
                        predicate: r.predicate,
                        chunk_text: r.chunk_text,
                        source_file: r.source_file,
                        chunk_index: r.chunk_index,
                        predicate_embedding: r.predicate_embedding
                    }]->(canon)
                    """,
                    dup_norm=duplicate_normalized,
                    canon_norm=canonical_name.lower().strip(),
                )
                # Delete duplicate and its edges
                tx.run(
                    "MATCH (dup:Entity {normalized_name: $dup_norm}) DETACH DELETE dup",
                    dup_norm=duplicate_normalized,
                )

    # ────────────────────────────────────────────────────────────────
    # Attribution queries
    # ────────────────────────────────────────────────────────────────

    def exact_match(
        self,
        subject: str,
        predicate: str,
        obj: str,
    ) -> Optional[dict]:
        """Return first relationship where all three match exactly (case-insensitive)."""
        s_norm = subject.lower().strip()
        o_norm = obj.lower().strip()
        p_norm = predicate.lower().strip()
        with self._session() as s:
            result = s.run(
                """
                MATCH (sub:Entity {normalized_name: $s_norm})
                      -[r:RELATES_TO]->
                      (obj:Entity {normalized_name: $o_norm})
                WHERE toLower(r.predicate) = $p_norm
                RETURN r.predicate AS predicate,
                       r.chunk_text AS chunk_text,
                       r.source_file AS source_file,
                       r.chunk_index AS chunk_index
                LIMIT 1
                """,
                s_norm=s_norm,
                o_norm=o_norm,
                p_norm=p_norm,
            )
            record = result.single()
            return dict(record) if record else None

    def semantic_fallback(
        self,
        subject: str,
        obj: str,
        query_embedding: list[float],
        top_k: int = 5,
    ) -> list[dict]:
        """
        Fetch all relationships between subject and object (both directions),
        compute cosine similarity against query_embedding in Python,
        return top_k ranked by similarity.
        """
        s_norm = subject.lower().strip()
        o_norm = obj.lower().strip()
        with self._session() as s:
            result = s.run(
                """
                MATCH (sub:Entity {normalized_name: $s_norm})
                      -[r:RELATES_TO]->
                      (obj:Entity {normalized_name: $o_norm})
                RETURN r.predicate AS predicate,
                       r.chunk_text AS chunk_text,
                       r.source_file AS source_file,
                       r.chunk_index AS chunk_index,
                       r.predicate_embedding AS pred_emb
                """,
                s_norm=s_norm,
                o_norm=o_norm,
            )
            rows = [dict(r) for r in result]

        if not rows:
            return []

        q = np.array(query_embedding, dtype=float)
        q_norm = np.linalg.norm(q) + 1e-8

        scored = []
        for row in rows:
            emb = row.get("pred_emb")
            if not emb:
                continue
            e = np.array(emb, dtype=float)
            sim = float(np.dot(q, e) / (q_norm * (np.linalg.norm(e) + 1e-8)))
            scored.append({**row, "similarity": sim})

        scored.sort(key=lambda x: x["similarity"], reverse=True)
        return scored[:top_k]

    def query_partial(
        self,
        subject: Optional[str] = None,
        obj: Optional[str] = None,
        predicate_embedding: Optional[list[float]] = None,
        top_k: int = 5,
        pool_size: int = 200,
    ) -> list[dict]:
        """
        Pattern query for partial triples (any of subject/object may be None).

        If predicate_embedding is provided, candidates are ranked by cosine
        similarity against the stored predicate embeddings.  Without it,
        results are returned in graph order, capped at top_k.
        """
        s_norm = subject.lower().strip() if subject else None
        o_norm = obj.lower().strip() if obj else None

        cypher = """
        MATCH (sub:Entity)-[r:RELATES_TO]->(obj:Entity)
        WHERE ($s_norm IS NULL OR sub.normalized_name = $s_norm)
          AND ($o_norm IS NULL OR obj.normalized_name = $o_norm)
        RETURN sub.name AS subject,
               obj.name AS object,
               r.predicate AS predicate,
               r.chunk_text AS chunk_text,
               r.source_file AS source_file,
               r.chunk_index AS chunk_index,
               r.predicate_embedding AS pred_emb
        LIMIT $pool
        """
        with self._session() as s:
            result = s.run(cypher, s_norm=s_norm, o_norm=o_norm, pool=pool_size)
            rows = [dict(r) for r in result]

        if not rows:
            return []

        if predicate_embedding is None:
            for r in rows:
                r["similarity"] = 0.0
            return rows[:top_k]

        q = np.array(predicate_embedding, dtype=float)
        q_norm = np.linalg.norm(q) + 1e-8

        scored = []
        for row in rows:
            emb = row.get("pred_emb")
            if not emb:
                continue
            e = np.array(emb, dtype=float)
            sim = float(np.dot(q, e) / (q_norm * (np.linalg.norm(e) + 1e-8)))
            scored.append({**row, "similarity": sim})

        scored.sort(key=lambda x: x["similarity"], reverse=True)
        return scored[:top_k]

    # ────────────────────────────────────────────────────────────────
    # Stats
    # ────────────────────────────────────────────────────────────────

    def server_info(self) -> dict:
        """Server URI + active database — for UI verification."""
        return {
            "uri": str(self._driver.address) if hasattr(self._driver, "address") else "n/a",
            "database": self._database,
        }

    def list_databases(self) -> list[str]:
        """Return all DB names visible to this connection (Enterprise/multi-DB only)."""
        try:
            with self._driver.session(database="system") as s:
                result = s.run("SHOW DATABASES YIELD name RETURN name")
                return [r["name"] for r in result]
        except Exception:
            return []

    def stats(self) -> dict:
        with self._session() as s:
            r = s.run(
                "MATCH (n:Entity) WITH count(n) AS nodes "
                "MATCH ()-[r:RELATES_TO]->() "
                "RETURN nodes, count(r) AS relations"
            )
            rec = r.single()
            if rec:
                return {"nodes": rec["nodes"], "relations": rec["relations"]}
            return {"nodes": 0, "relations": 0}

    def is_connected(self) -> bool:
        try:
            self._driver.verify_connectivity()
            return True
        except Exception:
            return False
