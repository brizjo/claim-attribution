"""
Entity clusterer — merges semantically-equivalent Entity nodes in Neo4j.
Uses SentenceTransformers embeddings + cosine similarity clustering.
"""

from __future__ import annotations

from typing import Optional, Callable
import numpy as np

from config import settings
from src.graph.neo4j_client import Neo4jClient


class EntityClusterer:
    """
    Clusters Entity nodes by semantic similarity and merges duplicates.

    Algorithm:
    1. Fetch all entity names from Neo4j
    2. Embed names with SentenceTransformer
    3. Compute pairwise cosine similarity matrix
    4. Greedily form clusters where similarity >= threshold
    5. Select canonical entity (longest name) per cluster
    6. Merge duplicates into canonical in Neo4j
    """

    def __init__(
        self,
        client: Neo4jClient,
        embedding_model: str = settings.PREDICATE_EMBEDDING_MODEL,
        similarity_threshold: float = settings.ENTITY_CLUSTER_THRESHOLD,
    ):
        self._client = client
        self._model_name = embedding_model
        self._threshold = similarity_threshold
        self._encoder = None

    def _load_encoder(self) -> None:
        if self._encoder is not None:
            return
        from sentence_transformers import SentenceTransformer
        self._encoder = SentenceTransformer(self._model_name)

    def cluster(
        self,
        progress_callback: Optional[Callable[[str], None]] = None,
    ) -> int:
        """
        Find and merge duplicate entities.

        Returns:
            Number of merge operations performed.
        """
        self._load_encoder()

        # Fetch all entities
        entities = self._client.get_all_entity_names()
        if not entities:
            return 0

        names = [e["name"] for e in entities]
        normalized_names = [e["normalized_name"] for e in entities]

        # Encode all names
        embeddings = self._encoder.encode(names, show_progress_bar=False)

        # Compute pairwise cosine similarity (upper triangle only)
        embeddings = np.array(embeddings, dtype=np.float32)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8
        normalized_emb = embeddings / norms

        # Find pairs above threshold using union-find
        parent = list(range(len(entities)))

        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]

        def union(x, y):
            px, py = find(x), find(y)
            if px != py:
                parent[px] = py

        # Only check upper triangle (i < j)
        for i in range(len(entities)):
            for j in range(i + 1, len(entities)):
                sim = float(np.dot(normalized_emb[i], normalized_emb[j]))
                if sim >= self._threshold:
                    union(i, j)

        # Group entities by cluster
        clusters = {}
        for i in range(len(entities)):
            root = find(i)
            if root not in clusters:
                clusters[root] = []
            clusters[root].append(i)

        # Merge duplicates in each cluster (only if cluster size > 1)
        merge_count = 0
        for cluster_indices in clusters.values():
            if len(cluster_indices) > 1:
                # Select canonical = longest name in cluster
                canonical_idx = max(cluster_indices, key=lambda i: len(names[i]))
                canonical_name = names[canonical_idx]

                # Merge all others into canonical
                for idx in cluster_indices:
                    if idx != canonical_idx:
                        dup_normalized = normalized_names[idx]
                        self._client.merge_entity_into_canonical(
                            dup_normalized, canonical_name
                        )
                        merge_count += 1
                        if progress_callback:
                            progress_callback(
                                f"Merged {names[idx]} → {canonical_name}"
                            )

        return merge_count
