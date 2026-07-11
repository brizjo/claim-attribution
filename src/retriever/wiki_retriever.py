"""
Modulo Retriever — Recupero documenti da HuggingFace Dataset con indicizzazione vettoriale.

Questo modulo gestisce:
1. Il download di dataset (es. RAG-RewardBench) tramite la libreria `datasets`
2. Il chunking dei documenti contestuali in segmenti di dimensione configurabile
3. L'indicizzazione dei chunk in un database vettoriale ChromaDB
4. Il recupero dei documenti più rilevanti per una data query (top-K)
"""

import hashlib
from typing import Optional

import chromadb
from chromadb.config import Settings as ChromaSettings
from sentence_transformers import SentenceTransformer

from config import settings

try:
    from datasets import load_dataset
except ImportError:
    load_dataset = None


class WikiRetriever:
    """
    Recupera e indicizza documenti da un dataset HuggingFace in un vector store ChromaDB.

    Workflow:
        1. fetch_articles()  → scarica il dataset configurato
        2. chunk_text()      → spezza ogni documento in chunk di N caratteri
        3. index_articles()  → embedda e salva i chunk in ChromaDB
        4. query()           → dato un testo, restituisce i top-K chunk simili
    """

    def __init__(
        self,
        embedding_model_name: str = settings.EMBEDDING_MODEL,
        collection_name: str = settings.CHROMA_COLLECTION_NAME,
        persist_directory: str = settings.CHROMA_PERSIST_DIR,
        top_k: int = settings.TOP_K_DOCUMENTS,
    ):
        self.top_k = top_k

        # ── Embedding model ────────────────────────────────────────
        self.embedding_model = SentenceTransformer(embedding_model_name)

        # ── ChromaDB client ────────────────────────────────────────
        self.chroma_client = chromadb.Client(
            ChromaSettings(
                persist_directory=persist_directory,
                anonymized_telemetry=False,
            )
        )
        self.collection = self.chroma_client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    # ────────────────────────────────────────────────────────────────
    # Fetch from HuggingFace
    # ────────────────────────────────────────────────────────────────

    def fetch_articles(self) -> list[dict[str, str]]:
        """
        Scarica il dataset HuggingFace e ne estrae il testo utile.

        Returns:
            Lista di dict con chiavi "title" e "text".
        """
        if load_dataset is None:
            raise ImportError(
                "datasets non è installato. "
                "Installa con: pip install datasets"
            )

        print(f"Loading dataset {settings.HF_DATASET_NAME}...")
        ds = load_dataset(settings.HF_DATASET_NAME, split=settings.HF_DATASET_SPLIT)
        
        articles = []
        limit = min(settings.MAX_DATASET_DOCS, len(ds))
        
        for i in range(limit):
            row = ds[i]
            # RAG-RewardBench and similar usually have 'context' or 'text' fields.
            text = row.get("context", row.get("text", row.get("passage", "")))
            title = row.get("title", row.get("question", f"Doc_{i}"))
            
            if text:
                articles.append({
                    "title": str(title)[:100], 
                    "text": str(text)
                })

        return articles

    # ────────────────────────────────────────────────────────────────
    # Chunking
    # ────────────────────────────────────────────────────────────────

    @staticmethod
    def chunk_text(
        text: str,
        chunk_size: int = settings.CHUNK_SIZE,
        overlap: int = settings.CHUNK_OVERLAP,
    ) -> list[str]:
        """
        Suddivide un testo in chunk di dimensione fissa con overlap.

        Args:
            text:       Testo sorgente.
            chunk_size: Numero massimo di caratteri per chunk.
            overlap:    Sovrapposizione tra chunk consecutivi.

        Returns:
            Lista di stringhe (chunk).
        """
        chunks = []
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]
            if chunk.strip():
                chunks.append(chunk.strip())
            start += chunk_size - overlap
        return chunks

    # ────────────────────────────────────────────────────────────────
    # Indexing
    # ────────────────────────────────────────────────────────────────

    def index_articles(self, articles: list[dict[str, str]]) -> int:
        """
        Indicizza gli articoli nel vector store ChromaDB.

        Args:
            articles: Lista di dict con "title" e "text".

        Returns:
            Numero totale di chunk indicizzati.
        """
        all_chunks: list[str] = []
        all_ids: list[str] = []
        all_metadatas: list[dict] = []

        for article in articles:
            chunks = self.chunk_text(article["text"])
            for i, chunk in enumerate(chunks):
                doc_id = hashlib.md5(
                    f"{article['title']}_{i}".encode()
                ).hexdigest()
                all_chunks.append(chunk)
                all_ids.append(doc_id)
                all_metadatas.append(
                    {"source": article["title"], "chunk_index": i}
                )

        if all_chunks:
            # Upsert in piccoli batch se all_chunks è molto grande
            batch_size = 5000
            for i in range(0, len(all_chunks), batch_size):
                batch_chunks = all_chunks[i:i+batch_size]
                batch_ids = all_ids[i:i+batch_size]
                batch_metas = all_metadatas[i:i+batch_size]
                
                embeddings = self.embedding_model.encode(batch_chunks).tolist()
                
                self.collection.upsert(
                    ids=batch_ids,
                    documents=batch_chunks,
                    embeddings=embeddings,
                    metadatas=batch_metas,
                )

        return len(all_chunks)

    def build_index(self) -> int:
        """
        Pipeline completa: scarica logica da HF → chunk → indicizza.

        Returns:
            Numero di chunk indicizzati.
        """
        articles = self.fetch_articles()
        return self.index_articles(articles)

    # ────────────────────────────────────────────────────────────────
    # Query / Retrieval
    # ────────────────────────────────────────────────────────────────

    def query(self, query_text: str, top_k: Optional[int] = None) -> list[dict]:
        """
        Recupera i top-K chunk miscelando Dense Similarity e exact Lexical Hits (Hybrid RAG).
        """
        k = top_k or self.top_k
        # Retrieve a broader pool to re-rank lexically
        pool_size = k * 10 
        query_embedding = self.embedding_model.encode([query_text]).tolist()

        try:
            results = self.collection.query(
                query_embeddings=query_embedding,
                n_results=pool_size,
                include=["documents", "metadatas", "distances"],
            )
        except chromadb.errors.NotEnoughElementsException:
            # Fallback if collection is too small
            try:
                results = self.collection.query(
                    query_embeddings=query_embedding,
                    n_results=k,
                    include=["documents", "metadatas", "distances"],
                )
            except chromadb.errors.NotEnoughElementsException:
                return []

        retrieved = []
        if results and results.get("documents") and len(results["documents"]) > 0 and results["documents"][0]:
            
            # Estrarre le parole chiave univoche della query ignorando stopwords (lunghezza <= 3)
            query_words = set(w.lower() for w in query_text.split() if len(w) > 3)

            for doc, meta, dist in zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
            ):
                # Normalized Dense Score roughly between 0 and 1
                dense_score = max(0.0, 1.0 - dist)
                
                # Unique Keyword Coverage Score
                doc_words = set(w.lower() for w in doc.split())
                unique_hits = len(query_words.intersection(doc_words))
                
                # Hybrid fusion: Add a 0.5 boost for each unique query word found.
                # This guarantees that a chunk containing "anime", "studio", AND "sunrise"
                # will absolutely crush a chunk containing only "anime" and "studio".
                hybrid_score = dense_score + (unique_hits * 0.5)
                
                retrieved.append(
                    {
                        "document": doc, 
                        "metadata": meta, 
                        "distance": dist,
                        "hybrid_score": hybrid_score
                    }
                )

        # Re-rank based on Hybrid Score
        retrieved.sort(key=lambda x: x["hybrid_score"], reverse=True)
        return retrieved[:k]
