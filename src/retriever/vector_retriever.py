"""
Modulo Vector Retriever — Persistent Knowledge Base con ChromaDB e BGE-M3.

Implementa un database vettoriale persistente su disco:
  - Al primo avvio: scarica il dataset HuggingFace, chunka il testo,
    computa gli embeddings con BAAI/bge-m3, e salva il DB su D:.
  - Ai successivi avvii: carica il DB direttamente da disco.
    BGE-M3 viene usato solo per embeddare le brevi query di ricerca.

Questo modulo sostituisce wiki_retriever.py per il nuovo pipeline
In-Generation Attribution.
"""

import hashlib
import os
from pathlib import Path
from typing import Optional

import chromadb
from sentence_transformers import SentenceTransformer

from config import settings


class VectorRetriever:
    """
    Recupera documenti da un vector store ChromaDB persistente.

    Workflow al primo avvio:
        1. Scarica il dataset anime da HuggingFace
        2. Chunka il testo dei documenti
        3. Computa embeddings con BGE-M3
        4. Salva il DB persistente su disco (D:\\rag_vector_db)

    Workflow ai successivi avvii:
        - Carica il DB da disco (nessun download / nessun re-embedding)
        - BGE-M3 è usato solo per embeddare le query di ricerca (leggero)
    """

    def __init__(
        self,
        embedding_model_name: str = settings.EMBEDDING_MODEL,
        collection_name: str = settings.CHROMA_COLLECTION_NAME,
        persist_directory: str = settings.CHROMA_PERSIST_DIR,
        top_k: int = settings.TOP_K_DOCUMENTS,
    ):
        self.top_k = top_k
        self.persist_directory = persist_directory
        self.collection_name = collection_name

        # ── Embedding model (BGE-M3) ──────────────────────────────
        # Il modello viene scaricato in D:\hf_home\sentence_transformers
        # al primo utilizzo, poi caricato dalla cache locale.
        self.embedding_model = SentenceTransformer(
            embedding_model_name,
            cache_folder=settings.SENTENCE_TRANSFORMERS_HOME,
        )

        # ── ChromaDB Persistent Client ────────────────────────────
        # Usa PersistentClient (API moderna chromadb >= 0.4.0)
        os.makedirs(persist_directory, exist_ok=True)
        self.chroma_client = chromadb.PersistentClient(path=persist_directory)
        self.collection = self.chroma_client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

        # Flag: il DB è già popolato?
        self._db_ready = self.collection.count() > 0

    # ────────────────────────────────────────────────────────────────
    # Proprietà pubblica: lo stato del DB
    # ────────────────────────────────────────────────────────────────

    @property
    def is_ready(self) -> bool:
        """True se il vector DB è già popolato e pronto per le query."""
        return self._db_ready

    @property
    def document_count(self) -> int:
        """Numero totale di chunk nel DB."""
        return self.collection.count()

    # ────────────────────────────────────────────────────────────────
    # Auto-init: controlla se il DB esiste, altrimenti lo costruisce
    # ────────────────────────────────────────────────────────────────

    def ensure_ready(self, progress_callback=None) -> int:
        """
        Garantisce che il DB sia pronto. Se vuoto, esegue il build completo.

        Args:
            progress_callback: Funzione opzionale (message: str) per
                               aggiornare lo stato nell'UI.

        Returns:
            Numero di chunk nel DB.
        """
        if self._db_ready:
            if progress_callback:
                progress_callback(
                    f"✅ Vector DB caricato da disco: {self.document_count} chunks"
                )
            return self.document_count

        # DB vuoto → build completo
        if progress_callback:
            progress_callback("📥 Downloading dataset da HuggingFace...")

        articles = self._fetch_dataset()

        if progress_callback:
            progress_callback(
                f"✂️ Chunking di {len(articles)} documenti..."
            )

        n_chunks = self._index_articles(articles, progress_callback)
        self._db_ready = True

        if progress_callback:
            progress_callback(
                f"✅ Indicizzati {n_chunks} chunks con BGE-M3. "
                f"DB salvato in {self.persist_directory}"
            )

        return n_chunks

    # ────────────────────────────────────────────────────────────────
    # Fetch dal dataset HuggingFace
    # ────────────────────────────────────────────────────────────────

    @staticmethod
    def _safe_str(value, default: str = "") -> str:
        """Converte un valore in stringa pulita, gestendo None/NaN."""
        if value is None:
            return default
        s = str(value).strip()
        if s.lower() in ("none", "nan", "null", ""):
            return default
        return s

    def _build_comprehensive_document(self, row: dict, index: int) -> dict:
        """
        Costruisce un documento strutturato completo per un anime,
        includendo TUTTI i campi disponibili nel dataset.

        Questo è essenziale per testare rigorosamente la RAG attribution:
        ogni fatto (studio, episodi, anno, regista, ecc.) deve essere
        presente nel knowledge base affinché l'LLM possa citarlo e
        il sistema possa verificare la corrispondenza con ROUGE-L.

        Args:
            row:   Una riga del dataset HuggingFace.
            index: Indice della riga nel dataset.

        Returns:
            Dict con "title" e "text", oppure None se il documento è vuoto.
        """
        _s = self._safe_str

        # ── Core Identification ───────────────────────────────────
        main_title = _s(row.get("Main Title"), f"Anime_{index}")
        title_en = _s(row.get("Official Title (en)"))
        title_ja = _s(row.get("Official Title (ja)"))

        # ── Content ───────────────────────────────────────────────
        synopsis = _s(row.get("Synopsis"))
        tags = _s(row.get("processed_tags"))

        # ── Classification ────────────────────────────────────────
        anime_type = _s(row.get("filter_type"))       # TV, Movie, OVA, etc.
        year = _s(row.get("filter_year"))
        rating = _s(row.get("Max Rating"))

        # ── Production ────────────────────────────────────────────
        studio = _s(row.get("Animation Work"))
        director = _s(row.get("Direction"))
        music = _s(row.get("Music"))
        original_work = _s(row.get("Original Work"))

        # ── Episode Data ──────────────────────────────────────────
        episodes = _s(row.get("Episode"))
        duration = _s(row.get("Duration"))

        # ── Build structured document with explicit labels ────────
        # Ogni riga ha un label esplicito per massimizzare il
        # matching ROUGE-L durante l'audit di attribuzione.
        parts = []

        # Title block (always present)
        title_line = f"Title: {main_title}."
        if title_en and title_en != main_title:
            title_line += f" English Title: {title_en}."
        if title_ja:
            title_line += f" Japanese Title: {title_ja}."
        parts.append(title_line)

        # Classification facts
        facts = []
        if anime_type:
            facts.append(f"Type: {anime_type}")
        if year:
            facts.append(f"Year: {year}")
        if rating:
            facts.append(f"Rating: {rating}/10")
        if facts:
            parts.append(". ".join(facts) + ".")

        # Production facts
        prod = []
        if studio:
            prod.append(f"Studio: {studio}")
        if director:
            prod.append(f"Director: {director}")
        if music:
            prod.append(f"Music: {music}")
        if original_work:
            prod.append(f"Original Work: {original_work}")
        if prod:
            parts.append(". ".join(prod) + ".")

        # Episode facts
        ep_info = []
        if episodes:
            ep_info.append(f"Episodes: {episodes}")
        if duration:
            ep_info.append(f"Duration: {duration}")
        if ep_info:
            parts.append(". ".join(ep_info) + ".")

        # Synopsis (the main body)
        if synopsis:
            parts.append(f"Synopsis: {synopsis}")

        # Tags
        if tags:
            parts.append(f"Tags: {tags}.")

        text = " ".join(parts).strip()

        if not text or len(text) < 20:
            return None

        return {
            "title": str(main_title)[:200],
            "text": text,
        }

    def _fetch_dataset(self) -> list[dict[str, str]]:
        """
        Scarica il dataset anime da HuggingFace e costruisce documenti
        COMPRENSIVI con tutti i metadati disponibili.

        Ogni documento include: titoli (principale, EN, JA), tipo, anno,
        rating, studio, regista, musica, opera originale, episodi,
        durata, sinossi e tag. Questo garantisce che il knowledge base
        contenga tutti i fatti verificabili per un test rigoroso di
        RAG attribution.

        Returns:
            Lista di dict con chiavi "title" e "text".
        """
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError(
                "La libreria 'datasets' non è installata. "
                "Installa con: pip install datasets"
            )

        print(f"[VectorRetriever] Loading dataset: {settings.HF_DATASET_NAME}")
        ds = load_dataset(
            settings.HF_DATASET_NAME,
            split=settings.HF_DATASET_SPLIT,
        )

        articles = []
        limit = len(ds)
        if settings.MAX_DATASET_DOCS is not None:
            limit = min(settings.MAX_DATASET_DOCS, limit)

        for i in range(limit):
            row = ds[i]
            doc = self._build_comprehensive_document(row, i)
            if doc:
                articles.append(doc)

        print(
            f"[VectorRetriever] Built {len(articles)} comprehensive documents "
            f"from {limit} dataset rows"
        )
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
    # Indexing con BGE-M3
    # ────────────────────────────────────────────────────────────────

    def _index_articles(
        self,
        articles: list[dict[str, str]],
        progress_callback=None,
    ) -> int:
        """
        Chunka gli articoli, computa embeddings BGE-M3, e salva in ChromaDB.

        Args:
            articles:          Lista di dict con "title" e "text".
            progress_callback: Funzione opzionale per aggiornamenti UI.

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

        if not all_chunks:
            return 0

        # Embedding e upsert in batch (BGE-M3 è pesante, batch piccoli)
        batch_size = 256
        total = len(all_chunks)

        for start in range(0, total, batch_size):
            end = min(start + batch_size, total)
            batch_chunks = all_chunks[start:end]
            batch_ids = all_ids[start:end]
            batch_metas = all_metadatas[start:end]

            if progress_callback:
                progress_callback(
                    f"🧠 Embedding batch {start//batch_size + 1}/"
                    f"{(total + batch_size - 1)//batch_size} "
                    f"({end}/{total} chunks)..."
                )

            embeddings = self.embedding_model.encode(
                batch_chunks,
                show_progress_bar=False,
                normalize_embeddings=True,  # BGE-M3 raccomanda normalizzazione
            ).tolist()

            self.collection.upsert(
                ids=batch_ids,
                documents=batch_chunks,
                embeddings=embeddings,
                metadatas=batch_metas,
            )

        return total

    # ────────────────────────────────────────────────────────────────
    # Query / Retrieval
    # ────────────────────────────────────────────────────────────────

    def query(self, query_text: str, top_k: Optional[int] = None) -> list[dict]:
        """
        Recupera i top-K chunk più rilevanti per la query.

        L'embedding della query è leggero (singola stringa corta).

        Args:
            query_text: La domanda o query di ricerca.
            top_k:      Numero di risultati da restituire.

        Returns:
            Lista di dict con chiavi:
              - "document": testo del chunk
              - "metadata": metadati (source, chunk_index)
              - "distance": distanza coseno dal query embedding
              - "source_id": identificatore univoco della fonte
        """
        k = top_k or self.top_k
        query_embedding = self.embedding_model.encode(
            [query_text],
            normalize_embeddings=True,
        ).tolist()

        try:
            results = self.collection.query(
                query_embeddings=query_embedding,
                n_results=k,
                include=["documents", "metadatas", "distances"],
            )
        except Exception:
            return []

        retrieved = []
        if (
            results
            and results.get("documents")
            and len(results["documents"]) > 0
            and results["documents"][0]
        ):
            for doc, meta, dist in zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
            ):
                retrieved.append({
                    "document": doc,
                    "metadata": meta,
                    "distance": dist,
                    "source_id": f"{meta.get('source', 'Unknown')}_chunk{meta.get('chunk_index', 0)}",
                })

        return retrieved

    # ────────────────────────────────────────────────────────────────
    # Reset — per debug/rebuild
    # ────────────────────────────────────────────────────────────────

    def reset_db(self):
        """Elimina la collection e forza un rebuild al prossimo ensure_ready()."""
        try:
            self.chroma_client.delete_collection(self.collection_name)
        except Exception:
            pass
        self.collection = self.chroma_client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        self._db_ready = False
