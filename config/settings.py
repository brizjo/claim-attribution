"""
Configurazione centralizzata per il sistema RAG Claim Attribution.
"""

# ── Ollama / Llama-3 ────────────────────────────────────────────────
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_MODEL = "llama3"
OLLAMA_TEMPERATURE = 0.3
OLLAMA_MAX_TOKENS = 1024

# ── Retriever ───────────────────────────────────────────────────────
EMBEDDING_MODEL = "all-MiniLM-L6-v2"       # SentenceTransformer model
CHROMA_COLLECTION_NAME = "wikipedia_docs"
CHROMA_PERSIST_DIR = "./chroma_db"
TOP_K_DOCUMENTS = 5                         # Documenti da recuperare
CHUNK_SIZE = 500                            # Caratteri per chunk
CHUNK_OVERLAP = 50                          # Overlap tra chunk

# ── HuggingFace Dataset (replaces Wikipedia API) ──────────────────────
HF_DATASET_NAME = "jinzhuoran/RAG-RewardBench"
HF_DATASET_SPLIT = "train"
MAX_DATASET_DOCS = 1000  # Limite per evitare crash di memoria


# ── Segmentazione ──────────────────────────────────────────────────
SPACY_MODEL = "en_core_web_sm"

# ── Attribution Metrics ─────────────────────────────────────────────
BERTSCORE_MODEL = "distilbert-base-uncased"
BERTSCORE_LANG = "en"

# ── Soglie di supporto per il gradient highlight ────────────────────
SUPPORT_THRESHOLD_HIGH = 0.8      # Verde intenso
SUPPORT_THRESHOLD_MEDIUM = 0.5    # Giallo
SUPPORT_THRESHOLD_LOW = 0.3       # Arancione
# Sotto SUPPORT_THRESHOLD_LOW → Rosso (potenziale allucinazione)

# ── Pesi per lo score composito ─────────────────────────────────────
WEIGHT_BERTSCORE = 0.6
WEIGHT_LEXICAL_OVERLAP = 0.4

# ── Prompt Template ─────────────────────────────────────────────────
RAG_PROMPT_TEMPLATE = """You are a knowledgeable assistant. Answer the user's question based ONLY on the provided context documents.
If the context does not contain enough information to answer, say so explicitly.

### Context Documents:
{context}

### User Question:
{question}

### Answer:"""
