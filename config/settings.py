"""
Configurazione centralizzata — Claim Attribution LPG/Neo4j.

Pipeline:
 - Ingestione: passaggi ALCE/ASQA → coreference → triple (REBEL | DeepSeek) → Neo4j
 - Attribution: claim → REBEL parse → exact match / semantic fallback cosine

Il corpus è ESCLUSIVAMENTE ALCE: nessun loader PDF/TXT (rimosso 2026-08-03).
"""

import os
from pathlib import Path

# ── Forza TUTTI i download/cache su D: — DEVE essere prima di qualsiasi import ──
HF_HOME = r"D:\hf_home"
SENTENCE_TRANSFORMERS_HOME = os.path.join(HF_HOME, "sentence_transformers")

os.environ["HF_HOME"] = HF_HOME
os.environ["HF_HUB_CACHE"] = os.path.join(HF_HOME, "hub")           # Model weights
os.environ["HUGGINGFACE_HUB_CACHE"] = os.path.join(HF_HOME, "hub")  # Legacy alias
os.environ["TRANSFORMERS_CACHE"] = os.path.join(HF_HOME, "transformers")
os.environ["SENTENCE_TRANSFORMERS_HOME"] = SENTENCE_TRANSFORMERS_HOME
os.environ["HF_DATASETS_CACHE"] = os.path.join(HF_HOME, "datasets") # Dataset cache
os.environ["TORCH_HOME"] = os.path.join(HF_HOME, "torch")           # PyTorch models
os.environ["XDG_CACHE_HOME"] = HF_HOME                              # Generic fallback

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ── Secrets: .env nella root del progetto (gitignored) ───────────────
# Chiavi lette da .env se presente, altrimenti dalle variabili d'ambiente
# di sistema.  python-dotenv è opzionale: senza il pacchetto si usa un
# parser minimale, così il progetto non si rompe se non è installato.
def _load_dotenv() -> None:
    env_path = PROJECT_ROOT / ".env"
    if not env_path.exists():
        return
    try:
        from dotenv import load_dotenv
        load_dotenv(env_path, override=False)
        return
    except ImportError:
        pass
    for line in env_path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


_load_dotenv()

# Ollama/Llama-3 rimosso dalla pipeline attiva (2026-08-03): l'unico LLM è
# DeepSeek via API.  OLLAMA_* vive in legacy/legacy_settings.py insieme a
# legacy/llama_generator.py.

# ── Neo4j ───────────────────────────────────────────────────────────
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "dr4gh3770")
# Target database (Neo4j 4+). Keep as "neo4j" unless you created another.
# CRITICAL: must match the database your Neo4j Browser is connected to.
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "neo4j")

# ── REBEL — Triple Extraction (English, BART, fast) ─────────────────
# Speed-optimised: rebel-large is ~1.5GB BART encoder-decoder, faster than
# mrebel-large (mBART ~2GB).  English-only — switch to Babelscape/mrebel-large
# + REBEL_SRC_LANG="it_IT" if multilingual ingest is required later.
REBEL_MODEL = "Babelscape/rebel-large"
REBEL_SRC_LANG = None            # only used when model name contains "mrebel"
REBEL_MAX_LENGTH = 256           # max new tokens generated
REBEL_BATCH_SIZE = 16            # batched forward; tune to GPU/CPU memory

# ── Estrattori di triple ─────────────────────────────────────────────
# Ogni arco in Neo4j porta la proprietà `extractor`: i grafi dei diversi
# estrattori coesistono senza mescolarsi (filtro in MERGE, idempotenza,
# e in TUTTE le query di attribution).
EXTRACTOR_REBEL = "rebel"
EXTRACTOR_DEEPSEEK = "deepseek"
AVAILABLE_EXTRACTORS = [EXTRACTOR_REBEL, EXTRACTOR_DEEPSEEK]
ACTIVE_EXTRACTOR = os.getenv("ACTIVE_EXTRACTOR", EXTRACTOR_REBEL)

# ── DeepSeek (estrattore LLM, API OpenAI-compatible) ─────────────────
# API key: mettila in `.env` nella root del progetto (già in .gitignore):
#     DEEPSEEK_API_KEY=sk-...
# Non hardcodarla qui.
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
DEEPSEEK_TEMPERATURE = 0.0       # SEMPRE 0: estrazione deterministica, mai creativa
DEEPSEEK_MAX_TOKENS = 1500
DEEPSEEK_TIMEOUT = 120           # secondi per richiesta
DEEPSEEK_MAX_RETRIES = 3

# ── Corpus ALCE/ASQA (unica sorgente di ingestione) ──────────────────
ALCE_DATA_PATH = os.getenv(
    "ALCE_DATA_PATH",
    r"D:\python_projects\rag\ALCE\data\asqa_eval_gtr_top100_reranked_oracle.json",
)
ALCE_DOCS_PER_ENTRY = 5          # top-5 passaggi già ri-rankati oracle

# Registro dei doc_id già processati, per estrattore (TSV).
# Serve a non riprocessare i documenti che producono ZERO triple:
# senza archi in Neo4j il check di idempotenza non li vedrebbe.
PROCESSED_REGISTRY_PATH = str(PROJECT_ROOT / "data" / "processed_ids.txt")

# ── Predicate Embedding (semantic fallback) ───────────────────────────
PREDICATE_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
SEMANTIC_THRESHOLD = 0.75       # cosine similarity threshold for fallback match
ENTITY_CLUSTER_THRESHOLD = 0.90 # cosine similarity threshold for entity clustering

# ── spaCy (coreference resolution) ───────────────────────────────────
SPACY_MODEL = "en_core_web_lg"  # Used by CoreferenceResolver (coreferee requires lg)

# Legacy in-generation/baseline settings (CERCA, ChromaDB, prompt anime)
# spostati in legacy/legacy_settings.py — vedi legacy/ per i moduli baseline.
