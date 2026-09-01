"""
Centralized configuration — Claim Attribution LPG/Neo4j.

Pipeline:
 - Ingestion: ALCE/ASQA passages -> coreference -> triples (REBEL | DeepSeek) -> Neo4j
 - Attribution: claim -> REBEL parse -> exact match / semantic fallback cosine

Corpus is EXCLUSIVELY ALCE: no PDF/TXT loader (removed 2026-08-03).
"""

import os
from pathlib import Path

# -- Force ALL downloads/cache to D: — must precede any HF import ----------
HF_HOME = r"D:\hf_home"
SENTENCE_TRANSFORMERS_HOME = os.path.join(HF_HOME, "sentence_transformers")

os.environ["HF_HOME"] = HF_HOME
os.environ["HF_HUB_CACHE"] = os.path.join(HF_HOME, "hub")           # Model weights
os.environ["HUGGINGFACE_HUB_CACHE"] = os.path.join(HF_HOME, "hub")  # Legacy alias
# TRANSFORMERS_CACHE deliberately NOT set: it's deprecated (transformers
# warns to use HF_HOME) and, worse, points to a second cache dir separate
# from HF_HUB_CACHE — a corrupted/incomplete download there (e.g. from an
# interrupted run) is read instead of the good copy in HF_HUB_CACHE, with
# no fallback. Single cache root only.
os.environ["SENTENCE_TRANSFORMERS_HOME"] = SENTENCE_TRANSFORMERS_HOME
os.environ["HF_DATASETS_CACHE"] = os.path.join(HF_HOME, "datasets") # Dataset cache
# Offline once cached: skip the online freshness/etag check on every load.
# That check is what stalls (0 B for minutes) on a flaky connection even
# when the model is already fully cached. Set HF_ALLOW_ONLINE=1 in the
# environment to temporarily re-enable network (e.g. to pull a new model).
if not os.getenv("HF_ALLOW_ONLINE"):
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["TORCH_HOME"] = os.path.join(HF_HOME, "torch")           # PyTorch models
os.environ["XDG_CACHE_HOME"] = HF_HOME                              # Generic fallback

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# -- Secrets: .env in project root (gitignored) ----------------------------
# Keys read from .env if present, otherwise from system environment vars.
# python-dotenv is optional: without the package a minimal parser is used,
# so the project does not break if it is not installed.
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

# Ollama/Llama-3 removed from the active pipeline (2026-08-03): the only LLM
# is DeepSeek via API.  OLLAMA_* lives in legacy/legacy_settings.py alongside
# legacy/llama_generator.py.

# -- Neo4j -----------------------------------------------------------------
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "")
# Target database (Neo4j 4+). Keep as "neo4j" unless you created another.
# CRITICAL: must match the database your Neo4j Browser is connected to.
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "neo4j")

# -- REBEL — Triple Extraction (English, BART, fast) -----------------------
# Speed-optimised: rebel-large is ~1.5GB BART encoder-decoder, faster than
# mrebel-large (mBART ~2GB).  English-only — switch to Babelscape/mrebel-large
# + REBEL_SRC_LANG="it_IT" if multilingual ingest is required later.
REBEL_MODEL = "Babelscape/rebel-large"
REBEL_SRC_LANG = None            # only used when model name contains "mrebel"
REBEL_MAX_LENGTH = 256           # max new tokens generated
REBEL_BATCH_SIZE = 16            # batched forward; tune to GPU/CPU memory

# -- Triple extractors -----------------------------------------------------
# Each edge in Neo4j carries an `extractor` property: graphs from different
# extractors coexist without mixing (filter on MERGE, idempotency,
# and in ALL attribution queries).
EXTRACTOR_REBEL = "rebel"
EXTRACTOR_DEEPSEEK = "deepseek"
AVAILABLE_EXTRACTORS = [EXTRACTOR_REBEL, EXTRACTOR_DEEPSEEK]
ACTIVE_EXTRACTOR = os.getenv("ACTIVE_EXTRACTOR", EXTRACTOR_REBEL)

# -- DeepSeek (LLM extractor, OpenAI-compatible API) -----------------------
# API key: put it in `.env` in the project root (already in .gitignore):
#     DEEPSEEK_API_KEY=sk-...
# Never hard-code it here.
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
DEEPSEEK_TEMPERATURE = 0.0       # ALWAYS 0: deterministic extraction, never creative
DEEPSEEK_MAX_TOKENS = 1500
DEEPSEEK_TIMEOUT = 120           # seconds per request
DEEPSEEK_MAX_RETRIES = 3

# -- ALCE/ASQA corpus (sole ingestion source) ------------------------------
ALCE_DATA_PATH = os.getenv(
    "ALCE_DATA_PATH",
    r"D:\python_projects\rag\ALCE\data\asqa_eval_gtr_top100_reranked_oracle.json",
)
ALCE_DOCS_PER_ENTRY = 5          # top-5 passages already oracle-reranked

# Registry of already-processed doc_ids, per extractor (TSV).
# Needed to avoid reprocessing documents that produce ZERO triples:
# without edges in Neo4j, the idempotency check would miss them.
PROCESSED_REGISTRY_PATH = str(PROJECT_ROOT / "data" / "processed_ids.txt")

# -- Output persistence (JSONL) --------------------------------------------
# Every intermediate result (coref, triples, attribution) is saved here.
OUTPUT_DIR = str(PROJECT_ROOT / "data" / "outputs")

# -- Predicate Embedding (semantic fallback) --------------------------------
PREDICATE_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
SEMANTIC_THRESHOLD = 0.75       # cosine similarity threshold for fallback match
ENTITY_CLUSTER_THRESHOLD = 0.90 # cosine similarity threshold for entity clustering

# -- spaCy (coreference resolution) ----------------------------------------
SPACY_MODEL = "en_core_web_lg"  # Used by CoreferenceResolver (coreferee requires lg)

# Legacy in-generation/baseline settings (CERCA, ChromaDB, anime prompt)
# moved to legacy/legacy_settings.py — see legacy/ for baseline modules.
