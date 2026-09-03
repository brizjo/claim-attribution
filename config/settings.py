"""
Centralized configuration — Claim Attribution LPG/Neo4j.

Pipeline:
 - Ingestion: ALCE/ASQA passages -> coreference -> triples (DeepSeek) -> Neo4j
 - Attribution: claim -> DeepSeek parse -> exact match / semantic fallback cosine

REBEL e' FUORI dalla pipeline principale (2026-09-03): l'unico estrattore di
triple e' DeepSeek.  I moduli REBEL restano solo per gli esperimenti
(`src/ui/experiments.py`, `scripts/run_hybrid_experiment.py`), che esistono
proprio per misurare quanto REBEL aggiunge.

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

# -- REBEL — SOLO ESPERIMENTI (fuori dalla pipeline principale) ------------
# Parametri usati unicamente da `src/ingestion/triple_extractor.py`, che oggi
# serve la pipeline ibrida degli esperimenti (varianti A/D) e non l'ingestione.
# English-only — switch to Babelscape/mrebel-large + REBEL_SRC_LANG="it_IT"
# if multilingual extraction is required later.
REBEL_MODEL = "Babelscape/rebel-large"
REBEL_SRC_LANG = None            # only used when model name contains "mrebel"
REBEL_MAX_LENGTH = 256           # max new tokens generated
REBEL_BATCH_SIZE = 16            # batched forward; tune to GPU/CPU memory

# -- Triple extractors -----------------------------------------------------
# Each edge in Neo4j carries an `extractor` property: graphs from different
# extractors coexist without mixing (filter on MERGE, idempotency,
# and in ALL attribution queries).
# `EXTRACTOR_REBEL` resta come ETICHETTA: identifica gli archi scritti dai
# grafi vecchi, che restano interrogabili. Non e' piu' un estrattore
# selezionabile: la pipeline principale ha un solo estrattore, DeepSeek.
EXTRACTOR_REBEL = "rebel"
EXTRACTOR_DEEPSEEK = "deepseek"
AVAILABLE_EXTRACTORS = [EXTRACTOR_DEEPSEEK]
ACTIVE_EXTRACTOR = os.getenv("ACTIVE_EXTRACTOR", EXTRACTOR_DEEPSEEK)

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

# -- Cache risposte LLM (riproducibilita' degli esperimenti) ---------------
# temperature=0 NON garantisce determinismo lato API: due run sullo stesso
# prompt hanno dato conteggi di triple diversi. La cache su disco, indicizzata
# per hash del prompt, rende un ri-run identico al precedente (e gratuito).
# LLM_CACHE=0 nell'ambiente la disattiva (per forzare chiamate fresche).
LLM_CACHE_ENABLED = os.getenv("LLM_CACHE", "1") not in ("0", "false", "False")
LLM_CACHE_DIR = str(PROJECT_ROOT / "data" / "cache" / "llm")

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

# -- Canonicalizzazione delle entita' (fase fra extract e write) ------------
# Terza fase della pipeline: le triple di una domanda sono tutte in memoria e
# nessuna e' ancora scritta -> e' il momento giusto per unificare i nodi.
# Lo scope e' un parametro perche' e' ablabile in tesi:
#   per_passage  — troppo stretto: la stessa entita' in 3 passaggi resta 3 nodi
#   per_question — default: unifica dentro la domanda (le sue 5 evidenze)
#   global       — troppo largo: collassa omonimi ("Louise") di domande diverse
CANON_SCOPE_PASSAGE = "per_passage"
CANON_SCOPE_QUESTION = "per_question"
CANON_SCOPE_GLOBAL = "global"
CANONICALIZATION_SCOPES = [
    CANON_SCOPE_PASSAGE, CANON_SCOPE_QUESTION, CANON_SCOPE_GLOBAL,
]
CANONICALIZATION_SCOPE = os.getenv("CANONICALIZATION_SCOPE", CANON_SCOPE_QUESTION)
# Stadio 2 (lessicale): soglia difflib per refusi/varianti. Il contenimento di
# token e le abbreviazioni con iniziale sono regole esatte, non usano la soglia.
CANONICALIZATION_LEXICAL_THRESHOLD = 0.90
# Stadio 3 (entity linking): "title" usa il campo `title` del passaggio ALCE
# (titolo Wikipedia, gratuito e offline); "spotlight" aggiunge DBpedia
# Spotlight (rete, degrada allo stadio 4 se non risponde); "none" lo disattiva.
ENTITY_LINKER = os.getenv("ENTITY_LINKER", "title")
DBPEDIA_SPOTLIGHT_URL = os.getenv(
    "DBPEDIA_SPOTLIGHT_URL", "https://api.dbpedia-spotlight.org/en/annotate"
)
DBPEDIA_SPOTLIGHT_CONFIDENCE = 0.5
DBPEDIA_SPOTLIGHT_TIMEOUT = 10   # secondi

# -- UI: tab esperimenti ---------------------------------------------------
# Gli esperimenti (pipeline ibrida A/D, cache LLM, run batch) sono strumenti di
# sviluppo, non funzionalita' del sistema: nascosti se non richiesti.
SHOW_EXPERIMENTS = os.getenv("SHOW_EXPERIMENTS", "0") not in ("", "0", "false", "False")

# -- spaCy (coreference resolution) ----------------------------------------
SPACY_MODEL = "en_core_web_lg"  # Used by CoreferenceResolver (coreferee requires lg)

# Legacy in-generation/baseline settings (CERCA, ChromaDB, anime prompt)
# moved to legacy/legacy_settings.py — see legacy/ for baseline modules.
