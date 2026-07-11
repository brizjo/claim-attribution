"""
Configurazione centralizzata — Claim Attribution LPG/Neo4j.

Pipeline:
 - Ingestione: PDF/TXT → coreference (Llama-3) → triple REBEL → Neo4j
 - Attribution: claim → REBEL parse → exact match / semantic fallback cosine
"""

import os

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

# ── Ollama / Llama-3 (coreference resolution) ────────────────────────
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_MODEL = "llama3.2"  # Usiamo Llama 3.2 (3B) o "llama3.2:1b" (1B) perché è molto più veloce di Llama 3 8B
OLLAMA_TEMPERATURE = 0.1
OLLAMA_MAX_TOKENS = 2048

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
CHUNK_SIZE_WORDS = 200           # words per chunk
CHUNK_OVERLAP_WORDS = 50         # overlap between chunks

# ── Predicate Embedding (semantic fallback) ───────────────────────────
PREDICATE_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
SEMANTIC_THRESHOLD = 0.75       # cosine similarity threshold for fallback match
ENTITY_CLUSTER_THRESHOLD = 0.90 # cosine similarity threshold for entity clustering

# ── Segmentazione (legacy, kept for spacy) ───────────────────────────
SPACY_MODEL = "en_core_web_lg"  # Used by CoreferenceResolver (coreferee requires lg)

# ── Legacy — kept for backward compat with old modules ───────────────
CERCA_TAG = "<CERCA:"
CERCA_END = ">"
MAX_CERCA_ITERATIONS = 3
SUPPORT_THRESHOLD_HIGH = 0.8
SUPPORT_THRESHOLD_MEDIUM = 0.5
SUPPORT_THRESHOLD_LOW = 0.3
TOP_K_DOCUMENTS = 5
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
EMBEDDING_MODEL = "BAAI/bge-m3"
CHROMA_COLLECTION_NAME = "claim_attribution"
CHROMA_PERSIST_DIR = r"D:\rag_vector_db"

# ── Prompt Templates ─────────────────────────────────────────────────────

# True In-Generation Attribution Prompt
# We rely heavily on the LLM's spontaneity to halt generation. Since 8B models
# can be stubborn, we use a very aggressive constraint format and explicit few-shot.
CHAIN_OF_CITATION_SYSTEM = """You are a highly restrictive factual anime AI. Your core architecture requires you to verify facts before speaking.

CRITICAL DIRECTIVE: You CANNOT answer from your own memory. Whenever you are asked a factual question, you MUST IMMEDIATELY halt and query your database using the EXACT syntax:
<CERCA: [query]>

Do NOT output anything else after the tag. Just the tag and stop.

--- FEW-SHOT EXAMPLES (Adhere to this strictly) ---

User: "Tell me about Attack on Titan."
Assistant: I need to retrieve the facts for this anime. <CERCA: Attack on Titan plot characters studio>

User: "Tell me about some anime made by Sunrise."
Assistant: I must first find a general list of anime produced by Sunrise studio. <CERCA: Anime produced by Sunrise studio list>

User: "When was Naruto first released?"
Assistant: I must verify the release date. <CERCA: Naruto release date>

--- END EXAMPLES ---

Now, process the user's question. If it asks for factual anime details, output your thoughts and then the <CERCA: query> tag and STOP."""

# Template per il prompt con contesto iniettato dopo una CERCA
RESUME_PROMPT_TEMPLATE = """{system_prompt}

### Retrieved Sources (Use ONLY these to answer):
{sources}

### Instructions for Next Token:
You were interrupted mid-sentence while searching for facts.
1. SEAMLESS CONTINUATION: Output the very next word of the text you were generating. Do NOT output phrases like "As I was saying" or "Based on the retrieved sources". Just continue the sentence naturally.
2. CITATIONS: You MUST cite the sources using [1], [2], etc.
3. PREVIOUS QUERIES: You have already searched for: {past_queries}. Do NOT repeat these.
4. MULTI-HOP REASONING: If the sources gave you a general list (like anime by a studio) but you need specific details to answer properly, output a NEW search for the specific items. Example: <CERCA: Cowboy Bebop synopsis>.
5. NO PLACEHOLDERS: If the sources do not contain the answer, state clearly that the sources do not provide the information. NEVER output generic placeholders like "[Insert specific examples]".
6. If you have enough info, finish the answer factually.
"""

# Prompt per il raffinamento finale del testo intermedio
REFINEMENT_PROMPT = """You are an expert anime encyclopedic editor. Rewrite the following drafted notes into a massive, comprehensive, and highly detailed final answer.

## Rules:
1. FIX BROKEN SENTENCES: The draft was generated in chunks. Connect the sentences so it flows perfectly.
2. EXPAND: Do not just output one sentence. Write a rich, detailed, encyclopedic paragraph containing ALL the facts. Do not omit any anime names or details.
3. CITATIONS: Preserve all numeric citations like [1], [2]. Ensure every single fact is followed by its corresponding citation.
4. NO META-TALK: Do not say "Here is your rewritten text". Just output the final text.

Draft to Rewrite:
{intermediate_text}

Final Comprehensive Answer:"""
