"""
Legacy settings — baseline RAG "in-generation" (CERCA / Chain-of-Citation).

Spostato fuori da config/settings.py il 2026-07-18: la pipeline principale è
grafo (REBEL → Neo4j). Questi valori servono solo ai moduli in legacy/
(orchestrator, vector_retriever, wiki_retriever, highlight_renderer) e al
futuro baseline RAG vettoriale.

Ri-esporta anche le impostazioni attive (Neo4j, cache HF, ecc.) così i moduli
legacy possono importare questo file al posto di config.settings.
"""

from config.settings import *  # noqa: F401,F403 — Neo4j/HF cache attivi

# ── Ollama / Llama-3 (legacy) ────────────────────────────────────────
# Spostato qui dal config attivo il 2026-08-03: la pipeline non usa più
# Ollama.  L'unico LLM attivo è DeepSeek via API (src/llm/deepseek_client.py).
# Serve solo a legacy/llama_generator.py e legacy/orchestrator.py.
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_MODEL = "llama3.2"
OLLAMA_TEMPERATURE = 0.1
OLLAMA_MAX_TOKENS = 2048

# ── In-Generation Attribution (CERCA loop) ───────────────────────────
CERCA_TAG = "<CERCA:"
CERCA_END = ">"
MAX_CERCA_ITERATIONS = 3
SUPPORT_THRESHOLD_HIGH = 0.8
SUPPORT_THRESHOLD_MEDIUM = 0.5
SUPPORT_THRESHOLD_LOW = 0.3

# ── Retrieval vettoriale (ChromaDB baseline) ─────────────────────────
TOP_K_DOCUMENTS = 5
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
EMBEDDING_MODEL = "BAAI/bge-m3"
CHROMA_COLLECTION_NAME = "claim_attribution"
CHROMA_PERSIST_DIR = r"D:\rag_vector_db"

# ── Prompt Templates (dataset anime, in-generation) ──────────────────

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
