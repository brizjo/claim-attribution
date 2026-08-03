"""
DeepSeek triple extractor — controparte LLM di REBEL.

Stessa interfaccia di `TripleExtractor`:
    .name                          → "deepseek"
    .extract(chunks) -> list[Triple]

Trasporto HTTP e API key: `src/llm/deepseek_client.py` (`.env` nella root,
mai chiavi hardcodate). Qui vivono solo il prompt e il parsing.

Nota di costo: 1 chiamata per passaggio (~100 parole in, ~300 token out).
Con 5 passaggi per domanda ALCE il costo per domanda è trascurabile, ma
l'ingestione dell'intero dataset (948×5 = 4740 chiamate) va fatta a lotti.
"""

from __future__ import annotations

import json
import logging
import re
from config import settings
from src.ingestion.triple_extractor import Triple
from src.llm.deepseek_client import DeepSeekClient

logger = logging.getLogger(__name__)


# ────────────────────────────────────────────────────────────────────
# Prompt
# ────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a precise information extraction engine. \
You convert an English encyclopedic passage into RDF-style triples \
(subject, predicate, object) for a knowledge graph.

Rules:
1. Extract ONLY facts stated explicitly in the passage. Never infer, \
complete, or use outside knowledge.
2. Subject and object must be concrete named entities, dates, numbers or \
noun phrases as they appear in the passage. Never use pronouns \
(he, she, it, they, this) — always resolve them to the full entity name.
3. The predicate is a short lowercase verb phrase or relation name \
(2-4 words, no articles), e.g. "born in", "member of", "won", \
"has population", "located in".
4. For every triple, quote the shortest VERBATIM sentence from the passage \
that supports it, in "claim_span". Copy it character by character; do not \
paraphrase.
5. Skip meta-statements about the article itself, navigation text and \
sentences that assert nothing factual.
6. If the passage contains no extractable fact, return an empty list.

Answer with a single JSON object, no markdown, no commentary:
{"triples": [{"subject": "...", "predicate": "...", "object": "...", \
"claim_span": "..."}]}"""

USER_PROMPT_TEMPLATE = """Title: {title}

Passage:
\"\"\"
{text}
\"\"\"

Extract the triples as JSON."""


def build_messages(text: str, title: str = "") -> list[dict]:
    """Messaggi chat per un passaggio — esposto per test/debug del prompt."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": USER_PROMPT_TEMPLATE.format(title=title or "(untitled)", text=text),
        },
    ]


# ────────────────────────────────────────────────────────────────────
# Parsing risposta
# ────────────────────────────────────────────────────────────────────

_JSON_BLOCK = re.compile(r"\{.*\}", re.DOTALL)


def parse_response(content: str) -> list[dict]:
    """
    Estrae la lista di triple dalla risposta.

    Tollera markdown fence o testo attorno al JSON: si isola il blocco
    `{...}` più esterno prima del parse.  Ritorna [] su output non parsabile
    (loggato): un passaggio senza triple è un dato valido, non un crash.
    """
    if not content:
        return []
    raw = content.strip()
    if raw.startswith("```"):
        raw = raw.strip("`")
        raw = raw[raw.find("{"):] if "{" in raw else raw
    match = _JSON_BLOCK.search(raw)
    if not match:
        logger.warning("DeepSeek: nessun JSON nella risposta (%.120s)", raw)
        return []
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError as exc:
        logger.warning("DeepSeek: JSON non valido (%s) — %.120s", exc, raw)
        return []

    items = data.get("triples", data) if isinstance(data, dict) else data
    if not isinstance(items, list):
        return []

    out = []
    for item in items:
        if not isinstance(item, dict):
            continue
        subject = str(item.get("subject", "")).strip()
        predicate = str(item.get("predicate", "")).strip()
        obj = str(item.get("object", item.get("obj", ""))).strip()
        if not (subject and predicate and obj):
            continue
        out.append({
            "subject": subject,
            "predicate": predicate,
            "obj": obj,
            "claim_span": str(item.get("claim_span", "")).strip(),
        })
    return out


# ────────────────────────────────────────────────────────────────────
# Extractor
# ────────────────────────────────────────────────────────────────────

class DeepSeekExtractor:
    """Estrae triple via DeepSeek chat API (JSON mode)."""

    name = settings.EXTRACTOR_DEEPSEEK

    def __init__(self, client: DeepSeekClient | None = None, **client_kwargs):
        self._client = client or DeepSeekClient(**client_kwargs)

    @property
    def model(self) -> str:
        return self._client.model

    # ── Health check (delegato al client) ───────────────────────────

    def is_available(self) -> bool:
        return self._client.is_available()

    def check_connection(self) -> tuple[bool, str]:
        return self._client.check_connection()

    def _call(self, text: str, title: str = "") -> str:
        return self._client.chat(build_messages(text, title), json_mode=True)

    # ── API pubblica (stessa firma di TripleExtractor.extract) ──────

    def extract(self, chunks: list[dict]) -> list[Triple]:
        """
        Una chiamata API per chunk (nessun batching: l'API è per-messaggio).
        Un chunk che fallisce viene loggato e saltato — gli altri proseguono.
        """
        triples: list[Triple] = []
        for chunk in chunks:
            text = chunk.get("text", "")
            if not text.strip():
                continue
            try:
                content = self._call(text, chunk.get("title", chunk.get("source_file", "")))
            except Exception as exc:
                logger.error(
                    "DeepSeek: chunk %s (source_id=%s) fallito — %s",
                    chunk.get("chunk_index"), chunk.get("source_id"), exc,
                )
                continue

            seen: set[tuple[str, str, str]] = set()
            for item in parse_response(content):
                key = (item["subject"].lower(), item["predicate"].lower(), item["obj"].lower())
                if key in seen:
                    continue
                seen.add(key)
                triples.append(Triple(
                    subject=item["subject"],
                    predicate=item["predicate"],
                    obj=item["obj"],
                    chunk_text=text,
                    source_file=chunk.get("source_file", ""),
                    chunk_index=chunk.get("chunk_index", 0),
                    source_id=chunk.get("source_id", ""),
                    extractor=self.name,
                    # Lo span dell'LLM viene ri-ancorato sul testo ORIGINALE
                    # dall'ingestor (qui il chunk è coref-risolto).
                    claim_span=item["claim_span"],
                ))

        return triples
