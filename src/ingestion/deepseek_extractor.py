"""
DeepSeek triple extractor — UNICO estrattore della pipeline (2026-09-03).

Interfaccia (la stessa che implementa anche `TripleExtractor`, oggi confinato
agli esperimenti):
    .name                          → "deepseek"
    .extract(chunks) -> list[Triple]

Trasporto HTTP e API key: `src/llm/deepseek_client.py` (`.env` nella root,
mai chiavi hardcodate). Qui vivono solo il prompt e il parsing.

Nota di costo: 1 chiamata per passaggio (~100 parole in, ~300 token out).
Con 5 passaggi per domanda ALCE il costo per domanda è trascurabile, ma
l'ingestione dell'intero dataset (948×5 = 4740 chiamate) va fatta a lotti.
"""

from __future__ import annotations

import logging
from config import settings
from src.ingestion.triple_extractor import Triple
from src.llm.deepseek_client import DeepSeekClient
from src.llm import json_repair

logger = logging.getLogger(__name__)


# ────────────────────────────────────────────────────────────────────
# Prompt
# ────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a precise information extraction engine. \
You convert an English encyclopedic passage into RDF-style triples \
(subject, predicate, object) for a knowledge graph.

This same extractor runs twice: once on source passages, once on generated \
answers. The two sets of triples are then matched against each other node by \
node. A triple whose subject or object does not NAME a specific thing can \
never be matched, so it is worse than no triple at all. Prefer FEWER and \
SHARPER triples over more triples.

Rules:
1. Extract ONLY facts stated explicitly in the passage. Never infer, \
complete, or use outside knowledge.
2. Subject and object must NAME something: a named entity, a date, a number \
or a specific quantity. A bare common noun phrase names nothing and is never \
acceptable in either field — "the government", "fourth place", "the match", \
"traditional values", "competing entities" are all invalid.
3. Resolve EVERY referring expression to the full name of what it denotes, \
using the passage and the Title. This covers pronouns (he, she, it, they) \
AND definite descriptions: "the country", "the company", "the band" must be \
written as the entity they refer to. If you cannot tell which entity a \
description denotes, drop the triple.
4. One entity per field. A coordination ("X and Y", "X, Y and Z") is several \
entities, not one: emit one triple per entity and repeat the predicate. \
Never write a coordination as a single subject or object.
   ("Alpha and Beta", "founded in", "1990")
   -> ("Alpha", "founded in", "1990") + ("Beta", "founded in", "1990")
   Exception: a proper name that merely contains "and" is ONE entity \
("Trinidad and Tobago", "Procter and Gamble") — do not split it.
5. The object carries no leading preposition; the preposition belongs to the \
predicate.
   ("Alpha", "was founded", "in Berlin") -> ("Alpha", "was founded in", "Berlin")
   ("Beta", "largest party until 2011", "in Ruritania")
   -> ("Beta", "largest party in until 2011", "Ruritania")
6. The object is an entity, not a clause. If stating the fact needs a whole \
clause, re-model it as entity-predicate-entity, or drop it.
   ("Alpha", "decided", "to enter negotiations with Beta on a merger")
   -> ("Alpha", "entered negotiations with", "Beta")
7. The predicate is a short lowercase verb phrase or relation name \
(2-4 words, no articles), e.g. "born in", "member of", "won", \
"has population", "located in".
8. For every triple, quote the shortest VERBATIM sentence from the passage \
that supports it, in "claim_span". Copy it character by character; do not \
paraphrase.
9. Skip meta-statements about the article itself, navigation text and \
sentences that assert nothing factual.
10. If the passage contains no extractable fact, return an empty list. An \
empty list is a valid and often correct answer.

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

def parse_response(content: str) -> list[dict]:
    """
    Estrae la lista di triple dalla risposta.

    Tollera markdown fence, testo attorno al JSON e JSON sintatticamente rotto
    (`src/llm/json_repair.py`): il JSON mode di DeepSeek in produzione emette a
    volte un array non chiuso, e un parse secco perderebbe l'intero passaggio.
    Ritorna [] solo se non si recupera nemmeno un record.
    """
    out = []
    for item in json_repair.records(content or "", "triples"):
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

    @property
    def client(self) -> DeepSeekClient:
        # Riusato da HybridExtractor: stesso trasporto, prompt diversi.
        return self._client

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
