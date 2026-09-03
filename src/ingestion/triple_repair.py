"""
Triple repair — seconda chance per le triple bocciate dai guardrail.

I guardrail (subject=object, generic_node, unresolved_reference, ...) scartano
l'output corrotto di DeepSeek, ma molte di quelle triple sono RIPARABILI: il
fatto c'è, è la forma a essere rotta ("it" al posto dell'entità, soggetto
uguale all'oggetto, sostantivo generico al posto del nome proprio).  Invece di
buttarle, la pipeline le rimanda a DeepSeek — UNA chiamata per frase fallita —
con il motivo dello scarto tradotto in un'istruzione correttiva.  Le triple
riparate ripassano gli STESSI guardrail: se falliscono ancora, muoiono lì
(un solo round di repair, mai un loop).

Il costo è contenuto: si paga solo per le frasi che hanno prodotto almeno una
tripla scartata, e la cache LLM (`DeepSeekClient`) rende i ri-run gratuiti.
"""

from __future__ import annotations

import logging

from src.ingestion.deepseek_extractor import parse_response

logger = logging.getLogger(__name__)

# Motivo del guardrail -> istruzione correttiva per il modello.
REASON_HINTS = {
    "subject_equals_object":
        "subject and object are the same entity — a triple must relate two "
        "DIFFERENT things; re-read the sentence and find the real object, or "
        "drop the triple",
    "unresolved_reference":
        "subject or object is a pronoun or deictic phrase ('he', 'it', 'that "
        "same year') — replace it with the full entity name stated in the "
        "sentence, or drop the triple",
    "generic_node":
        "the subject is a bare common noun ('game', 'teams') that identifies "
        "nothing — use the specific named entity, date or number from the "
        "sentence, or drop the triple",
    "empty_subject": "subject is empty or meaningless — extract the real one",
    "empty_object": "object is empty or meaningless — extract the real one",
    "no_predicate":
        "predicate is empty — use a short relation that states the actual "
        "fact",
    "entity_not_in_sentence":
        "subject or object appears neither in the sentence nor in the title "
        "— use ONLY entities written there, never outside knowledge",
    "subject_is_claim":
        "subject is almost the whole sentence — an entity is a short noun "
        "phrase, not a clause",
    "object_is_claim":
        "object is almost the whole sentence — an entity is a short noun "
        "phrase, not a clause",
    "prepositional_object":
        "the object starts with a preposition ('in Ireland', 'on 25 "
        "September') — the preposition belongs to the PREDICATE, not to the "
        "node: move it there and leave the bare entity as the object",
    "conjunction_mention":
        "subject or object coordinates two different entities ('X and Y') — "
        "that is two nodes, not one: emit one triple per entity, repeating "
        "the predicate",
}

REPAIR_SYSTEM_PROMPT = """You are a precise information extraction engine. \
An earlier pass extracted RDF-style triples (subject, predicate, object) from \
ONE English sentence, but automatic validation REJECTED the triples listed \
below, each with the reason.

Fix each triple so that it passes validation, using ONLY what the sentence \
states. If the sentence does not support a corrected version, drop that \
triple. Do not re-emit a triple unchanged, and do not invent new facts.

Rules:
1. Subject and object must be named entities, dates or numbers that appear \
in the sentence. Never a pronoun, never a deictic phrase, never a bare \
common noun.
2. Subject and object must be different.
3. The predicate is a short lowercase relation (2-4 words, no articles).
4. The object carries no leading preposition — it belongs to the predicate.
   ("Alpha", "was founded", "in Berlin") -> ("Alpha", "was founded in", "Berlin")
5. One entity per field. A coordination is several entities: emit one triple \
per entity, repeating the predicate. A proper name that merely contains \
"and" ("Trinidad and Tobago") is ONE entity — do not split it.
6. Dropping a triple is always allowed and often the right fix. Returning \
fewer, sharper triples is better than forcing a bad one through.

Answer with a single JSON object, no markdown, no commentary:
{"triples": [{"subject": "...", "predicate": "...", "object": "..."}]}"""

REPAIR_USER_TEMPLATE = """Title: {title}

Sentence:
\"\"\"
{sentence}
\"\"\"

Rejected triples:
{rejected}

Return the corrected triples as JSON (empty list if none can be fixed)."""


def _format_rejected(rejected: list[dict]) -> str:
    lines = []
    for i, r in enumerate(rejected):
        hint = REASON_HINTS.get(r.get("reason", ""), r.get("reason", "invalid"))
        lines.append(
            '{}. ("{}", "{}", "{}")\n   problem: {}'.format(
                i, r["subject"], r["predicate"], r["obj"], hint)
        )
    return "\n".join(lines)


def build_repair_messages(
    sentence: str, title: str, rejected: list[dict],
) -> list[dict]:
    """Messaggi chat per riparare le triple bocciate di UNA frase."""
    return [
        {"role": "system", "content": REPAIR_SYSTEM_PROMPT},
        {"role": "user", "content": REPAIR_USER_TEMPLATE.format(
            title=title or "(untitled)",
            sentence=sentence,
            rejected=_format_rejected(rejected) or "(none)",
        )},
    ]


def repair_sentence(
    client, sentence: str, title: str, rejected: list[dict],
) -> list[dict]:
    """
    Una chiamata DeepSeek per una frase con triple bocciate.

    `rejected`: [{"subject", "predicate", "obj", "reason"}, ...].
    Ritorna le triple riparate come dict (stesso formato di `parse_response`);
    lista vuota su qualunque fallimento — il chiamante ha già loggato le
    bocciate, quindi qui non si perde nulla che non fosse già perso.
    """
    if not rejected:
        return []
    try:
        content = client.chat(
            build_repair_messages(sentence, title, rejected), json_mode=True)
    except Exception as exc:
        logger.error("Repair call fallita sulla frase %r: %s", sentence[:60], exc)
        return []
    return parse_response(content)
