"""
Recupero di JSON malformato dalle risposte LLM.

Il JSON mode di DeepSeek non garantisce output sintatticamente valido: nei run
reali capita un array non chiuso (`...}}` invece di `...}]}`) o una virgola di
troppo.  Con un `json.loads` secco il passaggio intero si perde in silenzio —
zero triple, zero verdetti, nessun errore visibile.

Due livelli di recupero, dal meno al più invasivo:

  1. `loads_block`  — isola il blocco `{...}` più esterno e prova il parse;
     se fallisce chiude parentesi/graffe rimaste aperte (scansione
     string-aware) e ritenta.
  2. `iter_objects` — scansione a profondità: restituisce ogni oggetto
     `{...}` bilanciato trovato nel testo, parsato singolarmente.  Un oggetto
     corrotto non trascina con sé gli altri.

Nessuna dipendenza esterna: la logica sta in ~60 righe e vale per qualunque
prompt del progetto.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Iterator

logger = logging.getLogger(__name__)

_TRAILING_COMMA = re.compile(r",\s*([}\]])")


def _scan(text: str) -> Iterator[tuple[int, str, int]]:
    """Itera `(indice, carattere, profondità)` ignorando ciò che sta in stringa."""
    in_string = False
    escaped = False
    depth = 0
    for i, ch in enumerate(text):
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
            continue
        if ch in "{[":
            depth += 1
            yield i, ch, depth
        elif ch in "}]":
            yield i, ch, depth
            depth -= 1


def _close_open_brackets(text: str) -> str:
    """Chiude le parentesi rimaste aperte, nell'ordine inverso di apertura."""
    stack: list[str] = []
    for _, ch, _ in _scan(text):
        if ch in "{[":
            stack.append(ch)
        elif stack and ((ch == "}" and stack[-1] == "{") or (ch == "]" and stack[-1] == "[")):
            stack.pop()
    return text + "".join("}" if ch == "{" else "]" for ch in reversed(stack))


def loads_block(raw: str) -> Any | None:
    """
    Parsa il blocco JSON più esterno di `raw`, riparando le chiusure mancanti.
    Ritorna `None` se nemmeno la riparazione produce JSON valido.
    """
    if not raw:
        return None
    text = raw.strip()
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end < start:
        return None
    block = text[start:end + 1]

    for candidate in (block, _TRAILING_COMMA.sub(r"\1", block)):
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass
        repaired = _close_open_brackets(candidate)
        if repaired != candidate:
            try:
                return json.loads(repaired)
            except json.JSONDecodeError:
                pass
    return None


def iter_objects(raw: str) -> list[dict]:
    """
    Ultima risorsa: ogni oggetto `{...}` bilanciato del testo, parsato da solo.

    Serve quando la struttura esterna è rotta ma i singoli record sono sani —
    il caso tipico dell'array non chiuso: si recuperano tutte le triple/verdetti
    invece di perdere il passaggio.
    """
    if not raw:
        return []

    found: list[tuple[int, int, dict]] = []
    open_at: list[int] = []
    for i, ch, _ in _scan(raw):
        if ch == "{":
            open_at.append(i)
        elif ch == "}" and open_at:
            start = open_at.pop()
            try:
                obj = json.loads(_TRAILING_COMMA.sub(r"\1", raw[start:i + 1]))
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                found.append((start, i, obj))

    # Si tengono solo gli oggetti più interni: se il wrapper esterno si è
    # chiuso correttamente conterrebbe gli stessi record una seconda volta.
    innermost = [
        (s, e, obj) for s, e, obj in found
        if not any(s < s2 and e2 < e for s2, e2, _ in found)
    ]
    return [obj for _, _, obj in sorted(innermost, key=lambda r: r[0])]


def records(raw: str, key: str) -> list[dict]:
    """
    Lista di record da una risposta LLM: `{key: [...]}`, oppure un array nudo,
    oppure — se il JSON è irrecuperabile — i singoli oggetti recuperabili.
    """
    data = loads_block(raw)
    if isinstance(data, dict):
        items = data.get(key, None)
        if items is None:
            # wrapper con nome diverso: prende la prima lista di oggetti.
            items = next(
                (v for v in data.values()
                 if isinstance(v, list) and all(isinstance(i, dict) for i in v)),
                None,
            )
        if isinstance(items, list):
            return [i for i in items if isinstance(i, dict)]
    if isinstance(data, list):
        return [i for i in data if isinstance(i, dict)]

    salvaged = iter_objects(raw)
    if salvaged:
        logger.warning("JSON malformato: recuperati %d record da parse parziale",
                       len(salvaged))
    return salvaged
