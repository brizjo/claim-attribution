"""
Entity canonicalizer — TERZA FASE della pipeline, fra estrazione e scrittura.

    extract_entry  ->  canonicalize_entry  ->  write_entry
    (triple grezze)    (nodi unificati +      (solo I/O)
                        predicate_embedding)

Perche' qui e non altrove:
  * durante l'estrazione il modello vede una frase alla volta e non puo' sapere
    che "VanDeWeghe" nella frase 3 e' "Kiki VanDeWeghe" della frase 1;
  * dopo la scrittura e' troppo tardi: unire nodi gia' in Neo4j
    (`Neo4jClient.merge_entity_into_canonical`, ora legacy) e' fragile e perde
    proprieta'.
Il momento giusto e' quando tutte le triple della domanda sono in memoria e
nessuna e' ancora scritta.

SCOPE (parametro, ablabile — `settings.CANONICALIZATION_SCOPE`)
  per_passage   troppo stretto: `Josef Bican` compare in 3 passaggi della stessa
                domanda e resterebbe 3 nodi (recall persa in attribution);
  per_question  DEFAULT: unifica dentro la domanda ALCE e le sue 5 evidenze;
  global        troppo largo: `Louise` (19 archi su 5 passaggi) collasserebbe
                persone diverse e darebbe un vantaggio di recall dovuto a
                evidenza fuori dallo scope della domanda — da dichiarare come
                limite se usato.

CASCATA A 4 STADI — ogni menzione entra ed esce come
`(forma_canonica, stadio_risolutore, confidenza)`; il primo stadio che risolve
vince.  Lo stadio 1 e' sempre applicato (e' la forma di partenza); gli stadi
2-4 servono a UNIRE menzioni diverse, quindi lo stadio registrato per una
menzione e' quello che l'ha fusa nel suo cluster (1 = mai fusa / capocluster).

  1. Normalizzazione deterministica (parentesi, soprannomi, genitivo sassone,
     articolo iniziale, unicode).
  2. Similarita' lessicale deterministica (contenimento di token, abbreviazioni
     con iniziale, fuzzy per refusi).
  3. Entity linking deterministico su ID esterno: prima il campo `title` del
     passaggio ALCE (titolo Wikipedia, gratuito e offline), poi un linker di
     rete opzionale che DEVE degradare esplicitamente (log + stadio 4).
  4. Embedding + soglia coseno: unico stadio non deterministico, ripiego.

Le decisioni discutibili sono documentate in `tasks/lessons.md`.
"""

from __future__ import annotations

import logging
import re
import unicodedata
from dataclasses import dataclass, field, replace
from difflib import SequenceMatcher
from typing import Iterable, Optional, Protocol, Sequence

from config import settings

logger = logging.getLogger(__name__)

STAGE_NORMALIZE = 1
STAGE_LEXICAL = 2
STAGE_LINKING = 3
STAGE_EMBEDDING = 4

STAGE_LABELS = {
    STAGE_NORMALIZE: "normalization",
    STAGE_LEXICAL: "lexical",
    STAGE_LINKING: "linking",
    STAGE_EMBEDDING: "embedding",
}


# ────────────────────────────────────────────────────────────────────
# Stadio 1 — normalizzazione deterministica
# ────────────────────────────────────────────────────────────────────

# `Josef Bican (25 Sept 1913 - 12 Dec 2001)`, `Soccer Statistics Foundation (RSSSF)`
_PARENTHETICAL = re.compile(r"\s*[\(\[][^)\]]*[\)\]]")
# `Josef "Pepi" Bican` — solo virgolette doppie: l'apostrofo singolo e' parte di
# nomi legittimi (O'Brien) e del genitivo, gestito a parte.
_NICKNAME = re.compile(r'\s*["“”][^"“”]{1,40}["“”]\s*')
# Genitivo sassone CON nome seguente: `Wilt Chamberlain's set`, `Campbell's call`.
# Il seguito ("set", "call") non e' un'entita': si riduce al possessore, che lo e'.
# `Levi's` / `McDonald's` (nessun token dopo) restano intatti: li' l'apostrofo fa
# parte del nome.
_GENITIVE = re.compile(r"^(?P<owner>.+?)'s\s+\S.*$")
_LEADING_ARTICLE = re.compile(r"^(the|a|an)\s+", re.IGNORECASE)
_WS = re.compile(r"\s+")
_TOKEN = re.compile(r"[0-9a-z]+")
_TRIM_CHARS = " \t\r\n,;:.-–—·"


def normalize_mention(text: str) -> str:
    """
    Stadio 1: forma normalizzata di una menzione.  Deterministica e idempotente
    (`normalize(normalize(x)) == normalize(x)`).
    """
    s = unicodedata.normalize("NFKC", text or "")
    s = s.replace("’", "'")
    s = _PARENTHETICAL.sub(" ", s)
    s = _NICKNAME.sub(" ", s)
    s = _WS.sub(" ", s).strip()
    m = _GENITIVE.match(s)
    if m:
        s = m.group("owner")
    s = _LEADING_ARTICLE.sub("", s)
    s = s.strip(_TRIM_CHARS)
    return _WS.sub(" ", s).strip()


def _tokens(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


def _key(text: str) -> str:
    """Chiave di identita' fra forme normalizzate (case-insensitive)."""
    return " ".join(_tokens(text))


# ────────────────────────────────────────────────────────────────────
# Stadio 2 — regole lessicali deterministiche
# ────────────────────────────────────────────────────────────────────

def is_token_containment(short: str, long: str) -> bool:
    """
    `VanDeWeghe` ⊂ `Kiki VanDeWeghe`.  Vincoli contro le fusioni facili:
    la forma corta deve avere almeno un token alfabetico di 3+ caratteri
    (esclude `1961` ⊂ `1961 World Cup` e le sigle di una lettera).
    """
    a, b = _tokens(short), _tokens(long)
    if not a or not b or len(a) >= len(b):
        return False
    if not any(len(t) >= 3 and not t.isdigit() for t in a):
        return False
    return set(a).issubset(set(b))


def is_initials_abbreviation(a: str, b: str) -> bool:
    """`C. Ronaldo` / `Cristiano Ronaldo`: stesso numero di token, ultimo token
    uguale, i token precedenti sono iniziali che combaciano."""
    ta, tb = _tokens(a), _tokens(b)
    if len(ta) != len(tb) or len(ta) < 2 or ta[-1] != tb[-1]:
        return False
    initials = False
    for x, y in zip(ta[:-1], tb[:-1]):
        if x == y:
            continue
        if len(x) == 1 and y.startswith(x):
            initials = True
        elif len(y) == 1 and x.startswith(y):
            initials = True
        else:
            return False
    return initials


def lexical_similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, _key(a), _key(b)).ratio()


# ────────────────────────────────────────────────────────────────────
# Stadio 3 — entity linking (aggancio a un ID esterno)
# ────────────────────────────────────────────────────────────────────

class EntityLinker(Protocol):
    """Due menzioni con lo stesso ID esterno sono la stessa entita', senza soglia."""

    name: str

    def link(self, mention: str, titles: Sequence[str], text: str = "") -> Optional[str]:
        ...


class TitleLinker:
    """
    Segnale gratuito: ogni passaggio ALCE porta gia' `title`, il titolo
    dell'articolo Wikipedia da cui e' preso (es. "Josef Bican").  E' un ID
    esterno affidabile per l'entita' principale del passaggio, offline e senza
    soglia.

    Aggancia una menzione a un titolo dello scope se i suoi token sono
    contenuti in quelli del titolo.  Se combaciano DUE titoli diversi la
    menzione e' ambigua e non viene linkata (meglio lo stadio 4 che una fusione
    sbagliata).
    """

    name = "title"

    def link(self, mention: str, titles: Sequence[str], text: str = "") -> Optional[str]:
        mk = _key(mention)
        if not mk:
            return None
        matches = set()
        for title in titles:
            tk = _key(normalize_mention(title))
            if not tk:
                continue
            if mk == tk or is_token_containment(mention, normalize_mention(title)):
                matches.add(title)
        if len(matches) == 1:
            return f"wikipedia:{normalize_mention(next(iter(matches)))}"
        return None


class SpotlightLinker:
    """
    DBpedia Spotlight (rete).  Opzionale: `ENTITY_LINKER=spotlight`.

    Degrada ESPLICITAMENTE — al primo fallimento logga un warning e si
    disattiva per il resto del run, lasciando le menzioni allo stadio 4.  Mai
    fallire in silenzio (precedente: fastcoref che saltava senza dirlo).
    """

    name = "spotlight"

    def __init__(
        self,
        url: str = settings.DBPEDIA_SPOTLIGHT_URL,
        confidence: float = settings.DBPEDIA_SPOTLIGHT_CONFIDENCE,
        timeout: int = settings.DBPEDIA_SPOTLIGHT_TIMEOUT,
    ):
        self._url = url
        self._confidence = confidence
        self._timeout = timeout
        self.available = True

    def link(self, mention: str, titles: Sequence[str], text: str = "") -> Optional[str]:
        if not self.available or not mention.strip():
            return None
        try:
            import requests
            response = requests.get(
                self._url,
                params={"text": text or mention, "confidence": self._confidence},
                headers={"Accept": "application/json"},
                timeout=self._timeout,
            )
            response.raise_for_status()
            resources = response.json().get("Resources") or []
        except Exception as exc:
            self.available = False
            logger.warning(
                "DBpedia Spotlight non disponibile (%s): entity linking "
                "disattivato per questo run, si degrada allo stadio 4 (embedding)",
                exc,
            )
            return None

        mk = _key(mention)
        for res in resources:
            surface = res.get("@surfaceForm", "")
            if _key(surface) == mk and res.get("@URI"):
                return str(res["@URI"])
        return None


def _external_label(external_id: str) -> str:
    """Etichetta leggibile di un ID esterno, se ce l'ha (`wikipedia:<titolo>`).
    Gli URI DBpedia non danno garanzie sul label: restano senza etichetta."""
    if external_id.startswith("wikipedia:"):
        return external_id.split(":", 1)[1].strip()
    return ""


class ChainLinker:
    """Prova i linker in ordine; il primo che risponde vince."""

    def __init__(self, linkers: Sequence[EntityLinker]):
        self._linkers = list(linkers)
        self.name = "+".join(l.name for l in self._linkers) or "none"

    def link(self, mention: str, titles: Sequence[str], text: str = "") -> Optional[str]:
        for linker in self._linkers:
            found = linker.link(mention, titles, text)
            if found:
                return found
        return None


def build_linker(name: str = settings.ENTITY_LINKER) -> Optional[EntityLinker]:
    """
    Factory: "title" (default, offline), "spotlight" (title + DBpedia), "none".

    Alternative valutate in `tasks/lessons.md`: spaCy `entityLinker` (richiede
    un KB scaricato e un modello aggiuntivo), Wikipedia search API (una query
    HTTP per menzione, ambigua sui nomi comuni).
    """
    name = (name or "").strip().lower()
    if name in ("", "none", "off"):
        return None
    if name == "title":
        return TitleLinker()
    if name == "spotlight":
        return ChainLinker([TitleLinker(), SpotlightLinker()])
    raise ValueError(f"Entity linker sconosciuto: {name!r} (title | spotlight | none)")


# ────────────────────────────────────────────────────────────────────
# Dati
# ────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Mention:
    """Una menzione come esce dall'estrazione, con il suo contesto."""

    text: str            # forma verbatim (surface_form)
    source_id: str = ""
    sample_id: str = ""
    title: str = ""


@dataclass
class Resolution:
    """Esito della cascata per una menzione."""

    mention: str
    canonical: str
    stage: int
    confidence: float
    source_id: str = ""
    sample_id: str = ""
    external_id: str = ""

    def as_record(self) -> dict:
        return {
            "mention": self.mention,
            "canonical": self.canonical,
            "stage": self.stage,
            "stage_label": STAGE_LABELS.get(self.stage, str(self.stage)),
            "confidence": round(self.confidence, 4),
            "source_id": self.source_id,
            "sample_id": self.sample_id,
            "external_id": self.external_id,
        }


@dataclass
class CanonicalizationResult:
    """Cosa e' successo su una domanda — e' il deliverable, non un accessorio."""

    scope: str
    resolutions: list[Resolution] = field(default_factory=list)
    nodes_before: int = 0
    nodes_after: int = 0
    triples: int = 0
    predicates_embedded: int = 0

    @property
    def stage_counts(self) -> dict[int, int]:
        counts = {s: 0 for s in STAGE_LABELS}
        for r in self.resolutions:
            counts[r.stage] = counts.get(r.stage, 0) + 1
        return counts

    @property
    def merged(self) -> int:
        """Menzioni chiuse da uno stadio di fusione (2-4)."""
        return sum(1 for r in self.resolutions if r.stage > STAGE_NORMALIZE)

    def summary(self) -> dict:
        total = len(self.resolutions) or 1
        counts = self.stage_counts
        return {
            "scope": self.scope,
            "mentions": len(self.resolutions),
            "triples": self.triples,
            "nodes_before": self.nodes_before,
            "nodes_after": self.nodes_after,
            "merged": self.merged,
            "predicates_embedded": self.predicates_embedded,
            **{
                f"stage_{s}_{STAGE_LABELS[s]}": counts.get(s, 0)
                for s in sorted(STAGE_LABELS)
            },
            **{
                f"stage_{s}_pct": round(100.0 * counts.get(s, 0) / total, 1)
                for s in sorted(STAGE_LABELS)
            },
        }


class _UnionFind:
    def __init__(self, n: int):
        self._parent = list(range(n))

    def find(self, x: int) -> int:
        while self._parent[x] != x:
            self._parent[x] = self._parent[self._parent[x]]
            x = self._parent[x]
        return x

    def union(self, x: int, y: int) -> bool:
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return False
        self._parent[rx] = ry
        return True

    def groups(self) -> dict[int, list[int]]:
        out: dict[int, list[int]] = {}
        for i in range(len(self._parent)):
            out.setdefault(self.find(i), []).append(i)
        return out


# ────────────────────────────────────────────────────────────────────
# Canonicalizer
# ────────────────────────────────────────────────────────────────────

class EntityCanonicalizer:
    """
    Unifica le menzioni di uno scope e prepara le triple per la scrittura.

    Non tocca Neo4j: riceve un `IngestReport` con triple in memoria e lo
    restituisce con `subject`/`obj` canonicalizzati, `*_surface` verbatim,
    `*_external_id` valorizzati e `predicate_embedding` gia' calcolato.
    """

    def __init__(
        self,
        scope: str = settings.CANONICALIZATION_SCOPE,
        linker: Optional[EntityLinker] = None,
        encoder=None,
        embedding_model: str = settings.PREDICATE_EMBEDDING_MODEL,
        lexical_threshold: float = settings.CANONICALIZATION_LEXICAL_THRESHOLD,
        embedding_threshold: float = settings.ENTITY_CLUSTER_THRESHOLD,
        embed_predicates: bool = True,
        log: bool = True,
    ):
        if scope not in settings.CANONICALIZATION_SCOPES:
            raise ValueError(
                f"Scope sconosciuto: {scope!r} "
                f"(disponibili: {settings.CANONICALIZATION_SCOPES})"
            )
        self.scope = scope
        self._linker = build_linker() if linker is None else linker
        self._encoder = encoder
        self._model_name = embedding_model
        self._lexical_threshold = lexical_threshold
        self._embedding_threshold = embedding_threshold
        self._embed_predicates = embed_predicates
        self._log = log
        # Scope `global`: le menzioni gia' viste restano nello stato e vengono
        # ri-considerate a ogni chiamata (e' quello che "globale" significa).
        self._seen: list[Mention] = []

    # ── encoder (lazy: il modello si carica solo se serve davvero) ──

    def _load_encoder(self):
        if self._encoder is None:
            from sentence_transformers import SentenceTransformer
            self._encoder = SentenceTransformer(self._model_name)
        return self._encoder

    def _encode(self, texts: Sequence[str]):
        return self._load_encoder().encode(list(texts), show_progress_bar=False)

    # ── API ────────────────────────────────────────────────────────

    def scope_key(self, mention: Mention) -> str:
        if self.scope == settings.CANON_SCOPE_PASSAGE:
            return mention.source_id
        if self.scope == settings.CANON_SCOPE_QUESTION:
            return mention.sample_id
        return ""

    def canonicalize(self, report, progress=None) -> CanonicalizationResult:
        """
        Canonicalizza in-place le triple di un `IngestReport`.

        Idempotente: la cascata riparte sempre dalla `surface_form`, quindi
        canonicalizzare due volte da' lo stesso risultato.
        """
        mentions: list[Mention] = []
        for doc in report.docs:
            for triple in doc.triples:
                context = Mention(
                    text="",
                    source_id=doc.source_id,
                    sample_id=getattr(report, "sample_id", "") or doc.sample_id,
                    title=doc.title,
                )
                for surface in (
                    triple.subject_surface or triple.subject,
                    triple.object_surface or triple.obj,
                ):
                    mentions.append(
                        Mention(
                            text=surface,
                            source_id=context.source_id,
                            sample_id=context.sample_id,
                            title=context.title,
                        )
                    )

        resolutions = self.resolve(mentions)
        result = CanonicalizationResult(scope=self.scope)
        result.nodes_before = len({_key(m.text) for m in mentions if m.text.strip()})
        result.nodes_after = len({_key(r.canonical) for r in resolutions.values()})
        # Una riga per OCCORRENZA (non per forma): il log deve dire in quale
        # passaggio la menzione e' comparsa — e' cio' che rende ispezionabili i
        # falsi merge e contabili gli archi per nodo.
        result.resolutions = [
            replace(
                resolutions[(self.scope_key(m), m.text)],
                source_id=m.source_id,
                sample_id=m.sample_id,
            )
            for m in mentions
            if (self.scope_key(m), m.text) in resolutions
        ]

        # Riscrittura delle triple: forma canonica + surface verbatim.
        for doc in report.docs:
            new_triples = []
            for triple in doc.triples:
                s_surface = triple.subject_surface or triple.subject
                o_surface = triple.object_surface or triple.obj
                s_res = resolutions.get(self._resolution_key(doc, report, s_surface))
                o_res = resolutions.get(self._resolution_key(doc, report, o_surface))
                new_triples.append(triple._replace(
                    subject=s_res.canonical if s_res else triple.subject,
                    obj=o_res.canonical if o_res else triple.obj,
                    subject_surface=s_surface,
                    object_surface=o_surface,
                    subject_external_id=s_res.external_id if s_res else "",
                    object_external_id=o_res.external_id if o_res else "",
                ))
            doc.triples = new_triples
            result.triples += len(new_triples)

        if self._embed_predicates:
            result.predicates_embedded = self._attach_predicate_embeddings(report)

        if self._log:
            from src.ingestion.output_store import save_canonicalization
            save_canonicalization([r.as_record() for r in result.resolutions])

        if progress:
            progress(
                f"canonicalizzazione ({self.scope}): {result.nodes_before} -> "
                f"{result.nodes_after} nodi, {result.merged} menzioni fuse"
            )
        return result

    def _resolution_key(self, doc, report, surface: str) -> tuple[str, str]:
        mention = Mention(
            text=surface,
            source_id=doc.source_id,
            sample_id=getattr(report, "sample_id", "") or doc.sample_id,
            title=doc.title,
        )
        return (self.scope_key(mention), surface)

    def resolve(self, mentions: Iterable[Mention]) -> dict[tuple[str, str], Resolution]:
        """Cascata a 4 stadi. Chiave del risultato: `(scope_key, menzione)`."""
        mentions = [m for m in mentions if (m.text or "").strip()]
        if self.scope == settings.CANON_SCOPE_GLOBAL:
            known = {(m.text, m.source_id, m.title) for m in self._seen}
            self._seen.extend(
                m for m in mentions if (m.text, m.source_id, m.title) not in known
            )
            pool = self._seen
        else:
            pool = mentions

        groups: dict[str, list[Mention]] = {}
        for mention in pool:
            groups.setdefault(self.scope_key(mention), []).append(mention)

        out: dict[tuple[str, str], Resolution] = {}
        wanted = {(self.scope_key(m), m.text) for m in mentions}
        for key, group in groups.items():
            for surface, resolution in self._resolve_group(group).items():
                if (key, surface) in wanted:
                    out[(key, surface)] = resolution
        return out

    # ── la cascata vera e propria, su un singolo scope ──────────────

    def _resolve_group(self, mentions: Sequence[Mention]) -> dict[str, Resolution]:
        # Stadio 1: normalizzazione. Menzioni diverse che normalizzano nella
        # stessa forma sono gia' la stessa entita'.
        forms: list[str] = []            # forma normalizzata rappresentativa
        index_of: dict[str, int] = {}    # chiave normalizzata -> indice forma
        surfaces: dict[str, int] = {}    # menzione verbatim -> indice forma
        titles = sorted({m.title for m in mentions if m.title})
        context_of: dict[int, Mention] = {}
        counts: dict[int, int] = {}      # quante volte la forma e' stata vista

        for mention in mentions:
            normalized = normalize_mention(mention.text)
            if not normalized:
                normalized = mention.text.strip()
            key = _key(normalized)
            idx = index_of.get(key)
            if idx is None:
                idx = len(forms)
                index_of[key] = idx
                forms.append(normalized)
                context_of[idx] = mention
            elif len(normalized) > len(forms[idx]):
                forms[idx] = normalized      # tiene la forma piu' informativa
            counts[idx] = counts.get(idx, 0) + 1
            surfaces[mention.text] = idx

        uf = _UnionFind(len(forms))
        merged_at: dict[int, tuple[int, float]] = {}

        def _merge(i: int, j: int, stage: int, confidence: float) -> None:
            if uf.union(i, j):
                merged_at.setdefault(i, (stage, confidence))
                merged_at.setdefault(j, (stage, confidence))

        # Stadio 2: similarita' lessicale, solo fra forme che condividono un
        # token (blocking: evita il quadratico su scope grandi).
        by_token: dict[str, list[int]] = {}
        for i, form in enumerate(forms):
            for token in set(_tokens(form)):
                by_token.setdefault(token, []).append(i)

        candidates: set[tuple[int, int]] = set()
        for indices in by_token.values():
            for a_pos, i in enumerate(indices):
                for j in indices[a_pos + 1:]:
                    candidates.add((min(i, j), max(i, j)))

        for i, j in sorted(candidates):
            if uf.find(i) == uf.find(j):
                continue
            a, b = forms[i], forms[j]
            if is_token_containment(a, b) or is_token_containment(b, a):
                _merge(i, j, STAGE_LEXICAL, 1.0)
                continue
            if is_initials_abbreviation(a, b):
                _merge(i, j, STAGE_LEXICAL, 0.95)
                continue
            ratio = lexical_similarity(a, b)
            if ratio >= self._lexical_threshold:
                _merge(i, j, STAGE_LEXICAL, ratio)

        # Stadio 3: entity linking. Stesso ID esterno = stessa entita', senza soglia.
        external_of: dict[int, str] = {}
        if self._linker is not None:
            for i, form in enumerate(forms):
                context = context_of.get(i)
                try:
                    external = self._linker.link(
                        form, titles, (context.title if context else "")
                    )
                except Exception as exc:  # un linker rotto non ferma la pipeline
                    logger.warning("Entity linker fallito su %r: %s", form, exc)
                    external = None
                if external:
                    external_of[i] = external

            by_external: dict[str, list[int]] = {}
            for i, external in external_of.items():
                by_external.setdefault(external, []).append(i)
            for indices in by_external.values():
                for i, j in zip(indices, indices[1:]):
                    if uf.find(i) != uf.find(j):
                        _merge(i, j, STAGE_LINKING, 1.0)

        # Stadio 4: embedding + soglia — ripiego, unico stadio non deterministico.
        roots = sorted({uf.find(i) for i in range(len(forms))})
        if len(roots) > 1 and self._embedding_threshold <= 1.0:
            try:
                import numpy as np
                vectors = np.asarray(
                    self._encode([forms[r] for r in roots]), dtype=np.float32
                )
                norms = np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-8
                vectors = vectors / norms
                sims = vectors @ vectors.T
                for a_pos in range(len(roots)):
                    for b_pos in range(a_pos + 1, len(roots)):
                        sim = float(sims[a_pos][b_pos])
                        if sim >= self._embedding_threshold:
                            _merge(roots[a_pos], roots[b_pos], STAGE_EMBEDDING, sim)
            except Exception as exc:
                logger.warning(
                    "Stadio 4 (embedding) non disponibile: %s — le menzioni "
                    "restano come lasciate dagli stadi 1-3", exc,
                )

        # Forma canonica per cluster: la piu' lunga (piu' informativa), come in
        # `entity_clusterer.py`, a parita' la prima incontrata.
        canonical_of: dict[int, str] = {}
        external_of_cluster: dict[int, str] = {}
        for root, indices in uf.groups().items():
            # Forma piu' lunga (piu' informativa); a parita' di lunghezza vince
            # la piu' frequente, poi la prima incontrata: mai l'ordine
            # alfabetico, che su un refuso ("Ronalod") sceglierebbe l'errore.
            best = max(indices, key=lambda i: (len(forms[i]), counts.get(i, 0), -i))
            canonical = forms[best]
            for i in indices:
                if i in external_of:
                    external_of_cluster[root] = external_of[i]
                    # Se l'ID esterno porta con se' un'etichetta (titolo
                    # Wikipedia), quella E' il nome canonico dell'entita':
                    # meglio della menzione piu' lunga vista nel testo.
                    label = _external_label(external_of[i])
                    if label:
                        canonical = label
                    break
            canonical_of[root] = canonical

        out: dict[str, Resolution] = {}
        for mention in mentions:
            idx = surfaces[mention.text]
            root = uf.find(idx)
            # Chi E' gia' la forma canonica del cluster si chiude allo stadio 1:
            # nessuno stadio successivo l'ha dovuta riscrivere.
            if _key(forms[idx]) == _key(canonical_of[root]):
                stage, confidence = STAGE_NORMALIZE, 1.0
            elif idx in merged_at:
                stage, confidence = merged_at[idx]
            elif idx in external_of:
                # Riscritta sull'etichetta dell'ID esterno pur senza fusioni:
                # e' lo stadio 3 ad aver risolto la menzione.
                stage, confidence = STAGE_LINKING, 1.0
            else:
                stage, confidence = STAGE_NORMALIZE, 1.0
            out[mention.text] = Resolution(
                mention=mention.text,
                canonical=canonical_of[root],
                stage=stage,
                confidence=confidence,
                source_id=mention.source_id,
                sample_id=mention.sample_id,
                external_id=external_of_cluster.get(root, ""),
            )
        return out

    # ── embedding dei predicati (era in GraphWriter, in fase di scrittura) ──

    def _attach_predicate_embeddings(self, report) -> int:
        """
        Un solo batch di encoding per domanda invece di uno per write.
        `write_entry` riceve triple gia' pronte e si limita a scrivere.
        """
        pending = sorted({
            triple.predicate
            for doc in report.docs
            for triple in doc.triples
            if triple.predicate and not triple.predicate_embedding
        })
        if not pending:
            return 0
        try:
            vectors = self._encode(pending)
        except Exception as exc:
            logger.warning(
                "Embedding dei predicati non calcolato (%s): GraphWriter lo "
                "ricalcolera' in scrittura", exc,
            )
            return 0

        table = {
            predicate: tuple(float(x) for x in vector)
            for predicate, vector in zip(pending, vectors)
        }
        for doc in report.docs:
            doc.triples = [
                t if t.predicate_embedding
                else t._replace(predicate_embedding=table.get(t.predicate, ()))
                for t in doc.triples
            ]
        return len(table)
