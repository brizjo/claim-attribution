"""
Test della canonicalizzazione pre-write — i quattro stadi in isolamento sui
casi reali visti nei dati ALCE, lo scope, l'idempotenza e la conservazione
della `surface_form`.

Nessuna rete: il linker e' un doppio, l'encoder e' un doppio deterministico.
Il log su JSONL e' disattivato (`log=False`): i test non scrivono in
`data/outputs/`.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import settings  # noqa: E402
from src.ingestion.alce_ingestor import DocResult, IngestReport  # noqa: E402
from src.ingestion.entity_canonicalizer import (  # noqa: E402
    STAGE_EMBEDDING,
    STAGE_LEXICAL,
    STAGE_LINKING,
    STAGE_NORMALIZE,
    EntityCanonicalizer,
    Mention,
    SpotlightLinker,
    TitleLinker,
    is_initials_abbreviation,
    is_token_containment,
    normalize_mention,
)
from src.ingestion.triple_extractor import Triple  # noqa: E402


# ── doppi ─────────────────────────────────────────────────────────────

class StubEncoder:
    """
    Encoder deterministico: ogni testo appartiene a un "concetto"; testi dello
    stesso concetto hanno lo stesso vettore (coseno 1.0), gli altri sono
    ortogonali (coseno 0.0).  Registra i batch per verificare che l'encoding
    dei predicati sia UNO solo per domanda.
    """

    def __init__(self, concepts: dict[str, str] | None = None):
        self.concepts = concepts or {}
        self.batches: list[list[str]] = []

    def encode(self, texts, show_progress_bar=False):  # noqa: ARG002
        import numpy as np
        texts = list(texts)
        self.batches.append(texts)
        labels: list[str] = []
        for text in texts:
            labels.append(self.concepts.get(text, f"__{text}__"))
        vocabulary = sorted({*labels, *self.concepts.values()})
        vectors = np.zeros((len(texts), len(vocabulary) + 1), dtype="float32")
        for row, label in enumerate(labels):
            vectors[row][vocabulary.index(label)] = 1.0
        return vectors


class NoLinker:
    """Linker che non aggancia mai nulla — isola gli stadi 1, 2 e 4."""

    name = "none"

    def link(self, mention, titles, text=""):  # noqa: ARG002
        return None


class BrokenLinker:
    name = "broken"

    def link(self, mention, titles, text=""):  # noqa: ARG002
        raise RuntimeError("servizio di linking irraggiungibile")


def canonicalizer(**kwargs) -> EntityCanonicalizer:
    kwargs.setdefault("linker", NoLinker())
    kwargs.setdefault("encoder", StubEncoder())
    kwargs.setdefault("log", False)
    return EntityCanonicalizer(**kwargs)


def mention(text: str, source_id: str = "d0", sample_id: str = "q0",
            title: str = "") -> Mention:
    return Mention(text=text, source_id=source_id, sample_id=sample_id, title=title)


def triple(subject: str, obj: str, predicate: str = "member of",
           source_id: str = "d0") -> Triple:
    return Triple(
        subject=subject,
        predicate=predicate,
        obj=obj,
        chunk_text="testo del passaggio",
        source_file="Titolo",
        chunk_index=0,
        source_id=source_id,
        extractor="deepseek",
        claim_span="testo del passaggio",
    )


def report(sample_id: str, docs: list[tuple[str, str, list[Triple]]]) -> IngestReport:
    """docs = [(source_id, title, triples)]"""
    rep = IngestReport(extractor="deepseek", sample_id=sample_id, question="q?")
    for source_id, title, triples in docs:
        rep.docs.append(DocResult(
            source_id=source_id,
            title=title,
            chunk_index=0,
            original_text="testo del passaggio",
            sample_id=sample_id,
            triples=list(triples),
        ))
    return rep


def resolve(canon: EntityCanonicalizer, mentions) -> dict[str, object]:
    """Menzione -> Resolution, appiattendo la chiave di scope."""
    return {m: r for (_scope, m), r in canon.resolve(mentions).items()}


# ── Stadio 1: normalizzazione ─────────────────────────────────────────

@pytest.mark.parametrize("raw, expected", [
    ('Josef "Pepi" Bican (25 Sept 1913 – 12 Dec 2001)', "Josef Bican"),
    ("Soccer Statistics Foundation (RSSSF)", "Soccer Statistics Foundation"),
    ('Josef "Pepi" Bican', "Josef Bican"),
    ("The Sound of Silence", "Sound of Silence"),
    ("  Kiki   VanDeWeghe  ", "Kiki VanDeWeghe"),
])
def test_stage1_normalizes(raw, expected):
    assert normalize_mention(raw) == expected


def test_stage1_genitive_reduces_to_owner():
    # `Wilt Chamberlain's set` non e' un'entita': l'entita' e' il possessore.
    assert normalize_mention("Wilt Chamberlain's set") == "Wilt Chamberlain"
    assert normalize_mention("Campbell's call") == "Campbell"


def test_stage1_keeps_apostrophe_names():
    # Senza nome seguente l'apostrofo fa parte del nome: non si tocca.
    assert normalize_mention("Levi's") == "Levi's"
    assert normalize_mention("O'Brien") == "O'Brien"


def test_stage1_is_idempotent():
    for raw in ['Josef "Pepi" Bican (1913)', "The Beatles", "Campbell's call"]:
        once = normalize_mention(raw)
        assert normalize_mention(once) == once


def test_stage1_merges_mentions_that_normalize_alike():
    canon = canonicalizer()
    resolutions = resolve(canon, [
        mention('Josef "Pepi" Bican (25 Sept 1913 – 12 Dec 2001)'),
        mention("Josef Bican"),
    ])
    assert {r.canonical for r in resolutions.values()} == {"Josef Bican"}
    assert all(r.stage == STAGE_NORMALIZE for r in resolutions.values())


# ── Stadio 2: similarita' lessicale ───────────────────────────────────

def test_stage2_token_containment():
    canon = canonicalizer()
    resolutions = resolve(canon, [
        mention("VanDeWeghe"), mention("Kiki VanDeWeghe"),
    ])
    assert resolutions["VanDeWeghe"].canonical == "Kiki VanDeWeghe"
    assert resolutions["VanDeWeghe"].stage == STAGE_LEXICAL
    # La forma canonica non e' stata riscritta: si chiude allo stadio 1.
    assert resolutions["Kiki VanDeWeghe"].stage == STAGE_NORMALIZE


def test_stage2_initials_abbreviation():
    canon = canonicalizer()
    resolutions = resolve(canon, [
        mention("C. Ronaldo"), mention("Cristiano Ronaldo"),
    ])
    assert resolutions["C. Ronaldo"].canonical == "Cristiano Ronaldo"
    assert resolutions["C. Ronaldo"].stage == STAGE_LEXICAL
    assert is_initials_abbreviation("C. Ronaldo", "Cristiano Ronaldo")


def test_stage2_fuzzy_typo():
    canon = canonicalizer()
    resolutions = resolve(canon, [
        mention("Cristiano Ronaldo"), mention("Cristiano Ronalod"),
    ])
    assert len({r.canonical for r in resolutions.values()}) == 1
    merged = resolutions["Cristiano Ronalod"]
    assert merged.stage == STAGE_LEXICAL and merged.confidence >= 0.9


def test_stage2_does_not_merge_on_a_bare_number():
    assert not is_token_containment("1961", "1961 World Cup")
    canon = canonicalizer()
    resolutions = resolve(canon, [mention("1961"), mention("1961 World Cup")])
    assert len({r.canonical for r in resolutions.values()}) == 2


# ── Stadio 3: entity linking ──────────────────────────────────────────

def test_stage3_title_linker_unifies_without_threshold():
    # "Bican" e "Josef" non si contengono a vicenda: solo il titolo ALCE
    # ("Josef Bican") dice che sono la stessa entita'.
    canon = canonicalizer(linker=TitleLinker())
    resolutions = resolve(canon, [
        mention("Bican", source_id="d0", title="Josef Bican"),
        mention("Josef", source_id="d1", title="Josef Bican"),
    ])
    assert {r.canonical for r in resolutions.values()} == {"Josef Bican"}
    assert {r.external_id for r in resolutions.values()} == {"wikipedia:Josef Bican"}
    assert resolutions["Bican"].stage == STAGE_LINKING


def test_stage3_ambiguous_title_does_not_link():
    # Due titoli compatibili con la stessa menzione: meglio non agganciare.
    linker = TitleLinker()
    assert linker.link("Louise", ["Louise Brown", "Louise Smith"]) is None
    assert linker.link("Louise", ["Louise Brown"]) == "wikipedia:Louise Brown"


def test_stage3_broken_linker_degrades_without_failing():
    canon = canonicalizer(linker=BrokenLinker())
    resolutions = resolve(canon, [mention("Kiki VanDeWeghe"), mention("VanDeWeghe")])
    assert resolutions["VanDeWeghe"].canonical == "Kiki VanDeWeghe"
    assert all(r.external_id == "" for r in resolutions.values())


def test_spotlight_linker_degrades_explicitly(monkeypatch, caplog):
    """Il servizio di rete non risponde: log esplicito e disattivazione — mai
    un fallimento silenzioso (precedente: fastcoref)."""
    import types
    broken = types.SimpleNamespace(
        get=lambda *a, **k: (_ for _ in ()).throw(OSError("connessione rifiutata"))
    )
    monkeypatch.setitem(sys.modules, "requests", broken)
    linker = SpotlightLinker(url="http://localhost:1/annotate")
    with caplog.at_level("WARNING"):
        assert linker.link("Josef Bican", []) is None
    assert linker.available is False
    assert "Spotlight" in caplog.text


# ── Stadio 4: embedding ───────────────────────────────────────────────

def test_stage4_embedding_is_the_fallback():
    encoder = StubEncoder({"United States": "usa", "USA": "usa"})
    canon = canonicalizer(encoder=encoder, embedding_threshold=0.9)
    resolutions = resolve(canon, [mention("United States"), mention("USA")])
    assert len({r.canonical for r in resolutions.values()}) == 1
    merged = min(resolutions.values(), key=lambda r: len(r.mention))
    assert merged.stage == STAGE_EMBEDDING
    assert merged.confidence == pytest.approx(1.0)


def test_stage4_does_not_merge_unrelated_mentions():
    canon = canonicalizer(embedding_threshold=0.9)
    resolutions = resolve(canon, [mention("Honolulu"), mention("Nobel Peace Prize")])
    assert len({r.canonical for r in resolutions.values()}) == 2


# ── Scope ─────────────────────────────────────────────────────────────

def _louise_reports():
    q1 = report("q1", [("d1", "Louise Brown", [triple("Louise", "Bristol", source_id="d1")])])
    q2 = report("q2", [("d2", "Louise Smith", [triple("Louise", "Detroit", source_id="d2")])])
    return q1, q2


def test_scope_per_question_keeps_homonyms_apart():
    canon = canonicalizer(scope=settings.CANON_SCOPE_QUESTION, linker=TitleLinker())
    q1, q2 = _louise_reports()
    canon.canonicalize(q1)
    canon.canonicalize(q2)
    assert q1.docs[0].triples[0].subject == "Louise Brown"
    assert q2.docs[0].triples[0].subject == "Louise Smith"


def test_scope_global_collapses_homonyms():
    """Il rovescio della medaglia dello scope globale — documentato, non un bug."""
    canon = canonicalizer(scope=settings.CANON_SCOPE_GLOBAL)
    q1, q2 = _louise_reports()
    canon.canonicalize(q1)
    canon.canonicalize(q2)
    assert q1.docs[0].triples[0].subject == q2.docs[0].triples[0].subject


def test_scope_per_passage_does_not_unify_across_passages():
    canon = canonicalizer(scope=settings.CANON_SCOPE_PASSAGE)
    rep = report("q1", [
        ("d1", "Kiki VanDeWeghe", [triple("Kiki VanDeWeghe", "Denver Nuggets", source_id="d1")]),
        ("d2", "Kiki VanDeWeghe", [triple("VanDeWeghe", "Portland", source_id="d2")]),
    ])
    canon.canonicalize(rep)
    assert rep.docs[0].triples[0].subject == "Kiki VanDeWeghe"
    assert rep.docs[1].triples[0].subject == "VanDeWeghe"


def test_scope_per_question_unifies_across_passages():
    canon = canonicalizer(scope=settings.CANON_SCOPE_QUESTION)
    rep = report("q1", [
        ("d1", "Kiki VanDeWeghe", [triple("Kiki VanDeWeghe", "Denver Nuggets", source_id="d1")]),
        ("d2", "Kiki VanDeWeghe", [triple("VanDeWeghe", "Portland", source_id="d2")]),
    ])
    result = canon.canonicalize(rep)
    assert rep.docs[1].triples[0].subject == "Kiki VanDeWeghe"
    assert result.nodes_before > result.nodes_after


def test_unknown_scope_is_rejected():
    with pytest.raises(ValueError):
        canonicalizer(scope="per_corpus")


# ── Riscrittura delle triple, surface_form, idempotenza ───────────────

def test_surface_form_is_recoverable():
    canon = canonicalizer()
    rep = report("q1", [("d1", "Josef Bican", [
        triple('Josef "Pepi" Bican', "Rapid Wien", source_id="d1"),
    ])])
    canon.canonicalize(rep)
    written = rep.docs[0].triples[0]
    assert written.subject == "Josef Bican"
    assert written.subject_surface == 'Josef "Pepi" Bican'   # verbatim
    assert written.object_surface == "Rapid Wien"


def test_canonicalization_is_idempotent():
    canon = canonicalizer(scope=settings.CANON_SCOPE_QUESTION, linker=TitleLinker())
    rep = report("q1", [
        ("d1", "Kiki VanDeWeghe", [triple("Kiki VanDeWeghe", "Denver Nuggets", source_id="d1")]),
        ("d2", "Kiki VanDeWeghe", [triple("VanDeWeghe", "Portland", source_id="d2")]),
    ])
    first = canon.canonicalize(rep)
    snapshot = [
        (t.subject, t.obj, t.subject_surface, t.object_surface, t.subject_external_id)
        for doc in rep.docs for t in doc.triples
    ]
    second = canon.canonicalize(rep)
    assert snapshot == [
        (t.subject, t.obj, t.subject_surface, t.object_surface, t.subject_external_id)
        for doc in rep.docs for t in doc.triples
    ]
    # Il secondo giro non ri-embedda i predicati (li trova gia' pronti): il
    # resto del riepilogo e' identico.
    drop = "predicates_embedded"
    assert {k: v for k, v in first.summary().items() if k != drop} ==            {k: v for k, v in second.summary().items() if k != drop}
    assert second.summary()[drop] == 0


# ── Embedding dei predicati (spostato qui da GraphWriter) ─────────────

def test_predicate_embeddings_are_computed_once_per_question():
    encoder = StubEncoder()
    canon = canonicalizer(encoder=encoder)
    rep = report("q1", [
        ("d1", "T", [triple("A", "B", predicate="member of", source_id="d1"),
                     triple("C", "D", predicate="place of birth", source_id="d1")]),
        ("d2", "T", [triple("E", "F", predicate="member of", source_id="d2")]),
    ])
    result = canon.canonicalize(rep)

    assert result.predicates_embedded == 2      # predicati DISTINTI
    predicate_batches = [b for b in encoder.batches
                         if set(b) >= {"member of", "place of birth"}]
    assert len(predicate_batches) == 1          # un solo batch per domanda
    for doc in rep.docs:
        for t in doc.triples:
            assert t.predicate_embedding, "write_entry deve ricevere triple gia' embeddate"


def test_existing_embeddings_are_not_recomputed():
    encoder = StubEncoder()
    canon = canonicalizer(encoder=encoder)
    ready = triple("A", "B")._replace(predicate_embedding=(0.1, 0.2))
    rep = report("q1", [("d1", "T", [ready])])
    result = canon.canonicalize(rep)
    assert result.predicates_embedded == 0
    assert rep.docs[0].triples[0].predicate_embedding == (0.1, 0.2)


# ── Log / statistiche (il deliverable) ────────────────────────────────

def test_summary_counts_every_mention_by_stage():
    canon = canonicalizer(scope=settings.CANON_SCOPE_QUESTION)
    rep = report("q1", [
        ("d1", "Kiki VanDeWeghe", [triple("Kiki VanDeWeghe", "Denver Nuggets", source_id="d1")]),
        ("d2", "Kiki VanDeWeghe", [triple("VanDeWeghe", "Denver Nuggets", source_id="d2")]),
    ])
    result = canon.canonicalize(rep)
    summary = result.summary()
    assert summary["mentions"] == 4                       # 2 triple x 2 estremi
    assert sum(result.stage_counts.values()) == 4
    assert summary["stage_2_lexical"] == 1                # solo "VanDeWeghe"
    assert summary["nodes_before"] == 3 and summary["nodes_after"] == 2
    assert result.merged == 1


def test_records_carry_the_columns_the_analysis_needs():
    canon = canonicalizer()
    rep = report("q1", [("d1", "Josef Bican", [
        triple('Josef "Pepi" Bican', "Rapid Wien", source_id="d1"),
    ])])
    record = canon.canonicalize(rep).resolutions[0].as_record()
    assert set(record) >= {
        "mention", "canonical", "stage", "confidence",
        "source_id", "sample_id", "external_id",
    }
    assert record["sample_id"] == "q1" and record["source_id"] == "d1"


def test_ingest_report_carries_the_summary():
    canon = canonicalizer()
    rep = report("q1", [("d1", "T", [triple("A", "B", source_id="d1")])])
    assert rep.canonicalization == {}
    result = canon.canonicalize(rep)
    rep.canonicalization = result.summary()
    assert rep.canonicalization["scope"] == settings.CANON_SCOPE_QUESTION
