"""
Guardrail + repair nella pipeline PRINCIPALE (`AlceIngestor.extract_doc`).

Nessuna rete, nessun modello: estrattore, client LLM, resolver e anchorer
sono stub.  Copre le classi portate dagli esperimenti (2026-09-03):

  * tripla corrotta (subject=object) -> bocciata -> RIPARATA da DeepSeek
  * riparazione a sua volta corrotta -> muore al secondo giro (stage=repair)
  * estrattore senza `.client` (stub/esperimenti) -> nessun repair, solo scarto
  * coref che fallisce sul singolo passaggio -> NON uccide il passaggio
  * dedup su (S, P, O)
  * claim_span = frase ORIGINALE allineata alla frase risolta
"""

from __future__ import annotations

import pytest

from src.ingestion import guardrails, output_store
from src.ingestion.alce_ingestor import AlceIngestor
from src.ingestion.coref_resolver import CorefUnavailable
from src.ingestion.triple_extractor import Triple


# ── Stub ─────────────────────────────────────────────────────────────

class StubAnchorer:
    """Tutto ancorato tranne i sostantivi generici designati."""

    GENERIC = {"game", "teams", "players", "striker"}
    # Nomi propri che CONTENGONO "and" ma sono una sola entita': con lo
    # spaCy vero lo dice la NER, qui va dichiarato.
    SINGLE_ENTITIES = {"trinidad and tobago", "procter and gamble"}

    def is_anchored(self, entity: str, sentence: str) -> bool:
        return entity.lower().strip() not in self.GENERIC

    def is_single_entity(self, entity: str, sentence: str) -> bool:
        return entity.lower().strip() in self.SINGLE_ENTITIES


class StubResolver:
    def __init__(self, mapping: dict[str, str] | None = None, fail: bool = False):
        self._mapping = mapping or {}
        self._fail = fail

    def resolve(self, text: str) -> str:
        if self._fail:
            raise CorefUnavailable("fastcoref buggato (stub)")
        return self._mapping.get(text, text)


class StubClient:
    """Client LLM finto: risposta fissa per la chiamata di repair."""

    def __init__(self, response: str = '{"triples": []}'):
        self.response = response
        self.calls: list[list[dict]] = []

    def chat(self, messages, json_mode: bool = False) -> str:
        self.calls.append(messages)
        return self.response


class StubExtractor:
    name = "deepseek"

    def __init__(self, by_sentence: dict[str, list[tuple[str, str, str]]],
                 client: StubClient | None = None):
        self._by_sentence = by_sentence
        if client is not None:
            self.client = client

    def extract(self, chunks: list[dict]) -> list[Triple]:
        out = []
        for c in chunks:
            for s, p, o in self._by_sentence.get(c["text"], []):
                out.append(Triple(
                    subject=s, predicate=p, obj=o, chunk_text=c["text"],
                    source_file=c.get("source_file", ""),
                    chunk_index=c.get("chunk_index", 0),
                    source_id=c.get("source_id", ""), extractor=self.name,
                ))
        return out


# ── Fixture ──────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def tmp_outputs(tmp_path, monkeypatch):
    """I JSONL di output finiscono in tmp, mai in data/outputs."""
    monkeypatch.setattr(output_store, "_BASE", tmp_path)
    yield tmp_path


def make_ingestor(extractor, resolver=None) -> AlceIngestor:
    return AlceIngestor(
        client=None,
        extractor=extractor,
        resolver=resolver or StubResolver(),
        anchorer=StubAnchorer(),
    )


SENTENCE = "Barack Obama was born in Honolulu."
CHUNK = {"source_id": "42", "title": "Barack Obama", "text": SENTENCE,
         "chunk_index": 0}


# ── Test ─────────────────────────────────────────────────────────────

def test_corrupt_triple_repaired_by_deepseek(tmp_outputs):
    """subject=object -> bocciata -> il repair la corregge -> tenuta."""
    client = StubClient(
        '{"triples": [{"subject": "Barack Obama", "predicate": "born in", '
        '"object": "Honolulu"}]}'
    )
    extractor = StubExtractor(
        {SENTENCE: [("Barack Obama", "born in", "Barack Obama")]},
        client=client,
    )
    result = make_ingestor(extractor).extract_doc(CHUNK)

    assert result.error == ""
    assert len(client.calls) == 1                      # una call di repair
    assert result.repaired == 1
    assert [(t.subject, t.predicate, t.obj) for t in result.triples] == \
        [("Barack Obama", "born in", "Honolulu")]
    assert result.triples[0].chunk_text == SENTENCE    # testo originale
    assert result.triples[0].claim_span == SENTENCE

    assert len(result.discarded) == 1
    rec = result.discarded[0]
    assert rec["discard_reason"] == "subject_equals_object"
    assert rec["stage"] == "extract"
    # Loggata su JSONL
    assert (tmp_outputs / "triples_discarded.jsonl").exists()


def test_repair_output_rechecked_by_guardrails():
    """La riparazione con un pronome muore al secondo giro (stage=repair)."""
    client = StubClient(
        '{"triples": [{"subject": "Barack Obama", "predicate": "born in", '
        '"object": "he"}]}'
    )
    extractor = StubExtractor(
        {SENTENCE: [("Barack Obama", "born in", "Barack Obama")]},
        client=client,
    )
    result = make_ingestor(extractor).extract_doc(CHUNK)

    assert result.triples == []
    assert result.repaired == 0
    reasons = {(r["stage"], r["discard_reason"]) for r in result.discarded}
    assert reasons == {("extract", "subject_equals_object"),
                       ("repair", "unresolved_reference")}


def test_extractor_without_client_skips_repair():
    """Il Protocol non impone un client LLM: senza `.client` niente repair."""
    extractor = StubExtractor(
        {SENTENCE: [("Barack Obama", "born in", "Barack Obama")]})
    result = make_ingestor(extractor).extract_doc(CHUNK)

    assert result.triples == []
    assert result.repaired == 0
    assert [r["discard_reason"] for r in result.discarded] == \
        ["subject_equals_object"]


def test_generic_node_discarded():
    sentence = "Barack Obama watched the game in Honolulu."
    chunk = {**CHUNK, "text": sentence}
    extractor = StubExtractor(
        {sentence: [("Barack Obama", "watched game in", "Honolulu"),
                    ("game", "held in", "Honolulu")]})
    result = make_ingestor(extractor).extract_doc(chunk)

    assert [(t.subject, t.obj) for t in result.triples] == \
        [("Barack Obama", "Honolulu")]
    assert [r["discard_reason"] for r in result.discarded] == ["generic_node"]


def test_title_is_legitimate_context():
    """Il modello risolve "The iPhone" col titolo "iPhone (1st generation)":
    non va scartato per entity_not_in_sentence (caso reale, 7 triple perse)."""
    sentence = "The iPhone was officially announced on January 9, 2007."
    chunk = {"source_id": "14664751", "title": "iPhone (1st generation)",
             "text": sentence, "chunk_index": 0}
    extractor = StubExtractor(
        {sentence: [("iPhone (1st generation)", "announced on",
                     "January 9, 2007")]})
    result = make_ingestor(extractor).extract_doc(chunk)

    assert len(result.triples) == 1
    assert result.discarded == []


def test_copula_predicate_kept():
    """"Bican | was | footballer" e' un fatto classificatorio legittimo:
    no_predicate scatta solo su predicato VUOTO (caso reale)."""
    sentence = "Josef Bican was a Czech-Austrian professional footballer."
    chunk = {**CHUNK, "text": sentence}
    extractor = StubExtractor(
        {sentence: [("Josef Bican", "was",
                     "Czech-Austrian professional footballer")]})
    result = make_ingestor(extractor).extract_doc(chunk)

    assert len(result.triples) == 1
    assert result.discarded == []


def test_generic_object_kept_generic_subject_discarded():
    """generic_node solo sul SOGGETTO: "played as | striker" sopravvive,
    il soggetto-calamita "game" muore (casi reali)."""
    sentence = "Josef Bican watched the game and played as a striker."
    chunk = {**CHUNK, "text": sentence}
    extractor = StubExtractor(
        {sentence: [("Josef Bican", "played as", "striker"),
                    ("game", "watched by", "Josef Bican")]})
    result = make_ingestor(extractor).extract_doc(chunk)

    assert [(t.subject, t.obj) for t in result.triples] == \
        [("Josef Bican", "striker")]
    assert [r["discard_reason"] for r in result.discarded] == ["generic_node"]


def test_prepositional_object_discarded():
    """La preposizione appartiene al PREDICATO, non al nodo.

    "in Ireland" e "Republic of Ireland" sono la stessa entita' ma nessuno
    dei 4 stadi di canonicalizzazione li unifica quando la preposizione resta
    attaccata (containment fallisce, lessicale 0.62, coseno 0.65, soglie a
    0.90).  Caso reale, campione Irlanda."""
    sentence = "Fianna Fail was the largest party in Ireland until 2011."
    chunk = {**CHUNK, "text": sentence}
    extractor = StubExtractor(
        {sentence: [("Fianna Fail", "largest party until 2011", "in Ireland")]})
    result = make_ingestor(extractor).extract_doc(chunk)

    assert result.triples == []
    assert [r["discard_reason"] for r in result.discarded] == \
        ["prepositional_object"]


def test_conjunction_mention_discarded():
    """"X and Y" e' DUE nodi, non uno.

    Senza questo guardrail lo stadio 2 fondeva `Fine Gael` dentro
    `Fianna Fail and Fine Gael` (la testa `gael` combacia) ma non
    `Fianna Fail`: il partito in coda alla congiunzione spariva, quello in
    testa no.  Caso reale, campione Irlanda."""
    sentence = ("Fianna Fail and Fine Gael were on opposing sides of the "
                "Irish Civil War.")
    chunk = {**CHUNK, "text": sentence}
    extractor = StubExtractor(
        {sentence: [("Fianna Fail and Fine Gael", "were on opposing sides of",
                     "Irish Civil War")]})
    result = make_ingestor(extractor).extract_doc(chunk)

    assert result.triples == []
    assert [r["discard_reason"] for r in result.discarded] == \
        ["conjunction_mention"]


def test_proper_name_containing_and_is_not_a_conjunction():
    """"Trinidad and Tobago" contiene `and` ma e' UN nodo: a distinguerlo e'
    la NER, non la stringa."""
    anchorer = StubAnchorer()
    sentence = "Trinidad and Tobago qualified for the 2006 World Cup."
    assert not guardrails.is_coordination(
        "Trinidad and Tobago", sentence, anchorer)
    assert guardrails.is_coordination(
        "Fianna Fail and Fine Gael", sentence, anchorer)
    # Il titolo e' il secondo segnale, per quando la NER sbaglia.
    assert not guardrails.is_coordination(
        "Alpha and Beta", sentence, anchorer,
        title="Alpha and Beta national football team")


def test_comma_alone_is_not_a_coordination():
    """"January 9, 2007" e' una data, non due entita': senza congiunzione
    esplicita non si splitta (regressione: bocciava triple legittime)."""
    anchorer = StubAnchorer()
    sentence = "The iPhone was officially announced on January 9, 2007."
    assert not guardrails.is_coordination("January 9, 2007", sentence, anchorer)


def test_coref_failure_does_not_kill_passage():
    """Bug fastcoref sul passaggio -> testo originale + coref_failed, non errore."""
    extractor = StubExtractor(
        {SENTENCE: [("Barack Obama", "born in", "Honolulu")]})
    result = make_ingestor(
        extractor, resolver=StubResolver(fail=True)).extract_doc(CHUNK)

    assert result.error == ""
    assert result.coref_failed is True
    assert result.resolved_text == SENTENCE            # originale, non risolto
    assert len(result.triples) == 1


def test_duplicate_triples_deduped():
    extractor = StubExtractor(
        {SENTENCE: [("Barack Obama", "born in", "Honolulu"),
                    ("Barack Obama", "born in", "Honolulu")]})
    result = make_ingestor(extractor).extract_doc(CHUNK)

    assert len(result.triples) == 1
    assert [r["discard_reason"] for r in result.discarded] == ["duplicate"]
    assert result.discarded[0]["stage"] == "dedup"


def test_claim_span_aligned_to_original_sentence():
    """La tripla dalla frase risolta eredita la frase ORIGINALE come span."""
    original = "Barack Obama was born in Honolulu. He served as president."
    resolved = ("Barack Obama was born in Honolulu. "
                "Barack Obama served as president.")
    chunk = {**CHUNK, "text": original}
    extractor = StubExtractor({
        "Barack Obama was born in Honolulu.":
            [("Barack Obama", "born in", "Honolulu")],
        "Barack Obama served as president.":
            [("Barack Obama", "served as", "president")],
    })
    result = make_ingestor(
        extractor, resolver=StubResolver({original: resolved})
    ).extract_doc(chunk)

    spans = {(t.subject, t.predicate, t.obj): t.claim_span
             for t in result.triples}
    assert spans[("Barack Obama", "born in", "Honolulu")] == \
        "Barack Obama was born in Honolulu."
    # Lo span resta VERBATIM sull'originale anche se il modello ha visto
    # la frase coref-risolta.
    assert spans[("Barack Obama", "served as", "president")] == \
        "He served as president."
    for t in result.triples:
        assert t.claim_span in original
