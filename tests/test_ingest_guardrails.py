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

from src.ingestion import output_store
from src.ingestion.alce_ingestor import AlceIngestor
from src.ingestion.coref_resolver import CorefUnavailable
from src.ingestion.triple_extractor import Triple


# ── Stub ─────────────────────────────────────────────────────────────

class StubAnchorer:
    """Tutto ancorato tranne i sostantivi generici designati."""

    GENERIC = {"game", "teams", "players"}

    def is_anchored(self, entity: str, sentence: str) -> bool:
        return entity.lower().strip() not in self.GENERIC


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
