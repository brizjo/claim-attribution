"""
Test della pipeline ibrida — guardrail, span assegnato dalla pipeline,
contabilita' dei candidati REBEL.  Nessuna rete: REBEL e DeepSeek sono stub.

Copre le regressioni che hanno prodotto dati sbagliati nel primo giro:
  * `claim_span` chiesto al modello -> vuoto -> triple valide scartate;
  * candidati REBEL contati due volte (matched + rejected != prodotte);
  * un candidato "tenuto" dal validatore ma ucciso dai guardrail contato
    comunque fra le confermate.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ingestion import guardrails  # noqa: E402
from src.ingestion.hybrid_extractor import (  # noqa: E402
    MATCHED_STATUSES,
    ORIGIN_REBEL_CONFIRMED,
    ORIGIN_REBEL_VALIDATED,
    VARIANT_A,
    VARIANT_D,
    HybridExtractor,
    pair_key,
    parse_verdicts,
    rebel_vocabulary,
)
from src.ingestion.triple_extractor import Triple  # noqa: E402

ORIGINAL = ("Barack Obama was born in Honolulu in 1961. "
            "He served as president of the United States. "
            "He won the Nobel Peace Prize that same year.")
RESOLVED = ("Barack Obama was born in Honolulu in 1961. "
            "Barack Obama served as president of the United States. "
            "Barack Obama won the Nobel Peace Prize that same year.")
CHUNK = {"source_id": "doc-1", "title": "Barack Obama", "chunk_index": 0,
         "text": ORIGINAL, "source_file": "alce"}


class FakeSplitter:
    def split(self, text):
        return [s.strip() + "." for s in text.rstrip(".").split(". ") if s.strip()]


SENTENCES = FakeSplitter().split(RESOLVED)

REBEL_BY_SENTENCE = {
    0: [("Barack Obama", "place of birth", "Honolulu"),
        ("Barack Obama", "date of birth", "1961"),
        ("Honolulu", "located in", "Honolulu")],              # S == O
    1: [("Barack Obama", "position held", "president")],      # nodo generico
    2: [("Nobel Peace Prize", "winner", "that same year"),    # deittico
        ("Barack Obama", "award received", "Nobel Peace Prize"),
        ("Nobel Peace Prize", "conferred to", "Barack Obama")],  # solo REBEL
}

DEEPSEEK_BY_SENTENCE = {
    SENTENCES[0]: [("Barack Obama", "born in", "Honolulu"),
                   ("Barack Obama", "born in year", "1961")],
    SENTENCES[1]: [("Barack Obama", "served as", "president of the United States")],
    SENTENCES[2]: [("Barack Obama", "won", "Nobel Peace Prize"),
                   ("Barack Obama", "married to", "Michelle Obama")],  # non nel testo
}


class FakeRebel:
    def extract(self, chunks):
        out = []
        for chunk in chunks:
            idx = next((i for i, s in enumerate(SENTENCES) if s == chunk["text"]), None)
            for subject, predicate, obj in REBEL_BY_SENTENCE.get(idx, []):
                out.append(Triple(subject=subject, predicate=predicate, obj=obj,
                                  chunk_text=chunk["text"], source_file="alce",
                                  chunk_index=0, source_id="doc-1", extractor="rebel"))
        return out


class FakeClient:
    """Validatore che tiene solo il candidato "conferred to"."""

    def __init__(self):
        self.extract_calls = 0
        self.validate_calls = 0

    def chat(self, messages, json_mode=True, max_tokens=None):
        user = messages[1]["content"]
        if messages[0]["content"].startswith("You are a strict validator"):
            self.validate_calls += 1
            lines = [ln for ln in user.splitlines()
                     if ln.strip() and ln.strip()[0].isdigit()]
            return json.dumps({"verdicts": [
                {"index": i, "verdict": "keep" if "conferred to" in ln else "discard",
                 "reason": "test"}
                for i, ln in enumerate(lines)
            ]})
        self.extract_calls += 1
        sentence = next((s for s in SENTENCES if s in user), None)
        return json.dumps({"triples": [
            {"subject": s, "predicate": p, "object": o}
            for s, p, o in DEEPSEEK_BY_SENTENCE.get(sentence, [])
        ]})


def run(variant):
    client = FakeClient()
    extractor = HybridExtractor(rebel=FakeRebel(), deepseek_client=client,
                                splitter=FakeSplitter(), variant=variant,
                                vocabulary=["place of birth", "position held"])
    return extractor.run_passage(CHUNK, RESOLVED), client


@pytest.fixture(scope="module")
def result_a():
    return run(VARIANT_A)


@pytest.fixture(scope="module")
def result_d():
    return run(VARIANT_D)


# ── claim_span assegnato dalla pipeline ──────────────────────────────

@pytest.mark.parametrize("fixture_name", ["result_a", "result_d"])
def test_span_always_assigned_and_verbatim(fixture_name, request):
    passage, _ = request.getfixturevalue(fixture_name)
    assert passage.error == ""
    assert passage.survived, "nessuna tripla sopravvissuta"
    for triple in passage.survived:
        assert triple.claim_span, "claim_span vuoto: lo assegna la pipeline"
        assert triple.claim_span in ORIGINAL, "claim_span non verbatim sull'originale"
        assert triple.sentence in RESOLVED


@pytest.mark.parametrize("fixture_name", ["result_a", "result_d"])
def test_no_span_reason_is_gone(fixture_name, request):
    passage, _ = request.getfixturevalue(fixture_name)
    assert "no_span" not in {t.reason for t in passage.discarded}


# ── Contabilita' dei candidati REBEL ─────────────────────────────────

@pytest.mark.parametrize("fixture_name", ["result_a", "result_d"])
def test_rebel_accounting_invariant(fixture_name, request):
    passage, _ = request.getfixturevalue(fixture_name)
    assert passage.rebel_matched + len(passage.rebel_rejected) == \
        len(passage.rebel_candidates)
    assert all(c.status for c in passage.rebel_candidates)


@pytest.mark.parametrize("fixture_name", ["result_a", "result_d"])
def test_rebel_kept_never_exceeds_matched(fixture_name, request):
    passage, _ = request.getfixturevalue(fixture_name)
    assert passage.rebel_kept <= passage.rebel_matched


def test_validated_candidate_killed_by_guardrail_is_not_counted_as_matched():
    """Un "keep" del validatore non basta: se i guardrail uccidono la tripla,
    il candidato e' rigettato, non confermato."""
    class KeepEverything(FakeClient):
        def chat(self, messages, json_mode=True, max_tokens=None):
            if messages[0]["content"].startswith("You are a strict validator"):
                self.validate_calls += 1
                lines = [ln for ln in messages[1]["content"].splitlines()
                         if ln.strip() and ln.strip()[0].isdigit()]
                return json.dumps({"verdicts": [
                    {"index": i, "verdict": "keep", "reason": "test"}
                    for i in range(len(lines))]})
            return super().chat(messages, json_mode, max_tokens)

    extractor = HybridExtractor(rebel=FakeRebel(), deepseek_client=KeepEverything(),
                                splitter=FakeSplitter(), variant=VARIANT_D)
    passage = extractor.run_passage(CHUNK, RESOLVED)
    statuses = {(c.subject, c.predicate, c.obj): c.status
                for c in passage.rebel_candidates}
    assert statuses[("Honolulu", "located in", "Honolulu")] not in MATCHED_STATUSES
    assert passage.rebel_matched + len(passage.rebel_rejected) == \
        len(passage.rebel_candidates)


# ── Costo e provenienza ──────────────────────────────────────────────

def test_call_budget(result_a, result_d):
    passage_a, client_a = result_a
    passage_d, client_d = result_d
    assert client_a.validate_calls == 0
    assert passage_a.llm_calls == len(passage_a.units)
    assert client_d.validate_calls == 1
    assert passage_d.llm_calls == len(passage_d.units) + 1


def test_origin_taxonomy(result_a, result_d):
    passage_a, _ = result_a
    passage_d, _ = result_d
    assert any(t.origin == ORIGIN_REBEL_CONFIRMED for t in passage_a.survived)
    assert all(t.origin != ORIGIN_REBEL_VALIDATED for t in passage_a.survived), \
        "la variante A non valida candidati REBEL"
    assert any(t.origin == ORIGIN_REBEL_VALIDATED for t in passage_d.survived)


def test_vocabulary_from_rebel_output(result_a):
    passage, _ = result_a
    vocab = rebel_vocabulary(passage.units)
    assert "place of birth" in vocab and "winner" in vocab
    assert len(vocab) == len({t["predicate"] for u in passage.units for t in u.rebel})


# ── Guardrail ────────────────────────────────────────────────────────

SENTENCE = "Pele scored 1281 goals for Santos in that same year."


@pytest.mark.parametrize("subject,predicate,obj,reason", [
    ("Pele", "scored", "1281 goals", ""),
    ("Pele", "played for", "Santos", ""),
    ("Pele", "", "Santos", "no_predicate"),
    ("that same year", "involved", "Pele", "unresolved_reference"),
    ("he", "played for", "Santos", "unresolved_reference"),
    ("Pele", "played for", "Pele", "subject_equals_object"),
    ("Pele", "married to", "Michelle Obama", "entity_not_in_sentence"),
])
def test_guardrail_reasons(subject, predicate, obj, reason):
    verdict = guardrails.check(subject, predicate, obj, sentence=SENTENCE)
    assert verdict.reason == reason
    assert verdict.ok == (reason == "")


def test_generic_node_rejected():
    sentence = "The game between the two teams was played in 1999."
    verdict = guardrails.check("game", "played by", "teams", sentence=sentence)
    assert verdict.reason == "generic_node"


def test_pair_key_ignores_predicate_and_case():
    assert pair_key("Barack Obama", "Honolulu") == pair_key("barack obama", "honolulu")


def test_parse_verdicts_recovers_broken_json():
    broken = ('{"verdicts": [{"index": 0, "verdict": "keep", "reason": "a"}, '
              '{"index": 1, "verdict": "discard", "reason": "b"}}')
    verdicts = parse_verdicts(broken)
    assert len(verdicts) == 2 and verdicts[0]["verdict"] == "keep"
