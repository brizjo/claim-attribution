"""
Generatore grounded (studio <claim, context>, stadio 1) — nessuna rete.
"""

import pytest

from src.generation.answer_generator import (
    AnswerGenerator,
    GENERATOR_SYSTEM_PROMPT,
    build_generator_messages,
    format_passages,
)

PASSAGES = [
    {"source_id": "1", "title": "Josef Bican",
     "text": "Josef Bican was an Austrian-Czech footballer."},
    {"source_id": "2", "title": "Pelé",
     "text": "Pelé scored 1281 goals recognized by FIFA."},
]


class StubClient:
    model = "deepseek-chat"

    def __init__(self, response="Josef Bican scored the most goals."):
        self.response = response
        self.calls = []

    def chat(self, messages, json_mode=True, max_tokens=None):
        self.calls.append({"messages": messages, "json_mode": json_mode})
        return self.response

    def is_available(self):
        return True


def test_format_passages_integral_and_numbered():
    text = format_passages(PASSAGES)
    assert "[1] Josef Bican" in text
    assert "[2] Pelé" in text
    # Testo INTEGRALE, mai troncato.
    for p in PASSAGES:
        assert p["text"] in text


def test_messages_contain_question_passages_and_grounding_contract():
    msgs = build_generator_messages("Who scored the most goals?", PASSAGES)
    assert msgs[0]["role"] == "system"
    assert msgs[0]["content"] == GENERATOR_SYSTEM_PROMPT
    assert "ONLY" in GENERATOR_SYSTEM_PROMPT           # grounding esplicito
    assert "never pronouns" in GENERATOR_SYSTEM_PROMPT # triple-friendly
    assert "Who scored the most goals?" in msgs[1]["content"]
    assert PASSAGES[0]["text"] in msgs[1]["content"]


def test_generate_returns_answer_with_passages_and_no_json_mode():
    client = StubClient()
    gen = AnswerGenerator(client=client)
    ga = gen.generate("Who scored the most goals?", PASSAGES, sample_id="s1")

    assert ga.answer == "Josef Bican scored the most goals."
    assert ga.sample_id == "s1"
    assert ga.model == "deepseek-chat"
    assert ga.passages == PASSAGES
    # Risposta in prosa: la chiamata NON deve essere in JSON mode.
    assert client.calls[0]["json_mode"] is False


def test_generate_refuses_empty_inputs():
    gen = AnswerGenerator(client=StubClient())
    with pytest.raises(ValueError):
        gen.generate("   ", PASSAGES)
    with pytest.raises(ValueError):
        gen.generate("Who?", [])
