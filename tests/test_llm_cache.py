"""
Test della cache LLM su disco — riproducibilita' degli esperimenti.

L'API non e' deterministica nemmeno a temperature=0: senza cache due run dello
stesso esperimento davano conteggi diversi.  Qui si verifica che:
  * lo stesso prompt produca la stessa risposta senza richiamare l'API;
  * prompt diversi abbiano chiavi diverse;
  * due thread con lo STESSO prompt condividano UNA sola chiamata
    (single-flight) — la race che faceva divergere due run consecutivi.
"""

from __future__ import annotations

import sys
import threading
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.llm.deepseek_client import DeepSeekClient  # noqa: E402

MESSAGES = [
    {"role": "system", "content": "Reply with JSON."},
    {"role": "user", "content": "Return JSON {\"ok\": true}"},
]


class FakeResponse:
    status_code = 200

    def __init__(self, content: str):
        self._content = content

    def raise_for_status(self):
        pass

    def json(self):
        return {"choices": [{"message": {"content": self._content}}]}


@pytest.fixture
def client(tmp_path):
    return DeepSeekClient(api_key="test-key", cache_dir=str(tmp_path), use_cache=True)


def test_second_call_hits_cache(client, monkeypatch):
    calls = []

    def fake_post(url, json=None, headers=None, timeout=None):
        calls.append(url)
        return FakeResponse('{"n": %d}' % len(calls))

    monkeypatch.setattr("src.llm.deepseek_client.requests.post", fake_post)

    first = client.chat(MESSAGES)
    second = client.chat(MESSAGES)

    assert first == second == '{"n": 1}'
    assert len(calls) == 1
    stats = client.cache_stats()
    assert stats["hits"] == 1 and stats["misses"] == 1


def test_different_prompt_different_key(client):
    other = [MESSAGES[0], {"role": "user", "content": "different"}]
    assert client.cache_key(MESSAGES) != client.cache_key(other)


def test_key_includes_model_and_temperature(tmp_path):
    a = DeepSeekClient(api_key="k", cache_dir=str(tmp_path), model="m1")
    b = DeepSeekClient(api_key="k", cache_dir=str(tmp_path), model="m2")
    c = DeepSeekClient(api_key="k", cache_dir=str(tmp_path), model="m1", temperature=0.7)
    assert a.cache_key(MESSAGES) != b.cache_key(MESSAGES)
    assert a.cache_key(MESSAGES) != c.cache_key(MESSAGES)


def test_single_flight_shares_one_call(client, monkeypatch):
    """Due thread, stesso prompt: una sola chiamata e la stessa risposta."""
    calls = []

    def slow_post(url, json=None, headers=None, timeout=None):
        calls.append(url)
        time.sleep(0.3)                      # finestra in cui parte il secondo thread
        return FakeResponse('{"n": %d}' % len(calls))

    monkeypatch.setattr("src.llm.deepseek_client.requests.post", slow_post)

    results: list[str] = []

    def worker():
        results.append(client.chat(MESSAGES))

    threads = [threading.Thread(target=worker) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10)

    assert len(calls) == 1, "il prompt duplicato ha generato due chiamate API"
    assert results[0] == results[1] == '{"n": 1}'


def test_cache_disabled_always_calls(tmp_path, monkeypatch):
    calls = []

    def fake_post(url, json=None, headers=None, timeout=None):
        calls.append(url)
        return FakeResponse('{"n": %d}' % len(calls))

    monkeypatch.setattr("src.llm.deepseek_client.requests.post", fake_post)
    client = DeepSeekClient(api_key="k", cache_dir=str(tmp_path), use_cache=False)
    client.chat(MESSAGES)
    client.chat(MESSAGES)
    assert len(calls) == 2
