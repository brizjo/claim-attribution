"""
La pipeline principale ha UN SOLO estrattore: DeepSeek (2026-09-03).

REBEL resta nel repo ma solo per gli esperimenti: questi test intercettano il
rientro silenzioso di REBEL nella pipeline (una factory che lo ricostruisce, un
`ACTIVE_EXTRACTOR` che torna a `rebel`, il claim parsato con un estrattore
diverso da quello che ha scritto il grafo — cioe' la simmetria di estrazione
rotta, requisito di `regole_progetto.md` §4).

Nessuna rete: si controllano tipi e configurazione, non si chiama l'API.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import settings  # noqa: E402
from src.ingestion.alce_ingestor import build_extractor  # noqa: E402
from src.ingestion.deepseek_extractor import DeepSeekExtractor  # noqa: E402


def test_deepseek_is_the_only_selectable_extractor():
    assert settings.AVAILABLE_EXTRACTORS == [settings.EXTRACTOR_DEEPSEEK]


def test_active_extractor_defaults_to_deepseek(monkeypatch):
    monkeypatch.delenv("ACTIVE_EXTRACTOR", raising=False)
    reloaded = importlib.reload(settings)
    try:
        assert reloaded.ACTIVE_EXTRACTOR == reloaded.EXTRACTOR_DEEPSEEK
    finally:
        importlib.reload(settings)


def test_build_extractor_returns_deepseek_by_default():
    assert isinstance(build_extractor(), DeepSeekExtractor)
    assert isinstance(build_extractor(settings.EXTRACTOR_DEEPSEEK), DeepSeekExtractor)


def test_build_extractor_refuses_rebel_with_a_pointer_to_the_experiments():
    with pytest.raises(ValueError) as exc:
        build_extractor(settings.EXTRACTOR_REBEL)
    assert "esperimenti" in str(exc.value).lower()


def test_build_extractor_still_rejects_unknown_names():
    with pytest.raises(ValueError):
        build_extractor("mrebel")


def test_claim_parsing_uses_the_same_extractor_that_wrote_the_graph():
    """Simmetria di estrazione: il claim si parsa con DeepSeek, non con REBEL."""
    from src.attribution.claim_attributor import ClaimAttributor

    attributor = ClaimAttributor(client=None)
    assert isinstance(attributor._parser, DeepSeekExtractor)
    assert attributor._parser.name == settings.EXTRACTOR_DEEPSEEK


def test_rebel_label_survives_for_old_graphs():
    """L'etichetta resta: gli archi scritti prima restano interrogabili."""
    assert settings.EXTRACTOR_REBEL == "rebel"
