"""
L'app deve partire in ENTRAMBE le modalita': con e senza il tab esperimenti.

Il tab e' uno strumento di sviluppo, quindi e' nascosto per default e compare
solo con `SHOW_EXPERIMENTS=1`.  La regressione che questi test intercettano e'
banale ma costosa: uno spostamento di codice in `src/ui/` che lascia l'app
rotta in una delle due configurazioni.

Nessuna rete e nessun modello: l'app carica i pesi solo dentro i callback dei
bottoni, che qui non vengono premuti.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

APP = Path(__file__).resolve().parent.parent / "app.py"
EXPERIMENT_MARKER = "Esperimento ibrido"


def _run(monkeypatch, show_experiments: str):
    from streamlit.testing.v1 import AppTest

    monkeypatch.setenv("SHOW_EXPERIMENTS", show_experiments)
    # `settings` legge l'ambiente all'import: va ricaricato prima di far
    # partire l'app, altrimenti il test misura il valore di un altro run.
    import config.settings as settings
    importlib.reload(settings)

    at = AppTest.from_file(str(APP), default_timeout=120)
    at.run()
    return at


def _texts(at) -> str:
    return "\n".join(str(m.value) for m in at.markdown)


@pytest.mark.parametrize("flag", ["0", ""])
def test_app_starts_without_experiments(monkeypatch, flag):
    at = _run(monkeypatch, flag)
    assert not at.exception, at.exception
    assert EXPERIMENT_MARKER not in _texts(at)


@pytest.mark.parametrize("flag", ["1", "true"])
def test_app_starts_with_experiments(monkeypatch, flag):
    at = _run(monkeypatch, flag)
    assert not at.exception, at.exception
    assert EXPERIMENT_MARKER in _texts(at)


def test_experiments_module_is_importable_on_its_own():
    """`src/ui/experiments.py` non deve dipendere da simboli di `app.py`."""
    module = importlib.import_module("src.ui.experiments")
    assert callable(module.render)
