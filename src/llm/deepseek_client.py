"""
Client DeepSeek — unico LLM attivo del progetto (Ollama è in legacy/).

API OpenAI-compatible: POST {base_url}/chat/completions.
Si usa `requests` (già dipendenza) invece del SDK openai: una sola
chiamata HTTP, nessun motivo per aggiungere un pacchetto.

API KEY — NON hardcodarla. Letta da config/settings.py:
    1. `.env` nella root del progetto  →  DEEPSEEK_API_KEY=sk-...
    2. variabile d'ambiente di sistema DEEPSEEK_API_KEY
`.env` è già in .gitignore. Template: `.env.example`.

Usato da:
  * src/ingestion/deepseek_extractor.py  — estrazione triple
  * src/attribution/question_parser.py   — domanda → tripla parziale
"""

from __future__ import annotations

import logging
import time
from typing import Optional

import requests

from config import settings

logger = logging.getLogger(__name__)


class DeepSeekClient:
    """Chat completions con retry su rate-limit/5xx."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = settings.DEEPSEEK_MODEL,
        base_url: str = settings.DEEPSEEK_BASE_URL,
        temperature: float = settings.DEEPSEEK_TEMPERATURE,
        max_tokens: int = settings.DEEPSEEK_MAX_TOKENS,
        timeout: int = settings.DEEPSEEK_TIMEOUT,
        max_retries: int = settings.DEEPSEEK_MAX_RETRIES,
    ):
        self.api_key = api_key if api_key is not None else settings.DEEPSEEK_API_KEY
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.temperature = temperature      # 0.0 ovunque: mai output creativo
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.max_retries = max_retries

    # ── Health check ────────────────────────────────────────────────

    def is_available(self) -> bool:
        """True se la API key è configurata (nessuna chiamata di rete)."""
        return bool(self.api_key)

    def check_connection(self) -> tuple[bool, str]:
        """Ping reale su /models — per il badge in UI."""
        if not self.api_key:
            return False, "DEEPSEEK_API_KEY non configurata (vedi .env)"
        try:
            resp = requests.get(
                f"{self.base_url}/models",
                headers={"Authorization": f"Bearer {self.api_key}"},
                timeout=10,
            )
            if resp.status_code == 401:
                return False, "API key rifiutata (401)"
            resp.raise_for_status()
            return True, self.model
        except Exception as exc:
            return False, f"{type(exc).__name__}: {exc}"

    # ── Chat ────────────────────────────────────────────────────────

    def chat(
        self,
        messages: list[dict],
        json_mode: bool = True,
        max_tokens: Optional[int] = None,
    ) -> str:
        """
        Ritorna il contenuto del primo choice.

        `json_mode=True` attiva `response_format: json_object` — richiede che
        la parola "JSON" compaia nel prompt (vincolo dell'API DeepSeek).
        """
        if not self.api_key:
            raise RuntimeError(
                "DEEPSEEK_API_KEY mancante. Crea un file `.env` nella root "
                "del progetto con:  DEEPSEEK_API_KEY=sk-..."
            )

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": max_tokens or self.max_tokens,
            "stream": False,
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        last_exc: Optional[Exception] = None
        for attempt in range(self.max_retries):
            try:
                resp = requests.post(
                    f"{self.base_url}/chat/completions",
                    json=payload,
                    headers=headers,
                    timeout=self.timeout,
                )
                if resp.status_code in (429, 500, 502, 503) and attempt < self.max_retries - 1:
                    wait = 2 ** (attempt + 1)
                    logger.warning(
                        "DeepSeek HTTP %s — retry %s/%s tra %ss",
                        resp.status_code, attempt + 1, self.max_retries, wait,
                    )
                    time.sleep(wait)
                    continue
                resp.raise_for_status()
                return resp.json()["choices"][0]["message"]["content"]
            except requests.RequestException as exc:
                last_exc = exc
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** (attempt + 1))
                    continue
                raise RuntimeError(f"DeepSeek API irraggiungibile: {exc}") from exc
            except (KeyError, IndexError, ValueError) as exc:
                raise RuntimeError(f"Risposta DeepSeek malformata: {exc}") from exc

        raise RuntimeError(f"DeepSeek: tutti i retry falliti ({last_exc})")
