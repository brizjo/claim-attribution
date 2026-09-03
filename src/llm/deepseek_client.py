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

import hashlib
import json
import logging
import threading
import time
from pathlib import Path
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
        cache_dir: Optional[str] = None,
        use_cache: Optional[bool] = None,
    ):
        self.api_key = api_key if api_key is not None else settings.DEEPSEEK_API_KEY
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.temperature = temperature      # 0.0 ovunque: mai output creativo
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.max_retries = max_retries
        # Cache su disco indicizzata per hash del prompt: stesso prompt =>
        # stessa risposta, cosi' un esperimento e' ri-eseguibile identico.
        self.use_cache = settings.LLM_CACHE_ENABLED if use_cache is None else use_cache
        self.cache_dir = Path(cache_dir or settings.LLM_CACHE_DIR)
        self.cache_hits = 0
        self.cache_misses = 0
        self._lock = threading.Lock()
        # Single-flight: due thread con lo STESSO prompt devono condividere una
        # sola chiamata. Senza, entrambi mancano la cache, chiamano l'API e
        # ottengono risposte diverse (l'API non e' deterministica nemmeno a
        # temperature=0) — due run dello stesso esperimento divergevano di
        # qualche tripla proprio per questo.
        self._inflight: dict[str, threading.Event] = {}

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

    # ── Cache ───────────────────────────────────────────────────────

    def cache_key(self, messages: list[dict], max_tokens: Optional[int] = None) -> str:
        """Hash del prompt COMPLETO (modello, temperatura, tetto token, messaggi)."""
        payload = json.dumps({
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": max_tokens or self.max_tokens,
            "messages": messages,
        }, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _cache_path(self, key: str) -> Path:
        return self.cache_dir / key[:2] / f"{key}.json"

    def _cache_read(self, key: str) -> Optional[str]:
        path = self._cache_path(key)
        if not path.is_file():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))["content"]
        except (OSError, ValueError, KeyError) as exc:
            logger.warning("cache LLM illeggibile %s (%s) — si richiama l'API", key[:8], exc)
            return None

    def _cache_write(self, key: str, content: str, messages: list[dict]) -> None:
        path = self._cache_path(key)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({
                "key": key,
                "model": self.model,
                "temperature": self.temperature,
                "system": messages[0]["content"][:120] if messages else "",
                "content": content,
            }, ensure_ascii=False), encoding="utf-8")
        except OSError as exc:
            logger.warning("scrittura cache LLM fallita (%s)", exc)

    def _claim_key(self, key: str) -> Optional[threading.Event]:
        """
        `None` se questo thread deve fare la chiamata; altrimenti l'Event da
        attendere perche' un altro thread la sta gia' facendo.
        """
        with self._lock:
            event = self._inflight.get(key)
            if event is None:
                self._inflight[key] = threading.Event()
                return None
        return event

    def _release_key(self, key: str) -> None:
        with self._lock:
            event = self._inflight.pop(key, None)
        if event is not None:
            event.set()

    def cache_stats(self) -> dict:
        total = self.cache_hits + self.cache_misses
        return {
            "hits": self.cache_hits,
            "misses": self.cache_misses,
            "hit_rate": (self.cache_hits / total) if total else 0.0,
            "enabled": self.use_cache,
        }

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
        key = self.cache_key(messages, max_tokens)
        if self.use_cache:
            cached = self._cache_read(key)
            if cached is not None:
                with self._lock:
                    self.cache_hits += 1
                logger.info("LLM cache HIT %s", key[:8])
                return cached
            waiting = self._claim_key(key)
            if waiting is not None:
                # Stesso prompt gia' in volo: si attende quella risposta.
                waiting.wait(timeout=self.timeout * self.max_retries)
                cached = self._cache_read(key)
                if cached is not None:
                    with self._lock:
                        self.cache_hits += 1
                    logger.info("LLM cache HIT (single-flight) %s", key[:8])
                    return cached
            with self._lock:
                self.cache_misses += 1
            logger.info("LLM cache MISS %s", key[:8])

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

        try:
            return self._request(payload, headers, key, messages)
        finally:
            if self.use_cache:
                self._release_key(key)

    def _request(self, payload: dict, headers: dict, key: str,
                 messages: list[dict]) -> str:
        """Chiamata HTTP con retry; scrive in cache la risposta ottenuta."""
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
                content = resp.json()["choices"][0]["message"]["content"]
                if self.use_cache:
                    self._cache_write(key, content, messages)
                return content
            except requests.RequestException as exc:
                last_exc = exc
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** (attempt + 1))
                    continue
                raise RuntimeError(f"DeepSeek API irraggiungibile: {exc}") from exc
            except (KeyError, IndexError, ValueError) as exc:
                raise RuntimeError(f"Risposta DeepSeek malformata: {exc}") from exc

        raise RuntimeError(f"DeepSeek: tutti i retry falliti ({last_exc})")
