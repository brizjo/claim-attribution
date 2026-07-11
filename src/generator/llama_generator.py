"""
Modulo Generator — Generazione di risposte con Llama-3 tramite Ollama.

Supporta sia generazione standard che streaming con rilevamento di
stop-sequence per il pipeline In-Generation Attribution.

Il metodo generate_with_stop() è il cuore del sistema: genera token
fino a quando rileva il tag <CERCA:> nel testo, poi si ferma
e restituisce il testo parziale + la query di ricerca estratta.
"""

from typing import Optional

import requests
from typing import Optional, Callable

from config import settings


class LlamaGenerator:
    """
    Genera risposte utilizzando Llama-3 attraverso l'API locale di Ollama.

    Supporta:
      - generate():           generazione standard (batch)
      - generate_with_stop(): generazione streaming con stop su <CERCA:>
    """

    def __init__(
        self,
        model: str = settings.OLLAMA_MODEL,
        base_url: str = settings.OLLAMA_BASE_URL,
        temperature: float = settings.OLLAMA_TEMPERATURE,
        max_tokens: int = settings.OLLAMA_MAX_TOKENS,
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.temperature = temperature
        self.max_tokens = max_tokens

    # ────────────────────────────────────────────────────────────────
    # Generazione standard (non-streaming)
    # ────────────────────────────────────────────────────────────────

    def generate(self, prompt: str) -> str:
        """
        Invia il prompt a Ollama e restituisce la risposta generata.

        Args:
            prompt: Il prompt completo.

        Returns:
            Il testo generato da Llama-3.

        Raises:
            ConnectionError: Se Ollama non è raggiungibile.
            RuntimeError:    Se la risposta dell'API contiene un errore.
        """
        url = f"{self.base_url}/api/generate"
        payload = {
            "model": self.model,
            "prompt": prompt,
            "raw": True,
            "stream": False,
            "options": {
                "temperature": self.temperature,
                "num_predict": self.max_tokens,
            },
        }

        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = requests.post(url, json=payload, timeout=180)
                response.raise_for_status()
                break  # Success
            except requests.ConnectionError:
                raise ConnectionError(
                    f"Impossibile connettersi a Ollama su {self.base_url}. "
                    "Assicurati che Ollama sia avviato con: ollama serve"
                )
            except requests.HTTPError as e:
                if e.response is not None and e.response.status_code == 500 and attempt < max_retries - 1:
                    import time
                    wait = 2 ** (attempt + 1)
                    print(f"[LlamaGenerator] Ollama 500 error, retry {attempt+1}/{max_retries} in {wait}s...")
                    time.sleep(wait)
                else:
                    raise RuntimeError(f"Errore dall'API Ollama: {e}")

        data = response.json()
        return data.get("response", "")

    # ────────────────────────────────────────────────────────────────
    # Generazione streaming con stop-sequence detection
    # ────────────────────────────────────────────────────────────────

    def generate_with_stop(
        self,
        prompt: str,
        stop_tag: str = settings.CERCA_TAG,
        end_tag: str = settings.CERCA_END,
        stream_callback: Optional[Callable[[str], None]] = None,
    ) -> dict:
        """
        Genera in streaming, fermandosi quando rileva il tag <CERCA:>.

        Questo metodo è il cuore del pipeline In-Generation Attribution.
        Monitora il flusso di token dall'LLM e quando rileva la sequenza
        <CERCA: si ferma, estrae la query di ricerca, e restituisce
        il risultato parziale.

        Args:
            prompt:   Il prompt completo da inviare all'LLM.
            stop_tag: Il tag di stop da cercare (default: "<CERCA:").
            end_tag:  Il carattere di fine tag (default: ">").
            stream_callback: Funzione opzionale chiamata con ogni nuovo token (prima del tag di stop).

        Returns:
            Dict con chiavi:
              - "text":           Testo generato PRIMA del tag <CERCA:>
              - "cerca_query":    La query estratta (None se completato normalmente)
              - "stopped":        True se la generazione è stata fermata da <CERCA:>
              - "full_raw_text":  Testo grezzo completo (incluso il tag, per debug)
        """
        url = f"{self.base_url}/api/generate"
        payload = {
            "model": self.model,
            "prompt": prompt,
            "raw": True,
            "stream": True,
            "options": {
                "temperature": self.temperature,
                "num_predict": self.max_tokens,
            },
        }

        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = requests.post(url, json=payload, timeout=300, stream=True)
                response.raise_for_status()
                break  # Success
            except requests.ConnectionError:
                raise ConnectionError(
                    f"Impossibile connettersi a Ollama su {self.base_url}. "
                    "Assicurati che Ollama sia avviato con: ollama serve"
                )
            except requests.HTTPError as e:
                if e.response is not None and e.response.status_code == 500 and attempt < max_retries - 1:
                    import time
                    wait = 2 ** (attempt + 1)  # 2, 4, 8 seconds
                    print(f"[LlamaGenerator] Ollama 500 error, retry {attempt+1}/{max_retries} in {wait}s...")
                    time.sleep(wait)
                else:
                    raise

        full_text = ""
        cerca_query = None
        stopped = False

        for line in response.iter_lines():
            if not line:
                continue

            try:
                import json
                chunk = json.loads(line)
            except (json.JSONDecodeError, ValueError):
                continue

            token = chunk.get("response", "")
            full_text += token

            if token and stream_callback and not stopped:
                stream_callback(token)

            # ── Check per <CERCA: ...> nel buffer accumulato ──────
            if stop_tag in full_text and not stopped:
                tag_start = full_text.index(stop_tag)
                after_tag = full_text[tag_start + len(stop_tag):]

                if end_tag in after_tag:
                    # Tag completo: estraiamo la query
                    tag_end_pos = after_tag.index(end_tag)
                    cerca_query = after_tag[:tag_end_pos].strip()
                    stopped = True

                    # Il testo "pulito" è tutto ciò che precede il tag
                    clean_text = full_text[:tag_start].strip()

                    # Chiudi lo stream (non ci servono più token)
                    response.close()

                    return {
                        "text": clean_text,
                        "cerca_query": cerca_query,
                        "stopped": True,
                        "full_raw_text": full_text,
                    }

            # Se il modello ha terminato naturalmente
            if chunk.get("done", False):
                break

        # Generazione completata senza <CERCA:> — risposta finale
        # Pulizia: rimuovi eventuali <CERCA: parziali (senza >)
        clean_text = full_text
        if stop_tag in clean_text:
            tag_pos = clean_text.index(stop_tag)
            clean_text = clean_text[:tag_pos].strip()

        return {
            "text": clean_text,
            "cerca_query": None,
            "stopped": False,
            "full_raw_text": full_text,
        }

    # ────────────────────────────────────────────────────────────────
    # Health Check
    # ────────────────────────────────────────────────────────────────

    def is_available(self) -> bool:
        """Verifica se Ollama è raggiungibile e il modello è caricato."""
        try:
            resp = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if resp.status_code != 200:
                return False
            models = [m["name"] for m in resp.json().get("models", [])]
            # Controlla se il modello (o una sua variante) è disponibile
            return any(self.model in m for m in models)
        except Exception:
            return False
