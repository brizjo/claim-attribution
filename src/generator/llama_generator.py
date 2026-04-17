"""
Modulo Generator — Generazione di risposte con Llama-3 tramite Ollama.

Questo modulo si occupa di:
1. Comporre il prompt RAG (contesto + domanda)
2. Inviare il prompt a Llama-3 tramite l'API di Ollama
3. Restituire la risposta generata

Il sistema segue un approccio Post-Retrieval: i documenti recuperati
vengono inseriti nel prompt prima della generazione.
"""

from typing import Optional

import requests

from config import settings


class LlamaGenerator:
    """
    Genera risposte utilizzando Llama-3 attraverso l'API locale di Ollama.

    Attributes:
        model:       Nome del modello Ollama (default: llama3).
        base_url:    URL base dell'API Ollama.
        temperature: Temperatura di campionamento.
        max_tokens:  Numero massimo di token in output.
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
    # Prompt Building
    # ────────────────────────────────────────────────────────────────

    @staticmethod
    def build_prompt(
        question: str,
        context_documents: list[dict],
        template: str = settings.RAG_PROMPT_TEMPLATE,
    ) -> str:
        """
        Costruisce il prompt RAG inserendo i documenti di contesto.

        Args:
            question:          La domanda dell'utente.
            context_documents: Lista di dict con chiave "document" (e opzionalmente "metadata").
            template:          Template del prompt con placeholder {context} e {question}.

        Returns:
            Il prompt completo pronto per l'LLM.
        """
        context_parts = []
        for i, doc in enumerate(context_documents, start=1):
            source = doc.get("metadata", {}).get("source", "Unknown")
            text = doc.get("document", "")
            context_parts.append(f"[Document {i} — Source: {source}]\n{text}")

        context_str = "\n\n".join(context_parts)
        return template.format(context=context_str, question=question)

    # ────────────────────────────────────────────────────────────────
    # Generation via Ollama API
    # ────────────────────────────────────────────────────────────────

    def generate(self, prompt: str) -> str:
        """
        Invia il prompt a Ollama e restituisce la risposta generata.

        Args:
            prompt: Il prompt completo (contesto + domanda).

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
            "stream": False,
            "options": {
                "temperature": self.temperature,
                "num_predict": self.max_tokens,
            },
        }

        try:
            response = requests.post(url, json=payload, timeout=120)
            response.raise_for_status()
        except requests.ConnectionError:
            raise ConnectionError(
                f"Impossibile connettersi a Ollama su {self.base_url}. "
                "Assicurati che Ollama sia avviato con: ollama serve"
            )
        except requests.HTTPError as e:
            raise RuntimeError(f"Errore dall'API Ollama: {e}")

        data = response.json()
        return data.get("response", "")

    # ────────────────────────────────────────────────────────────────
    # Pipeline completa: prompt building + generation
    # ────────────────────────────────────────────────────────────────

    def run(
        self,
        question: str,
        context_documents: list[dict],
    ) -> dict[str, str]:
        """
        Pipeline completa: costruisce il prompt e genera la risposta.

        Args:
            question:          La domanda dell'utente.
            context_documents: Documenti di contesto dal retriever.

        Returns:
            Dict con chiavi:
              - "prompt":   il prompt inviato all'LLM
              - "response": la risposta generata
        """
        prompt = self.build_prompt(question, context_documents)
        response = self.generate(prompt)
        return {"prompt": prompt, "response": response}

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
