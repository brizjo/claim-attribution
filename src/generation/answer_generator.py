"""
Answer generator — DeepSeek risponde alla domanda ALCE usando SOLO i passaggi.

Primo stadio dello studio della coppia <claim, context>
(`regole_progetto.md`, Oggetto di studio A):

    1. il GENERATORE risponde dai passaggi ground-truth   <- QUESTO MODULO
    2. dalla risposta si estraggono i claim (DeepSeek, simmetria §4)
    3. i claim si verificano nel grafo (exact match -> fallback semantico)
    4. la risposta finale si rigenera dai soli claim supportati

Gli stadi 2-4 sono FUTURI: oggi esiste solo la generazione grounded, esposta
nel tab Claim Attribution della UI.

Vincolo di grounding: il modello deve attenersi ai passaggi forniti — niente
memoria di addestramento, niente fonti esterne, e deve dirlo esplicitamente
quando i passaggi non bastano.  Il prompt chiede inoltre nomi completi al
posto dei pronomi: la risposta verra' scomposta in triple dalla STESSA
pipeline dell'ingestione, e un pronome diventerebbe `unresolved_reference`.

Trasporto HTTP, temperature=0 e cache: `src/llm/deepseek_client.py`.
Qui vivono solo il prompt e il tipo di ritorno (come per l'estrattore).
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

from src.llm.deepseek_client import DeepSeekClient

logger = logging.getLogger(__name__)


# ────────────────────────────────────────────────────────────────────
# Prompt
# ────────────────────────────────────────────────────────────────────

GENERATOR_SYSTEM_PROMPT = """You are a question-answering engine operating \
under a strict grounding contract.

You receive a question and a set of numbered passages. The passages are the \
ONLY source of truth.

Rules:
1. Use ONLY facts stated in the passages. Never use your training memory, \
world knowledge or any external source — not even to fill obvious gaps.
2. The question may be ambiguous: if the passages support multiple \
interpretations, cover every interpretation the passages support.
3. If the passages do not contain enough information to answer, or only \
answer part of the question, say so explicitly instead of guessing.
4. Always write complete entity names, never pronouns ("Barack Obama", not \
"he"): the answer will later be decomposed into atomic factual claims.
5. Write plain English prose — no lists, no markdown, no citations, no \
commentary about the passages or about these rules.
6. Be complete but concise: every sentence must state at least one fact \
verifiable in the passages."""

GENERATOR_USER_TEMPLATE = """Question: {question}

Passages:
{passages}

Answer the question using only these passages."""


def format_passages(passages: list[dict]) -> str:
    """Passaggi INTEGRALI, numerati, con titolo — nessun troncamento."""
    blocks = []
    for i, p in enumerate(passages, 1):
        title = p.get("title", p.get("source_file", "")) or "(untitled)"
        blocks.append(f"[{i}] {title}\n{p.get('text', '')}")
    return "\n\n".join(blocks)


def build_generator_messages(question: str, passages: list[dict]) -> list[dict]:
    """Messaggi chat — esposti per test/debug del prompt."""
    return [
        {"role": "system", "content": GENERATOR_SYSTEM_PROMPT},
        {"role": "user", "content": GENERATOR_USER_TEMPLATE.format(
            question=question, passages=format_passages(passages))},
    ]


# ────────────────────────────────────────────────────────────────────
# Generatore
# ────────────────────────────────────────────────────────────────────

@dataclass
class GeneratedAnswer:
    question: str
    answer: str
    model: str
    seconds: float
    sample_id: str = ""
    # I passaggi visti dal modello (integrali): la UI li mostra accanto
    # alla risposta, e in futuro saranno il contesto embeddato nel grafo.
    passages: list[dict] = field(default_factory=list)


class AnswerGenerator:
    """Genera la risposta grounded via DeepSeek chat API (testo, NO json)."""

    name = "deepseek-generator"

    def __init__(self, client: DeepSeekClient | None = None, **client_kwargs):
        self._client = client or DeepSeekClient(**client_kwargs)

    @property
    def model(self) -> str:
        return self._client.model

    @property
    def client(self) -> DeepSeekClient:
        return self._client

    # ── Health check (delegato al client) ───────────────────────────

    def is_available(self) -> bool:
        return self._client.is_available()

    def check_connection(self) -> tuple[bool, str]:
        return self._client.check_connection()

    # ── API pubblica ────────────────────────────────────────────────

    def generate(
        self,
        question: str,
        passages: list[dict],
        sample_id: str = "",
    ) -> GeneratedAnswer:
        """
        Una chiamata API per domanda: tutti i passaggi in un solo prompt
        (5 passaggi ALCE ~ 500 parole: ampiamente nel contesto del modello).

        Solleva l'eccezione del client se la chiamata fallisce: una risposta
        non generata NON deve travestirsi da risposta vuota.
        """
        if not question.strip():
            raise ValueError("Domanda vuota: niente da generare")
        if not passages:
            raise ValueError(
                "Nessun passaggio fornito: il generatore e' grounded-only, "
                "senza contesto non deve rispondere"
            )

        t0 = time.time()
        answer = self._client.chat(
            build_generator_messages(question, passages),
            json_mode=False,
        ).strip()
        seconds = time.time() - t0

        logger.info(
            "Generator: %d char di risposta da %d passaggi (%.1fs)",
            len(answer), len(passages), seconds,
        )
        return GeneratedAnswer(
            question=question,
            answer=answer,
            model=self.model,
            seconds=seconds,
            sample_id=sample_id,
            passages=list(passages),
        )
