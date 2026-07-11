"""
Modulo Support Matrix — Audit Matematico Deterministico.

Implementa il framework di valutazione FActScore-like usando esclusivamente
metriche lessicali deterministiche:
  - Exact Match (contenenza testuale)
  - ROUGE-L (Longest Common Subsequence)

NON usa modelli generativi NLI per evitare Attestation Bias.
Ogni claim atomica viene valutata contro i chunk di contesto recuperati
durante la specifica iterazione CERCA che l'ha generata.
"""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from config import settings
from src.attribution.lexical_overlap import LexicalOverlap


@dataclass
class AttributionResult:
    """
    Risultato dell'attribuzione per un singolo fatto atomico.

    Attributes:
        fact:               Il testo del fatto atomico.
        support_score:      Score di supporto aggregato (0-1).
        best_evidence:      La frase del contesto con il match migliore.
        best_evidence_source: Fonte della frase con il match migliore.
        source_chunk_text:  Il testo esatto del chunk sorgente recuperato.
        source_id:          Identificatore univoco della fonte.
        rouge_l:            Score ROUGE-L del miglior match.
        exact_match:        Score Exact Match del miglior match.
    """
    fact: str
    support_score: float
    best_evidence: str = ""
    best_evidence_source: str = ""
    source_chunk_text: str = ""
    source_id: str = ""
    rouge_l: float = 0.0
    exact_match: float = 0.0


class SupportMatrix:
    """
    Calcola la matrice di supporto usando solo metriche lessicali.

    Lo score di ogni cella è il massimo tra Exact Match e ROUGE-L:
        score = max(exact_match, rouge_l)

    Questo approccio è completamente deterministico e non soffre
    di Attestation Bias (a differenza di modelli NLI generativi).
    """

    def __init__(self):
        self.lexical = LexicalOverlap()

        # Dati della matrice (popolati dopo compute())
        self.matrix: Optional[np.ndarray] = None
        self.rouge_matrix: Optional[np.ndarray] = None
        self.em_matrix: Optional[np.ndarray] = None
        self.atomic_facts: list[str] = []
        self.evidence_sentences: list[dict] = []

    # ────────────────────────────────────────────────────────────────
    # Calcolo della matrice
    # ────────────────────────────────────────────────────────────────

    def compute(
        self,
        atomic_facts: list[str],
        evidence_sentences: list[dict[str, str]],
    ) -> np.ndarray:
        """
        Calcola la matrice di supporto lessicale M×N.

        Args:
            atomic_facts:       Lista di M fatti atomici (dalla risposta LLM).
            evidence_sentences: Lista di N dict con chiave "sentence" (e "source").

        Returns:
            np.ndarray di shape (M, N) con gli score compositi.
        """
        self.atomic_facts = atomic_facts
        self.evidence_sentences = evidence_sentences

        sentences = [ev["sentence"] for ev in evidence_sentences]

        if not atomic_facts or not sentences:
            self.matrix = np.zeros((len(atomic_facts), len(sentences)))
            self.rouge_matrix = self.matrix.copy()
            self.em_matrix = self.matrix.copy()
            return self.matrix

        m = len(atomic_facts)
        n = len(sentences)

        # Calcola ROUGE-L matrix
        self.rouge_matrix = self.lexical.score_matrix(atomic_facts, sentences)

        # Calcola Exact Match matrix
        self.em_matrix = np.zeros((m, n))
        for i, fact in enumerate(atomic_facts):
            for j, ref in enumerate(sentences):
                self.em_matrix[i, j] = self.lexical.exact_match(fact, ref)

        # Matrice composita: max(exact_match, rouge_l) per ogni cella
        self.matrix = np.maximum(self.rouge_matrix, self.em_matrix)

        return self.matrix

    # ────────────────────────────────────────────────────────────────
    # Risultati di attribuzione
    # ────────────────────────────────────────────────────────────────

    def get_attribution_results(self) -> list[AttributionResult]:
        """
        Genera i risultati di attribuzione per ogni fatto atomico.

        Per ogni fatto, seleziona la frase del contesto con il punteggio
        composito più alto e restituisce un `AttributionResult`.

        Returns:
            Lista di AttributionResult, uno per fatto atomico.
        """
        if self.matrix is None:
            raise ValueError(
                "La matrice non è stata ancora calcolata. Chiama compute() prima."
            )

        results = []
        for i, fact in enumerate(self.atomic_facts):
            row = self.matrix[i]

            if len(row) == 0:
                results.append(AttributionResult(fact=fact, support_score=0.0))
                continue

            best_j = int(np.argmax(row))
            best_score = float(row[best_j])
            best_evidence = self.evidence_sentences[best_j]

            results.append(
                AttributionResult(
                    fact=fact,
                    support_score=best_score,
                    best_evidence=best_evidence["sentence"],
                    best_evidence_source=best_evidence.get("source", "Unknown"),
                    source_chunk_text=best_evidence.get("chunk_text", best_evidence["sentence"]),
                    source_id=best_evidence.get("source", "Unknown"),
                    rouge_l=float(self.rouge_matrix[i, best_j]),
                    exact_match=float(self.em_matrix[i, best_j]),
                )
            )

        return results

    # ────────────────────────────────────────────────────────────────
    # Utilities
    # ────────────────────────────────────────────────────────────────

    def get_fact_score(self, fact_index: int) -> float:
        """Restituisce lo score di supporto massimo per un fatto atomico."""
        if self.matrix is None or fact_index >= len(self.matrix):
            return 0.0
        return float(np.max(self.matrix[fact_index]))

    def get_overall_score(self) -> float:
        """Restituisce lo score medio di supporto su tutti i fatti atomici."""
        if self.matrix is None or self.matrix.size == 0:
            return 0.0
        # Media dei massimi per riga (= supporto medio per fatto)
        row_maxes = np.max(self.matrix, axis=1)
        return float(np.mean(row_maxes))

    def to_dict(self) -> dict:
        """Serializza la matrice e i risultati in un dizionario."""
        return {
            "atomic_facts": self.atomic_facts,
            "evidence_sentences": [
                ev["sentence"] for ev in self.evidence_sentences
            ],
            "matrix": self.matrix.tolist() if self.matrix is not None else [],
            "attribution_results": [
                {
                    "fact": r.fact,
                    "support_score": r.support_score,
                    "best_evidence": r.best_evidence,
                    "best_evidence_source": r.best_evidence_source,
                    "rouge_l": r.rouge_l,
                    "exact_match": r.exact_match,
                }
                for r in self.get_attribution_results()
            ],
            "overall_score": self.get_overall_score(),
        }
