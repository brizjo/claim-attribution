"""
Modulo Support Matrix — Core della Claim Attribution.

Questo modulo costruisce la matrice di supporto output-contesti:
  - Righe (Asse Y): fatti atomici estratti dalla risposta LLM
  - Colonne (Asse X): singole frasi dei documenti di contesto
  - Celle: score di supporto composito (BERTScore + Lexical Overlap)

La matrice è il cuore del sistema di attribuzione: ogni cella indica
quanto un dato fatto atomico è supportato da una specifica frase
del contesto recuperato.
"""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from config import settings
from src.attribution.semantic_similarity import SemanticSimilarity
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
        bertscore:          Score BERTScore del miglior match.
        lexical_score:      Score lessicale del miglior match.
    """
    fact: str
    support_score: float
    best_evidence: str = ""
    best_evidence_source: str = ""
    bertscore: float = 0.0
    lexical_score: float = 0.0


class SupportMatrix:
    """
    Calcola e gestisce la matrice di supporto fatti_atomici × frasi_contesto.

    Il punteggio di ogni cella è un composito pesato:
        score = w_bert * BERTScore_F1 + w_lex * ROUGE_L

    I pesi sono configurabili in settings.py.
    """

    def __init__(
        self,
        weight_bertscore: float = settings.WEIGHT_BERTSCORE,
        weight_lexical: float = settings.WEIGHT_LEXICAL_OVERLAP,
    ):
        self.weight_bertscore = weight_bertscore
        self.weight_lexical = weight_lexical
        self.semantic = SemanticSimilarity()
        self.lexical = LexicalOverlap()

        # Dati della matrice (popolati dopo compute())
        self.matrix: Optional[np.ndarray] = None
        self.bert_matrix: Optional[np.ndarray] = None
        self.lex_matrix: Optional[np.ndarray] = None
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
        Calcola la matrice di supporto composita.

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
            self.bert_matrix = self.matrix.copy()
            self.lex_matrix = self.matrix.copy()
            return self.matrix

        # Calcola le due matrici componenti
        self.bert_matrix = self.semantic.score_matrix(atomic_facts, sentences)
        self.lex_matrix = self.lexical.score_matrix(atomic_facts, sentences)

        # Matrice composita pesata
        self.matrix = (
            self.weight_bertscore * self.bert_matrix
            + self.weight_lexical * self.lex_matrix
        )

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
                    bertscore=float(self.bert_matrix[i, best_j]),
                    lexical_score=float(self.lex_matrix[i, best_j]),
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
                    "bertscore": r.bertscore,
                    "lexical_score": r.lexical_score,
                }
                for r in self.get_attribution_results()
            ],
            "overall_score": self.get_overall_score(),
        }
