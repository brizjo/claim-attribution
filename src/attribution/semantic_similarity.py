"""
Modulo Semantic Similarity — Calcolo della similarità semantica con BERTScore.

Utilizzato per popolare la matrice di supporto: misura quanto una claim
atomica (dalla risposta) è semanticamente simile a una frase del contesto.
"""

from typing import Optional

import numpy as np

from config import settings


class SemanticSimilarity:
    """
    Calcola la similarità semantica tra coppie di testi usando BERTScore.

    BERTScore sfrutta embeddings contestuali (es. DeBERTa) per calcolare
    precision, recall e F1 a livello di token tra due testi.
    """

    def __init__(
        self,
        model_type: str = settings.BERTSCORE_MODEL,
        lang: str = settings.BERTSCORE_LANG,
    ):
        self.model_type = model_type
        self.lang = lang
        self._scorer = None  # Lazy loading per evitare tempi di startup

    @property
    def scorer(self):
        """Lazy-load del modello BERTScore."""
        if self._scorer is None:
            from bert_score import BERTScorer
            self._scorer = BERTScorer(
                model_type=self.model_type,
                lang=self.lang,
                rescale_with_baseline=True,
            )
        return self._scorer

    # ────────────────────────────────────────────────────────────────
    # Singola coppia
    # ────────────────────────────────────────────────────────────────

    def score_pair(self, candidate: str, reference: str) -> dict[str, float]:
        """
        Calcola BERTScore tra un candidato e un riferimento.

        Args:
            candidate: Testo candidato (es. fatto atomico dalla risposta).
            reference: Testo di riferimento (es. frase dal contesto).

        Returns:
            Dict con "precision", "recall", "f1".
        """
        P, R, F1 = self.scorer.score(
            cands=[candidate],
            refs=[reference],
        )
        return {
            "precision": P.item(),
            "recall": R.item(),
            "f1": F1.item(),
        }

    # ────────────────────────────────────────────────────────────────
    # Batch: una claim contro N frasi di contesto
    # ────────────────────────────────────────────────────────────────

    def score_one_vs_many(
        self, candidate: str, references: list[str]
    ) -> list[float]:
        """
        Calcola l'F1 BERTScore di un candidato rispetto a molte referenze.

        Args:
            candidate:  Un fatto atomico.
            references: Lista di frasi di contesto.

        Returns:
            Lista di score F1 (uno per ogni referenza).
        """
        if not references:
            return []

        cands = [candidate] * len(references)
        _, _, F1 = self.scorer.score(cands=cands, refs=references)
        return F1.tolist()

    # ────────────────────────────────────────────────────────────────
    # Batch completo: M claims × N frasi
    # ────────────────────────────────────────────────────────────────

    def score_matrix(
        self, candidates: list[str], references: list[str]
    ) -> np.ndarray:
        """
        Calcola la matrice completa M×N di BERTScore F1.

        Args:
            candidates: Lista di M fatti atomici.
            references: Lista di N frasi di contesto.

        Returns:
            np.ndarray di shape (M, N) con gli score F1.
        """
        m = len(candidates)
        n = len(references)

        if m == 0 or n == 0:
            return np.zeros((m, n))

        matrix = np.zeros((m, n))
        for i, cand in enumerate(candidates):
            scores = self.score_one_vs_many(cand, references)
            matrix[i, :] = scores

        return matrix
