"""
Modulo Lexical Overlap — Metriche lessicali per la matrice di supporto.

Implementa Exact Match e ROUGE-L per misurare la sovrapposizione
lessicale tra fatti atomici della risposta e frasi del contesto.
"""

from typing import Optional

import numpy as np


class LexicalOverlap:
    """
    Calcola metriche di overlap lessicale tra coppie di testi.

    Supporta:
      - Exact Match (match esatto dopo normalizzazione)
      - ROUGE-L (longest common subsequence)
    """

    def __init__(self):
        self._rouge_scorer = None  # Lazy loading

    @property
    def rouge_scorer(self):
        """Lazy-load dello scorer ROUGE."""
        if self._rouge_scorer is None:
            from rouge_score import rouge_scorer
            self._rouge_scorer = rouge_scorer.RougeScorer(
                ["rougeL"], use_stemmer=True
            )
        return self._rouge_scorer

    # ────────────────────────────────────────────────────────────────
    # Exact Match
    # ────────────────────────────────────────────────────────────────

    @staticmethod
    def exact_match(candidate: str, reference: str) -> float:
        """
        Verifica se il candidato è contenuto nel riferimento o viceversa.

        Restituisce 1.0 se c'è match esatto (dopo normalizzazione),
        altrimenti 0.0. Verifica anche la contenenza parziale.

        Args:
            candidate: Testo candidato (fatto atomico).
            reference: Testo di riferimento (frase del contesto).

        Returns:
            1.0 se match, 0.0 altrimenti.
        """
        cand_norm = candidate.lower().strip()
        ref_norm = reference.lower().strip()

        if cand_norm == ref_norm:
            return 1.0
        if cand_norm in ref_norm or ref_norm in cand_norm:
            return 0.8  # Contenenza parziale
        return 0.0

    # ────────────────────────────────────────────────────────────────
    # ROUGE-L
    # ────────────────────────────────────────────────────────────────

    def rouge_l(self, candidate: str, reference: str) -> float:
        """
        Calcola il ROUGE-L F-measure tra candidato e riferimento.

        ROUGE-L misura la Longest Common Subsequence (LCS)
        normalizzata, catturando la struttura sequenziale condivisa.

        Args:
            candidate: Testo candidato (fatto atomico).
            reference: Testo di riferimento (frase del contesto).

        Returns:
            ROUGE-L F-measure (float tra 0 e 1).
        """
        scores = self.rouge_scorer.score(reference, candidate)
        return scores["rougeL"].fmeasure

    # ────────────────────────────────────────────────────────────────
    # Score combinato per una coppia
    # ────────────────────────────────────────────────────────────────

    def score_pair(self, candidate: str, reference: str) -> dict[str, float]:
        """
        Calcola tutte le metriche lessicali per una coppia.

        Returns:
            Dict con "exact_match", "rouge_l", "combined".
        """
        em = self.exact_match(candidate, reference)
        rl = self.rouge_l(candidate, reference)
        # Score combinato: prende il massimo tra le due metriche
        combined = max(em, rl)
        return {
            "exact_match": em,
            "rouge_l": rl,
            "combined": combined,
        }

    # ────────────────────────────────────────────────────────────────
    # Matrice completa M×N
    # ────────────────────────────────────────────────────────────────

    def score_matrix(
        self, candidates: list[str], references: list[str]
    ) -> np.ndarray:
        """
        Calcola la matrice M×N di overlap lessicale (ROUGE-L).

        Args:
            candidates: Lista di M fatti atomici.
            references: Lista di N frasi di contesto.

        Returns:
            np.ndarray di shape (M, N) con gli score ROUGE-L.
        """
        m = len(candidates)
        n = len(references)

        if m == 0 or n == 0:
            return np.zeros((m, n))

        matrix = np.zeros((m, n))
        for i, cand in enumerate(candidates):
            for j, ref in enumerate(references):
                matrix[i, j] = self.rouge_l(cand, ref)

        return matrix
