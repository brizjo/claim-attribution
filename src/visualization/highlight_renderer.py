"""
Modulo Highlight Renderer — Visualizzazione gradient per Streamlit.

Questo modulo genera HTML/CSS per colorare il testo generato con un
gradient basato sullo score di supporto dalla matrice di attribuzione:

  🟢  Verde intenso  →  score ≥ 0.8  (alto supporto)
  🟡  Giallo         →  score 0.5–0.8  (supporto parziale)
  🟠  Arancione      →  score 0.3–0.5  (supporto debole)
  🔴  Rosso          →  score < 0.3  (potenziale allucinazione)

Il rendering è pensato per st.markdown() con unsafe_allow_html=True.
"""

from typing import Optional

from config import settings
from src.attribution.matrix import AttributionResult


class HighlightRenderer:
    """
    Genera HTML con highlight gradient per visualizzare l'attribuzione
    nel frontend Streamlit.
    """

    # ── Palette colori (HSLA per controllo fine della sfumatura) ────
    COLORS = {
        "high":    {"bg": "rgba(16, 185, 129, {alpha})", "border": "#10b981"},   # Emerald
        "medium":  {"bg": "rgba(245, 158, 11, {alpha})", "border": "#f59e0b"},   # Amber
        "low":     {"bg": "rgba(249, 115, 22, {alpha})", "border": "#f97316"},   # Orange
        "none":    {"bg": "rgba(239, 68, 68, {alpha})",  "border": "#ef4444"},   # Red
    }

    def __init__(
        self,
        threshold_high: float = settings.SUPPORT_THRESHOLD_HIGH,
        threshold_medium: float = settings.SUPPORT_THRESHOLD_MEDIUM,
        threshold_low: float = settings.SUPPORT_THRESHOLD_LOW,
    ):
        self.threshold_high = threshold_high
        self.threshold_medium = threshold_medium
        self.threshold_low = threshold_low

    # ────────────────────────────────────────────────────────────────
    # Score → Colore
    # ────────────────────────────────────────────────────────────────

    def _get_color_level(self, score: float) -> str:
        """Determina il livello di colore in base allo score."""
        if score >= self.threshold_high:
            return "high"
        elif score >= self.threshold_medium:
            return "medium"
        elif score >= self.threshold_low:
            return "low"
        else:
            return "none"

    def _score_to_rgba(self, score: float) -> str:
        """Converte uno score in un colore RGBA con alpha proporzionale."""
        level = self._get_color_level(score)
        # Alpha proporzionale allo score (minimo 0.15 per visibilità)
        alpha = max(0.15, min(1.0, score))
        return self.COLORS[level]["bg"].format(alpha=f"{alpha:.2f}")

    def _score_to_border(self, score: float) -> str:
        """Restituisce il colore del bordo per il livello di score."""
        level = self._get_color_level(score)
        return self.COLORS[level]["border"]

    # ────────────────────────────────────────────────────────────────
    # CSS Globale
    # ────────────────────────────────────────────────────────────────

    @staticmethod
    def get_global_css() -> str:
        """Restituisce il CSS globale per lo stile degli highlight."""
        return """
        <style>
            .rag-attribution-container {
                font-family: 'Inter', 'Segoe UI', sans-serif;
                line-height: 1.8;
                padding: 1.2rem;
                background: rgba(15, 23, 42, 0.6);
                border-radius: 12px;
                border: 1px solid rgba(148, 163, 184, 0.1);
                backdrop-filter: blur(8px);
            }
            .rag-fact {
                display: inline;
                padding: 2px 6px;
                border-radius: 4px;
                margin: 1px 0;
                transition: all 0.3s ease;
                cursor: pointer;
                position: relative;
                border-bottom: 2px solid transparent;
            }
            .rag-fact:hover {
                filter: brightness(1.2);
                transform: translateY(-1px);
            }
            .rag-tooltip {
                visibility: hidden;
                opacity: 0;
                position: absolute;
                bottom: 125%;
                left: 50%;
                transform: translateX(-50%);
                background: #1e293b;
                color: #e2e8f0;
                padding: 10px 14px;
                border-radius: 8px;
                font-size: 0.78rem;
                line-height: 1.5;
                min-width: 280px;
                max-width: 400px;
                z-index: 1000;
                box-shadow: 0 10px 25px rgba(0,0,0,0.3);
                border: 1px solid rgba(148, 163, 184, 0.2);
                transition: all 0.2s ease;
                pointer-events: none;
            }
            .rag-fact:hover .rag-tooltip {
                visibility: visible;
                opacity: 1;
            }
            .rag-tooltip::after {
                content: '';
                position: absolute;
                top: 100%;
                left: 50%;
                margin-left: -6px;
                border-width: 6px;
                border-style: solid;
                border-color: #1e293b transparent transparent transparent;
            }
            .rag-score-bar {
                height: 4px;
                border-radius: 2px;
                margin-top: 6px;
                background: rgba(148, 163, 184, 0.2);
                overflow: hidden;
            }
            .rag-score-fill {
                height: 100%;
                border-radius: 2px;
                transition: width 0.5s ease;
            }
            .rag-legend {
                display: flex;
                gap: 16px;
                padding: 12px 0 4px;
                font-size: 0.8rem;
                color: #94a3b8;
                flex-wrap: wrap;
            }
            .rag-legend-item {
                display: flex;
                align-items: center;
                gap: 6px;
            }
            .rag-legend-dot {
                width: 12px;
                height: 12px;
                border-radius: 3px;
            }
            .rag-overall-score {
                text-align: right;
                font-size: 0.85rem;
                color: #94a3b8;
                padding: 8px 0;
            }
        </style>
        """

    # ────────────────────────────────────────────────────────────────
    # Rendering di un singolo fatto
    # ────────────────────────────────────────────────────────────────

    def render_fact(self, result: AttributionResult) -> str:
        """
        Genera l'HTML per un singolo fatto atomico con highlight e tooltip.

        Args:
            result: AttributionResult con score e evidenza.

        Returns:
            Stringa HTML per il fatto colorato.
        """
        bg_color = self._score_to_rgba(result.support_score)
        border_color = self._score_to_border(result.support_score)
        score_pct = f"{result.support_score * 100:.0f}"

        # Tooltip con dettagli dell'attribuzione
        tooltip_content = (
            f"<strong>Support Score:</strong> {score_pct}%<br>"
            f"<strong>BERTScore F1:</strong> {result.bertscore:.3f}<br>"
            f"<strong>Lexical (ROUGE-L):</strong> {result.lexical_score:.3f}<br>"
        )
        if result.best_evidence:
            # Escape HTML nel testo dell'evidenza
            evidence_escaped = (
                result.best_evidence
                .replace("&", "&amp;")
                .replace("<", "&lt;")
                .replace(">", "&gt;")
            )
            tooltip_content += (
                f"<hr style='border-color: rgba(148,163,184,0.2); margin: 6px 0;'>"
                f"<strong>Best Evidence:</strong><br>"
                f"<em>'{evidence_escaped[:150]}...'</em><br>"
                f"<span style='color:#64748b'>— {result.best_evidence_source}</span>"
            )

        # Score bar nel tooltip
        tooltip_content += (
            f"<div class='rag-score-bar'>"
            f"<div class='rag-score-fill' style='width:{score_pct}%; "
            f"background:{border_color};'></div>"
            f"</div>"
        )

        return (
            f"<span class='rag-fact' style='"
            f"background:{bg_color}; "
            f"border-bottom-color:{border_color};'>"
            f"{result.fact} "
            f"<span class='rag-tooltip'>{tooltip_content}</span>"
            f"</span>"
        )

    # ────────────────────────────────────────────────────────────────
    # Rendering completo della risposta
    # ────────────────────────────────────────────────────────────────

    def render_response(
        self,
        attribution_results: list[AttributionResult],
        overall_score: Optional[float] = None,
    ) -> str:
        """
        Genera l'HTML completo per la risposta con highlight gradient.

        Args:
            attribution_results: Lista di AttributionResult per ogni fatto.
            overall_score:       Score complessivo medio (opzionale).

        Returns:
            Stringa HTML completa da inserire in st.markdown().
        """
        # CSS globale
        html = self.get_global_css()

        # Container principale
        html += "<div class='rag-attribution-container'>"

        # Score complessivo
        if overall_score is not None:
            score_color = self._score_to_border(overall_score)
            html += (
                f"<div class='rag-overall-score'>"
                f"Overall Attribution Score: "
                f"<strong style='color:{score_color}'>"
                f"{overall_score * 100:.0f}%</strong>"
                f"</div>"
            )

        # Rendering di ogni fatto con il suo highlight
        html += "<p>"
        for result in attribution_results:
            html += self.render_fact(result) + " "
        html += "</p>"

        # Legenda
        html += (
            "<div class='rag-legend'>"
            "<span class='rag-legend-item'>"
            "<span class='rag-legend-dot' style='background:#10b981'></span>"
            "High Support (≥80%)</span>"
            "<span class='rag-legend-item'>"
            "<span class='rag-legend-dot' style='background:#f59e0b'></span>"
            "Partial (50-80%)</span>"
            "<span class='rag-legend-item'>"
            "<span class='rag-legend-dot' style='background:#f97316'></span>"
            "Weak (30-50%)</span>"
            "<span class='rag-legend-item'>"
            "<span class='rag-legend-dot' style='background:#ef4444'></span>"
            "Hallucination Risk (&lt;30%)</span>"
            "</div>"
        )

        html += "</div>"
        return html

    # ────────────────────────────────────────────────────────────────
    # Rendering della matrice come heatmap HTML
    # ────────────────────────────────────────────────────────────────

    def render_matrix_heatmap(
        self,
        matrix,
        fact_labels: list[str],
        evidence_labels: list[str],
    ) -> str:
        """
        Genera una heatmap HTML della matrice di supporto.

        Args:
            matrix:           np.ndarray (M×N).
            fact_labels:      Etichette delle righe (fatti atomici).
            evidence_labels:  Etichette delle colonne (frasi contesto).

        Returns:
            Stringa HTML della heatmap.
        """
        html = """
        <style>
            .matrix-table {
                border-collapse: collapse;
                font-size: 0.75rem;
                width: 100%;
                overflow-x: auto;
            }
            .matrix-table th, .matrix-table td {
                border: 1px solid rgba(148, 163, 184, 0.15);
                padding: 6px 8px;
                text-align: center;
                max-width: 200px;
                overflow: hidden;
                text-overflow: ellipsis;
                white-space: nowrap;
            }
            .matrix-table th {
                background: rgba(30, 41, 59, 0.8);
                color: #94a3b8;
                font-weight: 500;
            }
            .matrix-table td.fact-label {
                text-align: left;
                font-weight: 500;
                color: #e2e8f0;
                background: rgba(30, 41, 59, 0.5);
            }
        </style>
        <div style="overflow-x: auto;">
        <table class='matrix-table'>
        <thead><tr><th>Fact \\ Evidence</th>
        """

        # Header: etichette delle evidenze (troncate)
        for j, ev_label in enumerate(evidence_labels):
            truncated = ev_label[:40] + "..." if len(ev_label) > 40 else ev_label
            html += f"<th title='{ev_label}'>E{j+1}</th>"

        html += "</tr></thead><tbody>"

        # Righe: ogni fatto atomico
        for i, fact_label in enumerate(fact_labels):
            truncated_fact = fact_label[:50] + "..." if len(fact_label) > 50 else fact_label
            html += f"<tr><td class='fact-label' title='{fact_label}'>{truncated_fact}</td>"
            for j in range(len(evidence_labels)):
                score = float(matrix[i, j])
                bg = self._score_to_rgba(score)
                html += f"<td style='background:{bg}' title='Score: {score:.3f}'>{score:.2f}</td>"
            html += "</tr>"

        html += "</tbody></table></div>"
        return html
