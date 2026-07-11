"""
Modulo Highlight Renderer — Visualizzazione Highlight Gradient per Streamlit.

Genera HTML/CSS per colorare il testo generato con un gradient basato
sullo score di supporto lessicale (ROUGE-L + Exact Match):

  🟢  Verde intenso  →  score ≥ 0.8  (alto supporto)
  🟡  Giallo         →  score 0.5–0.8  (supporto parziale)
  🟠  Arancione      →  score 0.3–0.5  (supporto debole)
  🔴  Rosso          →  score < 0.3  (potenziale allucinazione)

Ogni frase ha un tooltip mouse-over che mostra:
  - Score di supporto, ROUGE-L e Exact Match
  - Il testo esatto del chunk sorgente usato per generare la claim
  - L'identificativo della fonte (anime title)

Il rendering è pensato per st.markdown() con unsafe_allow_html=True.
"""

from typing import Optional

from config import settings
from src.attribution.matrix import AttributionResult


class HighlightRenderer:
    """
    Genera HTML con highlight gradient e tooltip per visualizzare
    l'attribuzione nel frontend Streamlit.
    """

    # ── Palette colori (RGBA per controllo fine della sfumatura) ────
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
        """Restituisce il CSS globale per lo stile degli highlight e tooltips."""
        return """
        <style>
            .rag-attribution-container {
                font-family: 'Inter', 'Segoe UI', sans-serif;
                line-height: 1.85;
                padding: 1.5rem;
                background: rgba(15, 23, 42, 0.6);
                border-radius: 14px;
                border: 1px solid rgba(148, 163, 184, 0.1);
                backdrop-filter: blur(12px);
                box-shadow: 0 4px 20px rgba(0, 0, 0, 0.15);
            }
            .rag-fact {
                display: inline;
                padding: 3px 7px;
                border-radius: 5px;
                margin: 1px 0;
                transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1);
                cursor: pointer;
                position: relative;
                border-bottom: 2px solid transparent;
            }
            .rag-fact:hover {
                filter: brightness(1.25);
                transform: translateY(-1px);
                box-shadow: 0 2px 8px rgba(0, 0, 0, 0.2);
            }

            /* ── Tooltip (mouse-over) ─────────────────────────── */
            .rag-tooltip {
                visibility: hidden;
                opacity: 0;
                position: absolute;
                bottom: calc(100% + 12px);
                left: 50%;
                transform: translateX(-50%) translateY(4px);
                background: linear-gradient(135deg, #1e293b, #0f172a);
                color: #e2e8f0;
                padding: 14px 18px;
                border-radius: 10px;
                font-size: 0.8rem;
                line-height: 1.55;
                min-width: 320px;
                max-width: 440px;
                z-index: 10000;
                box-shadow: 0 12px 40px rgba(0, 0, 0, 0.4),
                            0 0 0 1px rgba(148, 163, 184, 0.15);
                transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1);
                pointer-events: none;
            }
            .rag-fact:hover .rag-tooltip {
                visibility: visible;
                opacity: 1;
                transform: translateX(-50%) translateY(0);
            }
            .rag-tooltip::after {
                content: '';
                position: absolute;
                top: 100%;
                left: 50%;
                margin-left: -7px;
                border-width: 7px;
                border-style: solid;
                border-color: #0f172a transparent transparent transparent;
            }

            /* ── Tooltip inner elements ───────────────────────── */
            .tooltip-header {
                display: flex;
                justify-content: space-between;
                align-items: center;
                margin-bottom: 8px;
                padding-bottom: 8px;
                border-bottom: 1px solid rgba(148, 163, 184, 0.15);
            }
            .tooltip-score-badge {
                display: inline-flex;
                align-items: center;
                gap: 4px;
                padding: 2px 10px;
                border-radius: 12px;
                font-size: 0.75rem;
                font-weight: 600;
            }
            .tooltip-metrics {
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 4px 16px;
                font-size: 0.78rem;
                margin-bottom: 8px;
            }
            .tooltip-metric-label {
                color: #94a3b8;
                font-size: 0.72rem;
                text-transform: uppercase;
                letter-spacing: 0.04em;
            }
            .tooltip-metric-value {
                font-weight: 600;
            }
            .tooltip-evidence {
                background: rgba(30, 41, 59, 0.8);
                border-left: 3px solid;
                padding: 8px 10px;
                border-radius: 0 6px 6px 0;
                font-size: 0.76rem;
                line-height: 1.5;
                color: #cbd5e1;
                font-style: italic;
                margin-top: 8px;
            }
            .tooltip-source-tag {
                display: inline-block;
                margin-top: 6px;
                padding: 2px 8px;
                border-radius: 8px;
                background: rgba(99, 102, 241, 0.15);
                color: #818cf8;
                font-size: 0.7rem;
                font-weight: 500;
            }

            /* ── Score bar ────────────────────────────────────── */
            .rag-score-bar {
                height: 4px;
                border-radius: 3px;
                margin-top: 8px;
                background: rgba(148, 163, 184, 0.15);
                overflow: hidden;
            }
            .rag-score-fill {
                height: 100%;
                border-radius: 3px;
                transition: width 0.5s ease;
            }

            /* ── Legend ───────────────────────────────────────── */
            .rag-legend {
                display: flex;
                gap: 16px;
                padding: 14px 0 4px;
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

            /* ── Overall score ────────────────────────────────── */
            .rag-overall-score {
                text-align: right;
                font-size: 0.85rem;
                color: #94a3b8;
                padding: 8px 0;
            }
        </style>
        """

    # ────────────────────────────────────────────────────────────────
    # Rendering di un singolo fatto con tooltip
    # ────────────────────────────────────────────────────────────────

    def render_fact(self, result: AttributionResult) -> str:
        """
        Genera l'HTML per un singolo fatto atomico con highlight e tooltip.

        Il tooltip mostra:
          - Support Score con badge colorato
          - ROUGE-L score
          - Exact Match score
          - Il testo esatto del chunk sorgente recuperato
          - Il nome della fonte (anime title)

        Args:
            result: AttributionResult con score e evidenza.

        Returns:
            Stringa HTML per il fatto colorato con tooltip.
        """
        bg_color = self._score_to_rgba(result.support_score)
        border_color = self._score_to_border(result.support_score)
        score_pct = f"{result.support_score * 100:.0f}"
        level = self._get_color_level(result.support_score)

        # Badge color per il tooltip header
        badge_bg = {
            "high": "rgba(16, 185, 129, 0.2)",
            "medium": "rgba(245, 158, 11, 0.2)",
            "low": "rgba(249, 115, 22, 0.2)",
            "none": "rgba(239, 68, 68, 0.2)",
        }[level]

        level_label = {
            "high": "✅ High Support",
            "medium": "⚠️ Partial",
            "low": "🔶 Weak",
            "none": "🔴 Hallucination Risk",
        }[level]

        # ── Costruzione del tooltip ───────────────────────────────
        tooltip = (
            # Header con badge
            f"<div class='tooltip-header'>"
            f"<span class='tooltip-score-badge' style='background:{badge_bg}; color:{border_color};'>"
            f"{level_label}</span>"
            f"<span style='font-weight:700; color:{border_color};'>{score_pct}%</span>"
            f"</div>"
            # Metriche
            f"<div class='tooltip-metrics'>"
            f"<div><span class='tooltip-metric-label'>ROUGE-L</span><br>"
            f"<span class='tooltip-metric-value'>{result.rouge_l:.3f}</span></div>"
            f"<div><span class='tooltip-metric-label'>Exact Match</span><br>"
            f"<span class='tooltip-metric-value'>{result.exact_match:.3f}</span></div>"
            f"</div>"
        )

        # Evidenza sorgente (Full Source Chunk)
        if result.source_chunk_text:
            evidence_escaped = (
                result.source_chunk_text
                .replace("&", "&amp;")
                .replace("<", "&lt;")
                .replace(">", "&gt;")
                .replace("'", "&#39;")
                .replace('"', "&quot;")
            )
            # Tronca a 400 caratteri in caso sia troppo lungo
            evidence_display = evidence_escaped[:400]
            if len(evidence_escaped) > 400:
                evidence_display += "..."

            if result.best_evidence_source:
                source_escaped = (
                    result.best_evidence_source
                    .replace("&", "&amp;")
                    .replace("<", "&lt;")
                    .replace(">", "&gt;")
                    .replace("'", "&#39;")
                )
                tooltip += (
                    f"<div style='margin-top:8px; margin-bottom:4px; font-weight:bold; font-size:0.8rem; color:{border_color};'>"
                    f"📄 Database Chunk: {source_escaped}"
                    f"</div>"
                )

            tooltip += (
                f"<div class='tooltip-evidence' style='border-left-color:{border_color}; font-size:0.8rem;'>"
                f"{evidence_display}"
                f"</div>"
            )

        # Score bar
        tooltip += (
            f"<div class='rag-score-bar'>"
            f"<div class='rag-score-fill' style='width:{score_pct}%; "
            f"background: linear-gradient(90deg, {border_color}, {border_color}88);'></div>"
            f"</div>"
        )

        # Escape the fact text too
        fact_escaped = (
            result.fact
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
        )

        return (
            f"<span class='rag-fact' style='"
            f"background:{bg_color}; "
            f"border-bottom-color:{border_color};'>"
            f"{fact_escaped} "
            f"<span class='rag-tooltip'>{tooltip}</span>"
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
        # CSS is now injected globally in app.py — no need to include it here.
        # This prevents Streamlit from stripping duplicate <style> blocks.
        html = ""

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

        # Header: etichette delle evidenze
        for j, ev_label in enumerate(evidence_labels):
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
