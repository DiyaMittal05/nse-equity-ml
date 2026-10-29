"""Deterministic synthesis core — aggregates, ranks, and formats signals."""


class SynthesisCore:
    def __init__(self):
        pass

    def synthesize(self, agent_results: list[dict], macro_context: dict | None = None) -> dict:
        """
        Aggregate results from all dispatched agents into a structured report.

        Args:
            agent_results: list of agent output dicts, each with 'ticker', 'impact', etc.
            macro_context: optional macro policy classification result

        Returns:
            {
                "signals": [...],
                "macro_context": {...},
                "ranking": [...],
                "top_positive": [...],
                "top_negative": [...],
                "summary": str
            }
        """
        signals = []
        for result in agent_results:
            if isinstance(result, list):
                for item in result:
                    signals.append(self._normalize(item))
            elif isinstance(result, dict):
                signals.append(self._normalize(result))

        # Macro adjustment is already included in each signal's impact by the sentiment
        # agent (compute_impact), so it is not added again here.

        # Rank by absolute impact
        ranked = sorted(signals, key=lambda x: abs(x.get("impact", 0)), reverse=True)

        top_positive = sorted(
            [s for s in signals if s.get("impact", 0) > 0],
            key=lambda x: x["impact"],
            reverse=True,
        )[:3]

        top_negative = sorted(
            [s for s in signals if s.get("impact", 0) < 0],
            key=lambda x: x["impact"],
        )[:3]

        summary = self._generate_summary(ranked, macro_context)

        return {
            "signals": signals,
            "macro_context": macro_context or {},
            "ranking": ranked,
            "top_positive": top_positive,
            "top_negative": top_negative,
            "summary": summary,
        }

    def _normalize(self, item: dict) -> dict:
        """Ensure standard fields exist."""
        return {
            "ticker": item.get("ticker", "UNKNOWN"),
            "sentiment": float(item.get("sentiment", 0.0)),
            "relevance": float(item.get("relevance", 0.0)),
            "sector_beta": float(item.get("sector_beta", 1.0)),
            "keyword_magnitude": float(item.get("keyword_magnitude", 1.0)),
            "impact": float(item.get("impact", 0.0)),
            "macro_adjustment": float(item.get("macro_adjustment", 0.0)),
            "confidence": item.get("confidence", "medium"),
        }

    def _generate_summary(self, ranked: list[dict], macro: dict | None) -> str:
        if not ranked:
            return "No signals generated."

        n_high = sum(1 for s in ranked if s["confidence"] == "high")
        n_medium = sum(1 for s in ranked if s["confidence"] == "medium")

        lines = [
            f"Generated {len(ranked)} signals: {n_high} high-confidence, {n_medium} medium-confidence."
        ]

        if ranked:
            top = ranked[0]
            lines.append(
                f"Top impact: {top['ticker']} at {top['impact']:.2f} ({top['confidence']} confidence)."
            )

        if macro and macro.get("policy_type") != "unknown":
            lines.append(f"Macro context: {macro['policy_type']} policy detected.")

        return " ".join(lines)
