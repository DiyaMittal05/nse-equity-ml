"""Narrative Agent — generates natural language from synthesis output."""


def run(synthesis_report: dict, user_query: str) -> str:
    """
    Generate a natural language summary from synthesis JSON.
    Currently uses template-based rendering (no LLM).
    When LLM key available, switch to prompt-based generation.

    Args:
        synthesis_report: output from SynthesisCore
        user_query: original user query

    Returns:
        Human-readable paragraph
    """
    signals = synthesis_report.get("signals", [])
    macro = synthesis_report.get("macro_context", {})

    if not signals:
        return "No significant sentiment signals detected for this query."

    # Sort by absolute impact
    top = sorted(signals, key=lambda x: abs(x.get("impact", 0)), reverse=True)[:3]

    lines = []

    # Macro context opening
    if macro and macro.get("policy_type") != "unknown":
        lines.append(
            f"The headline relates to a {macro['policy_type'].replace('_', ' ')} policy event, "
            f"which typically impacts the affected sectors because: {macro.get('reason', '')}"
        )

    # Ticker-specific lines
    for s in top:
        ticker = s["ticker"]
        impact = s["impact"]
        direction = "negative" if impact < 0 else "positive"
        confidence = s.get("confidence", "medium")
        relevance = s.get("relevance", 0.0)
        sentiment = s.get("sentiment", 0.0)

        lines.append(
            f"{ticker} shows a {confidence}-confidence {direction} impact of {impact:.2f}. "
            f"This is driven by a headline sentiment of {sentiment:.2f} with relevance {relevance:.2f}."
        )

    # Comparison if multiple tickers
    if len(top) >= 2:
        first, second = top[0], top[1]
        if abs(first["impact"]) > abs(second["impact"]) * 1.3:
            lines.append(
                f"{first['ticker']} is more affected than {second['ticker']}, "
                f"likely due to higher sector sensitivity or stronger headline relevance."
            )

    return " ".join(lines)
