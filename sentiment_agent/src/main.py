"""CLI entry point — natural language query interface."""

import argparse
import json
import sys

from src.graph.builder import get_graph
from src.data_sources.newsapi_feed import fetch_and_analyze


def main():
    parser = argparse.ArgumentParser(
        description="Agentic Sentiment Pipeline — natural language queries"
    )
    parser.add_argument(
        "query",
        nargs="?",
        default="",
        help='Natural language query, e.g. "Impact of RBI rate hike on PSU banks"',
    )
    parser.add_argument("--json", action="store_true", help="Output raw JSON")
    args = parser.parse_args()

    if not args.query:
        print("Usage: uv run python -m src.main <query>")
        print('Example: uv run python -m src.main "RBI hikes repo rate by 25 bps"')
        sys.exit(1)

    query = args.query.strip()
    lower = query.lower()

    # News branch — stays outside graph for now because it loops over headlines
    news_triggers = ["news", "headlines", "fetch"]
    if any(t in lower for t in news_triggers):
        print(f"Fetching latest headlines matching: {query}")
        india_only = "india" in lower
        narrative = any(w in lower for w in ["explain", "why", "how", "what is", "impact"])

        results = fetch_and_analyze(query=query, narrative=narrative, india_only=india_only)

        if args.json:
            print(json.dumps(results, indent=2, default=str))
            return

        print(f"\n=== Analyzed {len(results)} headlines ===\n")
        for r in results:
            h = r["headline"]
            src = r["source"]
            out = r["pipeline"]
            report = out.get("report", {})
            signals = report.get("signals", [])
            top = report.get("ranking", [])[:3]
            print(f"Headline: {h}")
            print(f"Source: {src} | Signals: {len(signals)} | Status: {out.get('status')}")
            if out.get("status") == "REJECT":
                print(f"  REJECTED: {out.get('reasons')}")
            else:
                for s in top:
                    print(f"  -> {s['ticker']}: impact={s['impact']:.2f} ({s['confidence']})")
            print()
        return

    # Single headline / topic analysis — graph handles everything
    graph = get_graph()
    initial_state = {
        "query": query,           # raw user input — intent parser will clean this
        "headline": "",           # populated by intent_parser_node
        "narrative_flag": False,  # populated by intent_parser_node
        "moderation": {},
        "routing": {},
        "macro_context": None,
        "psu_results": [],
        "nbfc_results": [],
        "report": None,
        "narrative": None,
        "status": "PENDING",
        "reasons": [],
    }

    result = graph.invoke(initial_state)

    if args.json:
        print(json.dumps(result, indent=2, default=str))
        return

    if result["status"] == "REJECT":
        print("REJECTED:", ", ".join(result["reasons"]))
        sys.exit(1)

    report = result["report"]
    if report:
        print(f"\nQuery: {query}")
        print(f"Extracted headline: {result['headline']}")
        print(f"Signals generated: {len(report['signals'])}")
        macro = report.get("macro_context", {})
        if macro and macro.get("policy_type") != "unknown":
            print(f"Macro context: {macro['policy_type']} ({macro.get('reason', '')})")
        print(f"Summary: {report['summary']}\n")

        print("Top signals:")
        for s in report["ranking"][:5]:
            print(
                f"  {s['ticker']}: impact={s['impact']:.2f}  "
                f"sentiment={s['sentiment']:.2f}  "
                f"relevance={s['relevance']:.2f}  ({s['confidence']})"
            )

    if result.get("narrative"):
        print(f"\nNarrative:\n  {result['narrative']}")


if __name__ == "__main__":
    main()
