"""Orchestrator: dispatch manifests, collect results, synthesize."""

from src.registry import ManifestLoader
from src.routing import UniverseRouter
from src.moderation import ContentFilter
from src.synthesis import SynthesisCore
from src.agents import sentiment_run, macro_run, narrative_run


class DispatchEngine:
    def __init__(self):
        self.loader = ManifestLoader()
        self.router = UniverseRouter()
        self.moderator = ContentFilter()
        self.synthesis = SynthesisCore()

    def run(self, query: str, headline: str = "", narrative: bool = False) -> dict:
        """
        End-to-end pipeline execution.

        Args:
            query: user query text
            headline: news headline to analyze
            narrative: if True, run optional LLM narrative agent

        Returns:
            {"status": "PASS"|"REJECT", "report": {...}|None, "narrative": str|None, "reasons": [...]}
        """
        # 1. Moderation
        mod = self.moderator.check(query, headline)
        if mod["status"] == "REJECT":
            return {"status": "REJECT", "report": None, "narrative": None, "reasons": mod["reasons"]}

        # 2. Routing
        route = self.router.route(query, headline)
        universes = route["universes"]
        tickers = route["tickers"]
        is_macro = route["macro"]

        # 3. Dispatch macro agent first (if applicable)
        macro_context = None
        if is_macro:
            macro_context = macro_run(headline)

        # 4. Dispatch sentiment agents per universe
        agent_results = []
        for u in universes:
            # Explicit tickers that match this universe
            allowed = self.router.psu_tickers if u == "psu" else self.router.nbfc_tickers
            universe_tickers = [t for t in tickers if t in allowed]

            # If no explicit tickers, let the agent resolve from headline (pass None)
            # If macro event and still no tickers, agent will expand to full universe
            results = sentiment_run(
                headline,
                u,
                tickers=universe_tickers if universe_tickers else None,
                macro_context=macro_context,
            )
            agent_results.extend(results)

        # 5. Synthesis
        report = self.synthesis.synthesize(agent_results, macro_context)

        # 6. Optional narrative
        narrative_text = None
        if narrative:
            narrative_text = narrative_run(report, query)

        return {
            "status": "PASS",
            "report": report,
            "narrative": narrative_text,
            "reasons": [],
        }
