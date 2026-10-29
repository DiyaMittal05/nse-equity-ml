"""LangGraph nodes — deterministic functions that operate on PipelineState."""

import re

from src.moderation import ContentFilter
from src.routing import UniverseRouter
from src.agents import sentiment_run, macro_run
from src.synthesis import SynthesisCore
from src.agents.narrative_agent import run as narrative_run

from .state import PipelineState

_moderator_inst = None
_router_inst = None
_synthesis_inst = None


def _moderator() -> ContentFilter:
    global _moderator_inst
    if _moderator_inst is None:
        _moderator_inst = ContentFilter()
    return _moderator_inst


def _router() -> UniverseRouter:
    global _router_inst
    if _router_inst is None:
        _router_inst = UniverseRouter()
    return _router_inst


def _synthesis() -> SynthesisCore:
    global _synthesis_inst
    if _synthesis_inst is None:
        _synthesis_inst = SynthesisCore()
    return _synthesis_inst


def intent_parser_node(state: PipelineState) -> PipelineState:
    raw = state.get("query", "").strip()
    if not raw:
        return {
            **state,
            "headline": "",
            "narrative_flag": False,
            "status": "REJECT",
            "reasons": ["Empty input"],
        }

    lower = raw.lower()

    narrative_triggers = [
        "explain", "what is", "what are", "how does", "how will", "how",
        "why did", "why", "tell me", "breakdown", "impact of", "meaning of",
        "interpret", "analyze", "what about", "describe",
    ]
    narrative_flag = any(t in lower for t in narrative_triggers)

    headline = raw

    prefixes = [
        r'^(what is|what are|how does|how will|how|why did|why|tell me about|explain|describe|analyze|interpret|breakdown|impact of|meaning of|what about|give me|show me)\s+',
        r'^(get me|fetch|find)\s+(the\s+)?(latest\s+|recent\s+)?',
    ]
    for p in prefixes:
        headline = re.sub(p, '', headline, flags=re.IGNORECASE)

    headline = re.sub(r'\s*\?$', '', headline, flags=re.IGNORECASE)
    headline = re.sub(r'\s*(please|pls|for me|today|now)$', '', headline, flags=re.IGNORECASE)

    if (headline.startswith('"') and headline.endswith('"')) or \
       (headline.startswith("'") and headline.endswith("'")):
        headline = headline[1:-1]

    headline = headline.strip()

    if not headline:
        headline = raw

    return {
        **state,
        "headline": headline,
        "narrative_flag": narrative_flag,
    }


def moderation_node(state: PipelineState) -> PipelineState:
    mod = _moderator().check(state["query"], state["headline"])
    return {
        **state,
        "moderation": mod,
        "status": mod["status"],
        "reasons": mod["reasons"],
    }


def routing_node(state: PipelineState) -> PipelineState:
    route = _router().route(state["query"], state["headline"])
    return {**state, "routing": route}


def macro_node(state: PipelineState) -> PipelineState:
    if state["routing"].get("macro"):
        ctx = macro_run(state["headline"])
        return {**state, "macro_context": ctx}
    return {**state, "macro_context": None}


def sentiment_node(state: PipelineState) -> PipelineState:
    headline = state["headline"]
    universes = state["routing"]["universes"]
    tickers = state["routing"]["tickers"]
    macro_ctx = state.get("macro_context")

    psu_results: list[dict] = []
    nbfc_results: list[dict] = []

    if "psu" in universes:
        psu_tickers = (
            [t for t in tickers if t in _router().psu_tickers]
            if tickers else None
        )
        psu_results = sentiment_run(
            headline, "psu", tickers=psu_tickers, macro_context=macro_ctx
        )

    if "nbfc" in universes:
        nbfc_tickers = (
            [t for t in tickers if t in _router().nbfc_tickers]
            if tickers else None
        )
        nbfc_results = sentiment_run(
            headline, "nbfc", tickers=nbfc_tickers, macro_context=macro_ctx
        )

    return {
        **state,
        "psu_results": psu_results,
        "nbfc_results": nbfc_results,
    }


def synthesis_node(state: PipelineState) -> PipelineState:
    all_results = state["psu_results"] + state["nbfc_results"]
    report = _synthesis().synthesize(all_results, state.get("macro_context"))
    return {**state, "report": report}


def narrative_node(state: PipelineState) -> PipelineState:
    if state.get("narrative_flag") and state["report"]:
        text = narrative_run(state["report"], state["query"])
        return {**state, "narrative": text}
    return {**state, "narrative": None}
