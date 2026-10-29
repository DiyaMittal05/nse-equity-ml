"""Shared typed state for the LangGraph pipeline."""

from typing import TypedDict, List, Dict, Any, Optional


class PipelineState(TypedDict):
    # Inputs
    query: str
    headline: str
    narrative_flag: bool

    # Intermediate
    moderation: Dict[str, Any]
    routing: Dict[str, Any]
    macro_context: Optional[Dict[str, Any]]
    psu_results: List[Dict[str, Any]]
    nbfc_results: List[Dict[str, Any]]

    # Outputs
    report: Optional[Dict[str, Any]]
    narrative: Optional[str]
    status: str
    reasons: List[str]
