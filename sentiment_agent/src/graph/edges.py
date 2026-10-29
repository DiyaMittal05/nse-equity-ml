"""Conditional edge routers for the LangGraph state machine."""

from typing import Literal


def moderation_router(state) -> Literal["PASS", "REJECT"]:
    return state["status"]


def macro_router(state) -> Literal["macro", "skip"]:
    if state["routing"].get("macro"):
        return "macro"
    return "skip"


def narrative_router(state) -> Literal["narrative", "end"]:
    if state.get("narrative_flag"):
        return "narrative"
    return "end"
