"""Build and compile the LangGraph state machine."""

from langgraph.graph import StateGraph, END

from .state import PipelineState
from .nodes import (
    intent_parser_node,
    moderation_node,
    routing_node,
    macro_node,
    sentiment_node,
    synthesis_node,
    narrative_node,
)
from .edges import moderation_router, macro_router, narrative_router


def build_pipeline_graph():
    graph = StateGraph(PipelineState)

    # Nodes
    graph.add_node("intent_parser", intent_parser_node)
    graph.add_node("moderation", moderation_node)
    graph.add_node("routing", routing_node)
    graph.add_node("macro", macro_node)
    graph.add_node("sentiment", sentiment_node)
    graph.add_node("synthesis", synthesis_node)
    graph.add_node("narrative", narrative_node)

    # Entry point
    graph.set_entry_point("intent_parser")

    # Intent parser -> moderation
    graph.add_edge("intent_parser", "moderation")

    # Moderation -> routing (PASS) or END (REJECT)
    graph.add_conditional_edges(
        "moderation",
        moderation_router,
        {"PASS": "routing", "REJECT": END},
    )

    # Routing -> macro (if macro headline) or sentiment (skip)
    graph.add_conditional_edges(
        "routing",
        macro_router,
        {"macro": "macro", "skip": "sentiment"},
    )

    # Macro -> sentiment
    graph.add_edge("macro", "sentiment")

    # Sentiment -> synthesis
    graph.add_edge("sentiment", "synthesis")

    # Synthesis -> narrative (if requested) or END
    graph.add_conditional_edges(
        "synthesis",
        narrative_router,
        {"narrative": "narrative", "end": END},
    )

    # Narrative -> END
    graph.add_edge("narrative", END)

    return graph.compile()


_pipeline_graph = None


def get_graph():
    global _pipeline_graph
    if _pipeline_graph is None:
        _pipeline_graph = build_pipeline_graph()
    return _pipeline_graph
