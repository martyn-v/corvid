import operator
from functools import partial
from typing import Annotated, Literal, NotRequired, TypedDict

from langchain_core.language_models import BaseChatModel
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import interrupt

from corvid.agent.ask import Exchange, apply_answers, questions_for
from corvid.agent.extract import extract_quote_request
from corvid.agent.fill import fill_from_recall
from corvid.agent.parse_email import ParsedEmail, parse_eml
from corvid.agent.recall import recall_missing_fields
from corvid.contracts import Provenance, QuoteRequest, present_fields
from corvid.memory.port import Memory, RecalledFact


class State(TypedDict):
    """Represents the state in the agent's state graph."""

    file_path: str
    parsed_email: NotRequired[ParsedEmail]
    quote_request: NotRequired[QuoteRequest]
    extraction_raw: NotRequired[dict]
    recalled: NotRequired[dict[str, list[RecalledFact]]]
    provenance: NotRequired[
        Annotated[dict[str, Provenance], operator.or_]
    ]  # Merge instead of replace; defaults to {} at runtime
    asked: NotRequired[list[Exchange]]  # every question put to the customer


def parse_node(state: State) -> dict:
    """Reads and parses the email file into the state."""
    with open(state["file_path"], "rb") as f:
        raw = f.read()

    result = parse_eml(raw)

    return {
        "parsed_email": result,
    }


def extract_node(state: State, *, model: BaseChatModel) -> dict:
    """Extracts the quote request from the parsed email."""

    if "parsed_email" not in state:
        raise ValueError("Parsed email is required for extracting quote request.")

    parsed_email = state["parsed_email"]

    result = extract_quote_request(parsed_email, model)

    return {
        "quote_request": result.request,
        "extraction_raw": result.raw,
        "provenance": {
            path: Provenance(source="email") for path in present_fields(result.request)
        },
    }


def route_after_extract(state: State) -> Literal["recall", "learn"]:
    if "quote_request" not in state:
        raise ValueError("Quote request is required for routing after extraction.")

    return "recall" if state["quote_request"].missing() else "learn"


async def recall_node(state: State, *, memory: Memory) -> dict:
    """Recalls missing fields from the knowledge graph."""
    if "quote_request" not in state:
        raise ValueError("Quote request is required for recall.")

    return {"recalled": await recall_missing_fields(state["quote_request"], memory)}


def fill_node(state: State) -> dict:
    """Fills in missing fields in the quote request from recalled data."""
    if "quote_request" not in state:
        raise ValueError("Quote request is required for filling.")

    request = state["quote_request"].model_copy(deep=True)
    provenance = dict(state.get("provenance", {}))
    before = set(provenance)

    fill_from_recall(request, provenance, state.get("recalled", {}))

    return {
        "quote_request": request,
        "provenance": {path: provenance[path] for path in provenance.keys() - before},
    }


def route_after_fill(state: State) -> Literal["ask", "learn"]:
    if "quote_request" not in state:
        raise ValueError("Quote request is required for routing after filling.")

    return "ask" if state["quote_request"].missing() else "learn"


def ask_node(state: State) -> dict:
    """Interrupts to ask the customer for the still-missing fields.

    The graph pauses here surfacing {path: question}; the caller resumes
    with {path: answer} for the questions the customer answered. Everything
    before the interrupt re-runs on resume, so it must stay pure.
    """
    if "quote_request" not in state:
        raise ValueError("Quote request is required for asking.")

    questions = questions_for(state["quote_request"])
    answers = interrupt(questions)

    request = state["quote_request"].model_copy(deep=True)
    provenance: dict[str, Provenance] = {}
    exchanges = apply_answers(request, provenance, questions, answers)

    return {
        "quote_request": request,
        "provenance": provenance,
        "asked": exchanges,
    }


def learn_node(state: State) -> dict:
    """Learns from the extracted and filled quote request and stores it in the knowledge graph."""
    return {}


def build_graph(
    model: BaseChatModel,
    memory: Memory,
    checkpointer: BaseCheckpointSaver | None = None,
) -> CompiledStateGraph[State]:
    """Builds a state graph for the agent.

    A checkpointer is required to cross the ask interrupt; without one the
    graph still runs requests that never need to ask.
    """

    builder = StateGraph(State)

    builder.add_node("parse_email", parse_node)
    builder.add_node("extract_request", partial(extract_node, model=model))
    builder.add_node("recall", partial(recall_node, memory=memory))
    builder.add_node("fill", fill_node)
    builder.add_node("ask", ask_node)
    builder.add_node("learn", learn_node)

    builder.add_edge(START, "parse_email")
    builder.add_edge("parse_email", "extract_request")
    builder.add_conditional_edges("extract_request", route_after_extract)
    builder.add_edge("recall", "fill")
    builder.add_conditional_edges("fill", route_after_fill)
    builder.add_edge("ask", "learn")
    builder.add_edge("learn", END)
    # GAPS: check if the extracted request has missing fields, if so go to RECALL, otherwise go to LEARN
    # RECALL: try to find missing fields using graphiti
    # FILL: fill in the missing fields in the extracted request, track provenance
    # GAPS2: check if the filled request has missing fields, if so go to ASK (simulates a request and response from customer), otherwise go to LEARN
    # LEARN: create an episode in the knowledge graph with the extracted request and provenance, and any filled fields

    return builder.compile(checkpointer=checkpointer)
