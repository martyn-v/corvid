from functools import partial
import operator
from typing import Literal, TypedDict
from langchain_core.language_models import BaseChatModel
from typing import NotRequired, Annotated
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from corvid.agent.extract import extract_quote_request
from corvid.agent.parse_email import parse_eml
from corvid.agent.parse_email import ParsedEmail
from corvid.contracts import Provenance, QuoteRequest, present_fields


class State(TypedDict):
    """Represents the state in the agent's state graph."""

    file_path: str
    parsed_email: NotRequired[ParsedEmail]
    quote_request: NotRequired[QuoteRequest]
    extraction_raw: NotRequired[dict]
    provenance: NotRequired[
        Annotated[dict[str, Provenance], operator.or_]
    ]  # Merge instead of replace; defaults to {} at runtime


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


def recall_node(state: State) -> dict:
    """Recalls missing fields from the knowledge graph."""
    return {}


def fill_node(state: State) -> dict:
    """Fills in missing fields in the quote request from recalled data."""
    return {}


def route_after_fill(state: State) -> Literal["ask", "learn"]:
    if "quote_request" not in state:
        raise ValueError("Quote request is required for routing after filling.")

    return "ask" if state["quote_request"].missing() else "learn"


def ask_node(state: State) -> dict:
    """Asks the customer for missing fields and fills them in."""
    return {}


def learn_node(state: State) -> dict:
    """Learns from the extracted and filled quote request and stores it in the knowledge graph."""
    return {}


def build_graph(model: BaseChatModel | None = None) -> CompiledStateGraph[State]:
    """Builds a state graph for the agent."""

    assert model is not None, "A language model must be provided to build the graph."

    builder = StateGraph(State)

    builder.add_node("parse_email", parse_node)
    builder.add_node("extract_request", partial(extract_node, model=model))
    builder.add_node("recall", recall_node)
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

    return builder.compile()
