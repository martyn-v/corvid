from functools import partial
from typing import TypedDict
from langchain_core.language_models import BaseChatModel
from typing import NotRequired
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from corvid.agent.extract import extract_quote_request
from corvid.agent.parse_email import parse_eml
from corvid.agent.parse_email import ParsedEmail
from corvid.contracts import QuoteRequest, Provenance


class State(TypedDict):
    """Represents the state in the agent's state graph."""

    file_path: str
    parsed_email: NotRequired[ParsedEmail]
    quote_request: NotRequired[QuoteRequest]
    extraction_raw: NotRequired[dict]
    provenance: NotRequired[
        dict[str, Provenance]
    ]  # FIXME: to implement provenance tracking in the graph, we need to add a provenance field to the state. This will allow us to track the source of each piece of information in the quote request, whether it was extracted from the email, learned from previous interactions, or answered through a question. The provenance field will be a dictionary mapping field names to Provenance objects, which contain information about the source and validity of each piece of data.


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
    }


def build_graph(model: BaseChatModel | None = None) -> CompiledStateGraph[State]:
    """Builds a state graph for the agent."""

    assert model is not None, "A language model must be provided to build the graph."

    builder = StateGraph(State)

    builder.add_node("parse_email", parse_node)
    builder.add_node("extract_request", partial(extract_node, model=model))

    builder.add_edge(START, "parse_email")
    builder.add_edge("parse_email", "extract_request")
    builder.add_edge("extract_request", END)
    # GAPS: check if the extracted request has missing fields, if so go to RECALL, otherwise go to LEARN
    # RECALL: try to find missing fields using graphiti
    # FILL: fill in the missing fields in the extracted request, track provenance
    # GAPS2: check if the filled request has missing fields, if so go to ASK (simulates a request and response from customer), otherwise go to LEARN
    # LEARN: create an episode in the knowledge graph with the extracted request and provenance, and any filled fields

    return builder.compile()
