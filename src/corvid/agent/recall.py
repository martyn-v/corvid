from typing import Literal

from corvid.contracts import QuoteRequest
from corvid.memory.port import Memory, RecalledFact


RECALL_QUESTIONS = {
    "origin.name": "Where does {customer} ship from?",
    "destination.name": "Where does {customer} ship to?",
    "requester.name": "Who is the contact person for {customer}?",
}  # Questions mirror the ontology's fact templates — see DESIGN.md, "Decisions that hold at any scale"

RECALL_EDGES: dict[str, tuple[str, Literal["source", "target"]]] = {
    "origin.name": ("SHIPS_FROM", "target"),
    "destination.name": ("SHIPS_TO", "target"),
    "requester.name": ("WORKS_FOR", "source"),
}  # Which edge type answers each field, and which endpoint node holds the value


async def recall_missing_fields(
    quote_request: QuoteRequest, memory: Memory
) -> dict[str, list[RecalledFact]]:
    """Recalls missing fields from the knowledge graph."""
    customer = quote_request.requester.company
    if customer is None:
        return {}

    recalled: dict[str, list[RecalledFact]] = {}
    for path in quote_request.missing():
        template = RECALL_QUESTIONS.get(path)
        if template is None:
            recalled[path] = []  # memory has no way to know this field
        else:
            recalled[path] = await memory.recall(
                customer, template.format(customer=customer)
            )
    return recalled
