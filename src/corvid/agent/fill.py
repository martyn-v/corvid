from datetime import datetime
from typing import Literal

from corvid.agent.recall import RECALL_EDGES
from corvid.contracts import Provenance, QuoteRequest, apply_fill
from corvid.memory.port import RecalledFact


def _value_endpoint(fact: RecalledFact, endpoint: Literal["source", "target"]) -> str:
    return fact.target_name if endpoint == "target" else fact.source_name


def _customer_endpoint(
    fact: RecalledFact, endpoint: Literal["source", "target"]
) -> str:
    return fact.source_name if endpoint == "target" else fact.target_name


def choose_fact(
    path: str, facts: list[RecalledFact], customer: str
) -> RecalledFact | None:
    """Picks the recalled edge that answers the field: right type, right customer, newest first."""
    expected = RECALL_EDGES.get(path)
    if expected is None:
        return None
    edge_name, endpoint = expected

    candidates = [
        f
        for f in facts
        if f.edge_name == edge_name and _customer_endpoint(f, endpoint) == customer
    ]
    if not candidates:
        return None
    # datetime.min only ever compares against itself: the bool ranks dated facts above undated
    return max(
        candidates, key=lambda f: (f.valid_at is not None, f.valid_at or datetime.min)
    )


def fill_from_recall(
    request: QuoteRequest,
    provenance: dict[str, Provenance],
    recalled: dict[str, list[RecalledFact]],
) -> None:
    """Fills missing fields in place from recalled edges, recording learned provenance."""
    customer = request.requester.company
    if customer is None:
        return  # no anchor to trust any recalled fact against

    for path in request.missing():
        fact = choose_fact(path, recalled.get(path, []), customer)
        if fact is None:
            continue
        _, endpoint = RECALL_EDGES[path]
        apply_fill(
            request,
            provenance,
            path,
            _value_endpoint(fact, endpoint),
            Provenance(source="learned", fact_uuid=fact.uuid, valid_at=fact.valid_at),
        )
