from datetime import datetime, timezone

from corvid.agent.fill import fill_from_recall
from corvid.contracts import Provenance, QuoteRequest
from corvid.memory.port import RecalledFact


def make_request(**overrides) -> QuoteRequest:
    data = {
        "requester": {
            "name": "John Doe",
            "email": "john.doe@example.com",
            "company": "Acme Corp",
        },
        "origin": {"name": None},
        "destination": {"name": "Miami"},
    }
    data.update(overrides)
    return QuoteRequest.model_validate(data)


def ships_from(
    target: str = "Cartagena",
    customer: str = "Acme Corp",
    uuid: str = "edge-1",
    valid_at: datetime | None = None,
) -> RecalledFact:
    return RecalledFact(
        fact=f"{customer} ships from {target}",
        uuid=uuid,
        edge_name="SHIPS_FROM",
        source_name=customer,
        target_name=target,
        valid_at=valid_at,
    )


def test_fills_missing_field_with_learned_provenance():
    """A matching edge fills the gap and records where the value came from."""
    # ARRANGE:
    request = make_request()
    provenance: dict[str, Provenance] = {}
    valid_at = datetime(2026, 1, 1, tzinfo=timezone.utc)
    recalled = {"origin.name": [ships_from(valid_at=valid_at)]}

    # ACT:
    fill_from_recall(request, provenance, recalled)

    # ASSERT:
    assert request.origin.name == "Cartagena"
    assert provenance["origin.name"] == Provenance(
        source="learned", fact_uuid="edge-1", valid_at=valid_at
    )


def test_ignores_facts_with_wrong_edge_type():
    """A SHIPS_TO edge never answers the origin, however relevant the search thought it was."""
    # ARRANGE:
    request = make_request()
    provenance: dict[str, Provenance] = {}
    wrong = RecalledFact(
        fact="Acme Corp ships to Miami",
        uuid="edge-2",
        edge_name="SHIPS_TO",
        source_name="Acme Corp",
        target_name="Miami",
        valid_at=None,
    )
    recalled = {"origin.name": [wrong]}

    # ACT:
    fill_from_recall(request, provenance, recalled)

    # ASSERT:
    assert request.origin.name is None
    assert provenance == {}


def test_fills_when_the_extracted_company_disagrees_with_the_graph():
    """Extraction inferring 'acme-alimentos' from the domain while the graph
    says 'ACME Alimentos SAS' must not drop the fact: trust is anchored on
    the sender's contact node at recall, not on a company string match."""
    # ARRANGE:
    request = make_request(
        requester={
            "name": "Marta Restrepo",
            "email": "marta@acme-alimentos.example.com",
            "company": "acme-alimentos",
        }
    )
    provenance: dict[str, Provenance] = {}
    recalled = {"origin.name": [ships_from(customer="ACME Alimentos SAS")]}

    # ACT:
    fill_from_recall(request, provenance, recalled)

    # ASSERT:
    assert request.origin.name == "Cartagena"
    assert provenance["origin.name"].source == "learned"


def test_prefers_the_newest_fact():
    """With several candidates, the latest valid_at wins; undated facts lose to dated ones."""
    # ARRANGE:
    request = make_request()
    provenance: dict[str, Provenance] = {}
    recalled = {
        "origin.name": [
            ships_from(target="Undated", uuid="edge-0", valid_at=None),
            ships_from(
                target="Bogota",
                uuid="edge-1",
                valid_at=datetime(2025, 6, 1, tzinfo=timezone.utc),
            ),
            ships_from(
                target="Cartagena",
                uuid="edge-2",
                valid_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
            ),
        ]
    }

    # ACT:
    fill_from_recall(request, provenance, recalled)

    # ASSERT:
    assert request.origin.name == "Cartagena"
    assert provenance["origin.name"].fact_uuid == "edge-2"


def test_fills_requester_name_from_source_endpoint():
    """WORKS_FOR holds the value on the source node, not the target."""
    # ARRANGE:
    request = make_request(
        requester={"name": None, "email": "erik@nordfrost.se", "company": "Nordfrost"}
    )
    provenance: dict[str, Provenance] = {}
    works_for = RecalledFact(
        fact="Erik Lindqvist works for Nordfrost",
        uuid="edge-3",
        edge_name="WORKS_FOR",
        source_name="Erik Lindqvist",
        target_name="Nordfrost",
        valid_at=None,
    )
    recalled = {"requester.name": [works_for]}

    # ACT:
    fill_from_recall(request, provenance, recalled)

    # ASSERT:
    assert request.requester.name == "Erik Lindqvist"
    assert provenance["requester.name"].source == "learned"


def test_fills_without_a_company():
    """A signature-less email carries no company, but recall was anchored on
    the sender's email — its facts still fill."""
    # ARRANGE:
    request = make_request(
        requester={"name": None, "email": "john.doe@example.com", "company": None}
    )
    provenance: dict[str, Provenance] = {}
    recalled = {"origin.name": [ships_from()]}

    # ACT:
    fill_from_recall(request, provenance, recalled)

    # ASSERT:
    assert request.origin.name == "Cartagena"
    assert provenance["origin.name"].source == "learned"
