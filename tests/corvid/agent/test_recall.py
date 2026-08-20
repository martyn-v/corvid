import pytest

from corvid.agent.recall import RECALL_EDGES, RECALL_QUESTIONS, recall_missing_fields
from corvid.contracts import QuoteRequest
from corvid.memory.ontology import edge_types
from corvid.memory.port import FakeMemory, RecalledFact


def test_recall_edges_mirror_recall_questions():
    """Every recall question has a matching edge type and value endpoint."""
    assert RECALL_EDGES == {
        "origin.name": ("SHIPS_FROM", "target"),
        "destination.name": ("SHIPS_TO", "target"),
        "requester.name": ("WORKS_FOR", "source"),
    }
    assert RECALL_EDGES.keys() == RECALL_QUESTIONS.keys()
    assert all(name in edge_types for name, _ in RECALL_EDGES.values())


def test_recall_questions_need_no_interpolation():
    """The center node supplies the customer context (DESIGN.md); questions
    are fixed strings that work even when the company is unknown."""
    for question in RECALL_QUESTIONS.values():
        assert "{" not in question


@pytest.mark.asyncio
async def test_recall_missing_fields():
    """Test the recall_missing_fields function."""
    # ARRANGE:
    quote_request = QuoteRequest.model_validate(
        {
            "requester": {
                "name": "John Doe",
                "email": "john.doe@example.com",
                "company": "Acme Corp",
            },
            "origin": {"name": None},
            "destination": {"name": None},
        }
    )

    fake_facts = [
        RecalledFact(
            fact="Acme Corp ships from Cartagena",
            uuid="1234",
            edge_name="SHIPS_FROM",
            source_name="Acme Corp",
            target_name="Cartagena",
            valid_at=None,
        )
    ]
    memory = FakeMemory(fake_facts)

    # ACT:
    recalled = await recall_missing_fields(quote_request, memory)

    # ASSERT:
    assert "origin.name" in recalled
    assert recalled["origin.name"] == fake_facts
    assert "destination.name" in recalled
    assert recalled["destination.name"] == fake_facts
    assert "requester.name" not in recalled


@pytest.mark.asyncio
async def test_recall_anchors_on_the_sender_email():
    """Every memory lookup is keyed on requester.email, with the display
    name as the fallback handle — never on the company."""
    # ARRANGE:
    quote_request = QuoteRequest.model_validate(
        {
            "requester": {
                "name": "John Doe",
                "email": "john.doe@example.com",
                "company": None,
            },
            "origin": {"name": None},
            "destination": {"name": "Miami"},
        }
    )
    memory = FakeMemory()

    # ACT:
    await recall_missing_fields(quote_request, memory)

    # ASSERT:
    assert memory.recall_calls  # company being unknown did not stop recall
    for call in memory.recall_calls:
        assert call["email"] == "john.doe@example.com"
        assert call["contact_name"] == "John Doe"


@pytest.mark.asyncio
async def test_recall_works_without_company_or_name():
    """A signature-less email still recalls: the From address is enough."""
    # ARRANGE:
    quote_request = QuoteRequest.model_validate(
        {
            "requester": {
                "name": None,
                "email": "john.doe@example.com",
                "company": None,
            },
            "origin": {"name": None},
            "destination": {"name": "Miami"},
        }
    )
    fake_facts = [
        RecalledFact(
            fact="Acme Corp ships from Cartagena",
            uuid="1234",
            edge_name="SHIPS_FROM",
            source_name="Acme Corp",
            target_name="Cartagena",
            valid_at=None,
        )
    ]
    memory = FakeMemory(fake_facts)

    # ACT:
    recalled = await recall_missing_fields(quote_request, memory)

    # ASSERT:
    assert recalled["origin.name"] == fake_facts
    assert recalled["requester.name"] == fake_facts  # recallable via WORKS_FOR


@pytest.mark.asyncio
async def test_returns_empty_list_for_unknowable_keys():
    """Test that recall_missing_fields returns an empty list for keys that cannot be recalled."""
    # ARRANGE: company is missing and no question template covers it
    quote_request = QuoteRequest.model_validate(
        {
            "requester": {
                "name": "John Doe",
                "email": "john.doe@example.com",
                "company": None,
            },
            "origin": {"name": None},
            "destination": {"name": None},
        }
    )

    memory = FakeMemory()

    # ACT:
    recalled = await recall_missing_fields(quote_request, memory)

    # ASSERT:
    assert "requester.company" in recalled
    assert recalled["requester.company"] == []


@pytest.mark.asyncio
async def test_returns_empty_dict_when_email_is_none():
    """No sender address means no anchor: nothing to recall against."""

    # ARRANGE:
    quote_request = QuoteRequest.model_validate(
        {
            "requester": {
                "name": "John Doe",
                "email": None,
                "company": "Acme Corp",
            },
            "origin": {"name": None},
            "destination": {"name": None},
        }
    )

    memory = FakeMemory()

    # ACT:
    recalled = await recall_missing_fields(quote_request, memory)

    # ASSERT:
    assert recalled == {}
    assert memory.recall_calls == []
