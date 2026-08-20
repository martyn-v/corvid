import pytest

from corvid.agent.recall import recall_missing_fields
from corvid.contracts import QuoteRequest
from corvid.memory.port import FakeMemory, RecalledFact


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

    fake_facts = [RecalledFact(fact="test", uuid="1234", valid_at=None)]
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
async def test_returns_empty_list_for_unknowable_keys():
    """Test that recall_missing_fields returns an empty list for keys that cannot be recalled."""
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

    fake_facts = [RecalledFact(fact="test", uuid="1234", valid_at=None)]
    memory = FakeMemory(fake_facts)

    # ACT:
    recalled = await recall_missing_fields(quote_request, memory)

    # ASSERT:
    assert "requester.email" in recalled
    assert recalled["requester.email"] == []


@pytest.mark.asyncio
async def test_returns_empty_array_when_company_is_none():
    """Test that recall_missing_fields returns an empty list when company is None."""

    # ARRANGE:
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
    assert recalled == {}
