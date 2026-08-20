import json
import pytest
from langchain_core.language_models import GenericFakeChatModel
from corvid.agent.graph import State
from corvid.agent.graph import build_graph
from langgraph.graph.state import CompiledStateGraph


def build_live_graph(extraction_response: str) -> CompiledStateGraph[State]:
    """Builds a live graph with a fake model that returns the given extraction response."""
    fake_model = GenericFakeChatModel(messages=iter([extraction_response]))
    graph = build_graph(fake_model)
    return graph


@pytest.mark.asyncio
async def test_graph_happy_path():
    """Test the happy path of the graph."""
    # ARRANGE:

    state: State = {
        "file_path": "tests/fixtures/emails/001-acme-alimentos-01.eml",
    }

    extraction_response = json.dumps(
        {
            "requester": {"name": "John Doe", "email": "john.doe@example.com"},
            "origin": {"name": "Cartagena"},
            "destination": {"name": "Miami"},
        }
    )

    graph = build_live_graph(extraction_response)

    # ACT:
    result = await graph.ainvoke(state)

    # ASSERT:
    assert result["parsed_email"] is not None
    assert result["quote_request"] is not None
    assert {path: p.source for path, p in result["provenance"].items()} == {
        "requester.name": "email",
        "requester.email": "email",
        "origin.name": "email",
        "destination.name": "email",
    }


@pytest.mark.asyncio
async def test_graph_with_recall_and_fill():
    """Test the graph's ability to recall and fill missing fields."""
    # ARRANGE:
    state: State = {
        "file_path": "tests/fixtures/emails/001-acme-alimentos-01.eml",
    }

    extraction_response = json.dumps(
        {
            "requester": {"name": "John Doe", "email": "john.doe@example.com"},
            "origin": {"name": None},
            "destination": {"name": None},
        }
    )

    graph = build_live_graph(extraction_response)

    # ACT:
    result = await graph.ainvoke(state)

    # ASSERT:
    assert result["parsed_email"] is not None
    assert result["quote_request"] is not None
    assert result["quote_request"].missing() == ["origin.name", "destination.name"]
    assert {path: p.source for path, p in result["provenance"].items()} == {
        "requester.name": "email",
        "requester.email": "email",
    }
