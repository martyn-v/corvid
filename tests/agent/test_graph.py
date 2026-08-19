import json
import pytest
from langchain_core.language_models import GenericFakeChatModel
from corvid.agent.graph import State
from corvid.agent.graph import build_graph
from langgraph.graph.state import CompiledStateGraph


@pytest.fixture
def live_graph() -> CompiledStateGraph[State]:
    """Fixture to provide a live graph instance for testing."""
    extraction_response = json.dumps(
        {
            "requester": {"name": "John Doe", "email": "john.doe@example.com"},
            "origin": {"name": "Cartagena"},
            "destination": {"name": "Miami"},
        }
    )

    fake_model = GenericFakeChatModel(messages=iter([extraction_response]))

    graph = build_graph(fake_model)
    return graph


@pytest.mark.asyncio
async def test_graph_happy_path(live_graph):
    """Test the happy path of the graph."""
    # ARRANGE:

    state: State = {
        "file_path": "tests/fixtures/emails/001-acme-alimentos-01.eml",
    }

    # ACT:
    result = await live_graph.ainvoke(state)

    # ASSERT:
    assert result["parsed_email"] is not None
    assert result["quote_request"] is not None
