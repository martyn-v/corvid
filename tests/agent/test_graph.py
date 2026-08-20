import json
import pytest
from langchain_core.language_models import GenericFakeChatModel
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command
from corvid.agent.graph import State
from corvid.agent.graph import build_graph
from langgraph.graph.state import CompiledStateGraph

from corvid.memory.port import FakeMemory, RecalledFact


def build_live_graph(
    extraction_response: str, facts: list[RecalledFact] = []
) -> CompiledStateGraph[State]:
    """Builds a live graph with a fake model that returns the given extraction response."""
    fake_model = GenericFakeChatModel(messages=iter([extraction_response]))
    fake_memory = FakeMemory(facts)
    graph = build_graph(fake_model, fake_memory)
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
            "requester": {
                "name": "John Doe",
                "email": "john.doe@example.com",
                "company": "Acme Alimentos",
            },
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
        "requester.company": "email",
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
            "requester": {
                "name": "John Doe",
                "email": "john.doe@example.com",
                "company": "Acme Alimentos",
            },
            "origin": {"name": None},
            "destination": {"name": "Miami"},
        }
    )

    facts: list[RecalledFact] = [
        RecalledFact(
            fact="Acme Alimentos ships from Cartagena",
            uuid="1",
            edge_name="SHIPS_FROM",
            source_name="Acme Alimentos",
            target_name="Cartagena",
            valid_at=None,
        )
    ]

    graph = build_live_graph(extraction_response, facts)

    # ACT:
    result = await graph.ainvoke(state)

    # ASSERT:
    assert result["parsed_email"] is not None
    assert result["quote_request"] is not None
    assert result["quote_request"].missing() == []
    assert result["quote_request"].origin.name == "Cartagena"
    assert {path: p.source for path, p in result["provenance"].items()} == {
        "requester.name": "email",
        "requester.email": "email",
        "requester.company": "email",
        "destination.name": "email",
        "origin.name": "learned",
    }
    assert result["provenance"]["origin.name"].fact_uuid == "1"
    assert result["recalled"] == {"origin.name": facts}


@pytest.mark.asyncio
async def test_graph_interrupts_to_ask_and_resumes_with_answers():
    """Fields still missing after fill pause the graph; resuming fills them as answered."""
    # ARRANGE: no company in the email and nothing recalled, so origin stays missing
    extraction_response = json.dumps(
        {
            "requester": {"name": "John Doe", "email": "john.doe@example.com"},
            "destination": {"name": "Miami"},
        }
    )
    fake_model = GenericFakeChatModel(messages=iter([extraction_response]))
    graph = build_graph(fake_model, FakeMemory(), checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": "case-1"}}
    state: State = {"file_path": "tests/fixtures/emails/001-acme-alimentos-01.eml"}

    # ACT: run until the ask interrupt
    result = await graph.ainvoke(state, config)

    # ASSERT: paused, surfacing one question per missing field
    questions = result["__interrupt__"][0].value
    assert set(questions) == {"requester.company", "origin.name"}

    # ACT: the customer answers one question and ignores the other
    result = await graph.ainvoke(
        Command(resume={"origin.name": "Cartagena"}), config
    )

    # ASSERT: the answer is filled with answered_question provenance
    assert result["quote_request"].origin.name == "Cartagena"
    assert result["provenance"]["origin.name"].source == "answered_question"
    assert result["quote_request"].missing() == ["requester.company"]
    assert "requester.company" not in result["provenance"]

    # ASSERT: both exchanges are recorded for LEARN to ingest
    by_path = {e.path: e for e in result["asked"]}
    assert by_path["origin.name"].answer == "Cartagena"
    assert by_path["requester.company"].answer is None


@pytest.mark.asyncio
async def test_graph_complete_request_skips_ask():
    """A fully stated request never interrupts."""
    extraction_response = json.dumps(
        {
            "requester": {
                "name": "John Doe",
                "email": "john.doe@example.com",
                "company": "Acme Alimentos",
            },
            "origin": {"name": "Cartagena"},
            "destination": {"name": "Miami"},
        }
    )
    fake_model = GenericFakeChatModel(messages=iter([extraction_response]))
    graph = build_graph(fake_model, FakeMemory(), checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": "case-2"}}

    result = await graph.ainvoke(
        {"file_path": "tests/fixtures/emails/001-acme-alimentos-01.eml"}, config
    )

    assert "__interrupt__" not in result
    assert result.get("asked", []) == []


@pytest.mark.asyncio
async def test_graph_learns_the_email_as_an_episode():
    """LEARN feeds the raw email (headers + body) to memory, named after the file."""
    # ARRANGE: a complete request, so the graph goes straight to learn
    extraction_response = json.dumps(
        {
            "requester": {
                "name": "John Doe",
                "email": "john.doe@example.com",
                "company": "Acme Alimentos",
            },
            "origin": {"name": "Cartagena"},
            "destination": {"name": "Miami"},
        }
    )
    fake_model = GenericFakeChatModel(messages=iter([extraction_response]))
    memory = FakeMemory()
    graph = build_graph(fake_model, memory)

    # ACT:
    result = await graph.ainvoke(
        {"file_path": "tests/fixtures/emails/001-acme-alimentos-01.eml"}
    )

    # ASSERT: one episode, named for the email file, body starts with the headers
    assert len(memory.episodes) == 1
    episode = memory.episodes[0]
    assert episode["name"] == "001-acme-alimentos-01"
    assert episode["body"].startswith("From: ")
    assert episode["date"] == result["parsed_email"].date
    assert episode["source_description"] == "customer email"


@pytest.mark.asyncio
async def test_graph_learns_answered_questions_as_their_own_episode():
    """Answers from the ask interrupt become a second, answered-question episode."""
    # ARRANGE: origin missing, nothing recalled → ask
    extraction_response = json.dumps(
        {
            "requester": {
                "name": "John Doe",
                "email": "john.doe@example.com",
                "company": "Acme Alimentos",
            },
            "destination": {"name": "Miami"},
        }
    )
    fake_model = GenericFakeChatModel(messages=iter([extraction_response]))
    memory = FakeMemory()
    graph = build_graph(fake_model, memory, checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": "case-3"}}

    # ACT:
    await graph.ainvoke(
        {"file_path": "tests/fixtures/emails/001-acme-alimentos-01.eml"}, config
    )
    await graph.ainvoke(Command(resume={"origin.name": "Cartagena"}), config)

    # ASSERT: email episode plus one Q&A episode with only the answered question
    assert [e["name"] for e in memory.episodes] == [
        "001-acme-alimentos-01",
        "001-acme-alimentos-01-answers",
    ]
    answers = memory.episodes[1]
    assert answers["source_description"] == "answered question"
    assert "Cartagena" in answers["body"]
    assert "Marta Restrepo" in answers["body"]  # the email's sender is the answerer
    assert "company" not in answers["body"]  # unanswered questions are not ingested


@pytest.mark.asyncio
async def test_graph_skips_answers_episode_when_nothing_was_answered():
    """An ask with no answers leaves only the email episode."""
    extraction_response = json.dumps(
        {
            "requester": {
                "name": "John Doe",
                "email": "john.doe@example.com",
                "company": "Acme Alimentos",
            },
            "destination": {"name": "Miami"},
        }
    )
    fake_model = GenericFakeChatModel(messages=iter([extraction_response]))
    memory = FakeMemory()
    graph = build_graph(fake_model, memory, checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": "case-4"}}

    await graph.ainvoke(
        {"file_path": "tests/fixtures/emails/001-acme-alimentos-01.eml"}, config
    )
    # every asked path present, all unanswered — never an empty dict
    await graph.ainvoke(
        Command(resume={"requester.company": None, "origin.name": None}), config
    )

    assert [e["name"] for e in memory.episodes] == ["001-acme-alimentos-01"]


def test_fill_node_returns_only_new_provenance():
    """fill_node reports just the fields it filled and leaves the input state untouched."""
    # ARRANGE:
    from corvid.agent.graph import fill_node
    from corvid.contracts import Provenance, QuoteRequest

    request = QuoteRequest.model_validate(
        {
            "requester": {
                "name": "John Doe",
                "email": "john.doe@example.com",
                "company": "Acme Alimentos",
            },
            "origin": {"name": None},
            "destination": {"name": "Miami"},
        }
    )
    state: State = {
        "file_path": "unused",
        "quote_request": request,
        "provenance": {"destination.name": Provenance(source="email")},
        "recalled": {
            "origin.name": [
                RecalledFact(
                    fact="Acme Alimentos ships from Cartagena",
                    uuid="1",
                    edge_name="SHIPS_FROM",
                    source_name="Acme Alimentos",
                    target_name="Cartagena",
                    valid_at=None,
                )
            ]
        },
    }

    # ACT:
    result = fill_node(state)

    # ASSERT:
    assert set(result["provenance"]) == {"origin.name"}
    assert result["quote_request"].origin.name == "Cartagena"
    assert request.origin.name is None  # input state not mutated
