from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from corvid.memory.graphiti import GraphitiMemory


class FakeDriver:
    """Replays canned execute_query responses and records the calls."""

    def __init__(self, responses: list[list[dict]]):
        self.calls: list[tuple[str, dict]] = []
        self._responses = list(responses)

    async def execute_query(self, query: str, **params):
        self.calls.append((query, params))
        return self._responses.pop(0), None, None


class FakeGraphiti:
    def __init__(self, driver: FakeDriver, edges: list[SimpleNamespace]):
        self.driver = driver
        self._edges = edges

    async def search(self, question, center_node_uuid=None, num_results=None):
        return self._edges


def edge(**overrides) -> SimpleNamespace:
    base = dict(
        fact="Acme Corp ships from Cartagena",
        uuid="edge-1",
        name="SHIPS_FROM",
        source_node_uuid="cust-1",
        target_node_uuid="loc-1",
        valid_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        invalid_at=None,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


CUSTOMER_LOOKUP = [{"uuid": "cust-1"}]


@pytest.mark.asyncio
async def test_recall_returns_enriched_facts():
    """Recall carries the edge type and resolved endpoint names, not just prose."""
    # ARRANGE:
    driver = FakeDriver(
        [
            CUSTOMER_LOOKUP,
            [
                {"uuid": "cust-1", "name": "Acme Corp"},
                {"uuid": "loc-1", "name": "Cartagena"},
            ],
        ]
    )
    memory = GraphitiMemory(FakeGraphiti(driver, [edge()]))

    # ACT:
    facts = await memory.recall("Acme Corp", "Where does Acme Corp ship from?")

    # ASSERT:
    assert len(facts) == 1
    fact = facts[0]
    assert fact.fact == "Acme Corp ships from Cartagena"
    assert fact.uuid == "edge-1"
    assert fact.edge_name == "SHIPS_FROM"
    assert fact.source_name == "Acme Corp"
    assert fact.target_name == "Cartagena"
    assert fact.valid_at == datetime(2026, 1, 1, tzinfo=timezone.utc)
    assert fact.invalid_at is None


@pytest.mark.asyncio
async def test_recall_drops_invalidated_edges():
    """A superseded edge (invalid_at set) never reaches the caller."""
    # ARRANGE:
    stale = edge(
        uuid="edge-old",
        target_node_uuid="loc-old",
        invalid_at=datetime(2026, 2, 1, tzinfo=timezone.utc),
    )
    driver = FakeDriver(
        [
            CUSTOMER_LOOKUP,
            [
                {"uuid": "cust-1", "name": "Acme Corp"},
                {"uuid": "loc-1", "name": "Cartagena"},
            ],
        ]
    )
    memory = GraphitiMemory(FakeGraphiti(driver, [stale, edge()]))

    # ACT:
    facts = await memory.recall("Acme Corp", "Where does Acme Corp ship from?")

    # ASSERT:
    assert [f.uuid for f in facts] == ["edge-1"]


@pytest.mark.asyncio
async def test_recall_resolves_names_in_one_query():
    """All endpoint names come from a single Cypher query over the uuids."""
    # ARRANGE:
    driver = FakeDriver(
        [
            CUSTOMER_LOOKUP,
            [
                {"uuid": "cust-1", "name": "Acme Corp"},
                {"uuid": "loc-1", "name": "Cartagena"},
                {"uuid": "loc-2", "name": "Miami"},
            ],
        ]
    )
    edges = [
        edge(),
        edge(
            uuid="edge-2",
            name="SHIPS_TO",
            target_node_uuid="loc-2",
            fact="Acme Corp ships to Miami",
        ),
    ]
    memory = GraphitiMemory(FakeGraphiti(driver, edges))

    # ACT:
    facts = await memory.recall("Acme Corp", "Where does Acme Corp ship to?")

    # ASSERT:
    assert len(driver.calls) == 2  # customer lookup + one name-resolution query
    assert {f.target_name for f in facts} == {"Cartagena", "Miami"}


@pytest.mark.asyncio
async def test_recall_skips_name_query_when_all_edges_invalidated():
    """No surviving edges means no name-resolution query and an empty result."""
    # ARRANGE:
    stale = edge(invalid_at=datetime(2026, 2, 1, tzinfo=timezone.utc))
    driver = FakeDriver([CUSTOMER_LOOKUP])
    memory = GraphitiMemory(FakeGraphiti(driver, [stale]))

    # ACT:
    facts = await memory.recall("Acme Corp", "Where does Acme Corp ship from?")

    # ASSERT:
    assert facts == []
    assert len(driver.calls) == 1


@pytest.mark.asyncio
async def test_recall_returns_empty_on_unknown_customer():
    """Cold start: no customer node, no search, no facts."""
    # ARRANGE:
    driver = FakeDriver([[]])
    memory = GraphitiMemory(FakeGraphiti(driver, [edge()]))

    # ACT:
    facts = await memory.recall("Nobody Inc.", "Where does Nobody Inc. ship from?")

    # ASSERT:
    assert facts == []
    assert len(driver.calls) == 1
