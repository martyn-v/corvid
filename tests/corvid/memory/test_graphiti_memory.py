from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from corvid.memory.graphiti import GraphitiMemory, wipe_group


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
        self.search_calls: list[dict] = []
        self.episodes: list[dict] = []

    async def search(
        self, question, center_node_uuid=None, num_results=None, group_ids=None
    ):
        self.search_calls.append(
            {
                "question": question,
                "center_node_uuid": center_node_uuid,
                "group_ids": group_ids,
            }
        )
        return self._edges

    async def add_episode(self, **kwargs):
        self.episodes.append(kwargs)


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


CONTACT_LOOKUP = [{"uuid": "contact-1"}]
EMAIL = "marta@acme.example.com"


@pytest.mark.asyncio
async def test_recall_anchors_on_the_contact_email():
    """The anchor is the sender's Contact node matched by email — the only
    identifier every email is guaranteed to carry."""
    # ARRANGE:
    driver = FakeDriver(
        [
            CONTACT_LOOKUP,
            [
                {"uuid": "cust-1", "name": "Acme Corp"},
                {"uuid": "loc-1", "name": "Cartagena"},
            ],
        ]
    )
    fake = FakeGraphiti(driver, [edge()])
    memory = GraphitiMemory(fake)

    # ACT:
    facts = await memory.recall(EMAIL, "Where does the customer ship from?")

    # ASSERT: the anchor query matches a Contact node on its email attribute
    anchor_query, anchor_params = driver.calls[0]
    assert "Entity:Contact" in anchor_query
    assert "Customer" not in anchor_query
    assert anchor_params["email"] == EMAIL
    # ASSERT: that node centers the search
    assert fake.search_calls[0]["center_node_uuid"] == "contact-1"
    assert len(facts) == 1


@pytest.mark.asyncio
async def test_recall_falls_back_to_the_contact_name():
    """If no Contact carries the email, the From display-name finds the node."""
    # ARRANGE: email lookup misses, name lookup hits
    driver = FakeDriver(
        [
            [],
            CONTACT_LOOKUP,
            [
                {"uuid": "cust-1", "name": "Acme Corp"},
                {"uuid": "loc-1", "name": "Cartagena"},
            ],
        ]
    )
    fake = FakeGraphiti(driver, [edge()])
    memory = GraphitiMemory(fake)

    # ACT:
    facts = await memory.recall(
        EMAIL, "Where does the customer ship from?", contact_name="Marta Restrepo"
    )

    # ASSERT: second lookup matched the Contact by name and anchored the search
    fallback_query, fallback_params = driver.calls[1]
    assert "Entity:Contact" in fallback_query
    assert fallback_params["name"] == "Marta Restrepo"
    assert fake.search_calls[0]["center_node_uuid"] == "contact-1"
    assert len(facts) == 1


@pytest.mark.asyncio
async def test_recall_returns_empty_on_unknown_sender():
    """Cold start: no contact node by email or name, no search, no facts."""
    # ARRANGE:
    driver = FakeDriver([[], []])
    fake = FakeGraphiti(driver, [edge()])
    memory = GraphitiMemory(fake)

    # ACT:
    facts = await memory.recall(
        "nobody@nowhere.example.com",
        "Where does the customer ship from?",
        contact_name="Nobody",
    )

    # ASSERT:
    assert facts == []
    assert fake.search_calls == []


@pytest.mark.asyncio
async def test_recall_without_fallback_name_skips_the_name_lookup():
    """No display-name means one anchor attempt only."""
    # ARRANGE:
    driver = FakeDriver([[]])
    memory = GraphitiMemory(FakeGraphiti(driver, [edge()]))

    # ACT:
    facts = await memory.recall(EMAIL, "Where does the customer ship from?")

    # ASSERT:
    assert facts == []
    assert len(driver.calls) == 1


@pytest.mark.asyncio
async def test_recall_returns_enriched_facts():
    """Recall carries the edge type and resolved endpoint names, not just prose."""
    # ARRANGE:
    driver = FakeDriver(
        [
            CONTACT_LOOKUP,
            [
                {"uuid": "cust-1", "name": "Acme Corp"},
                {"uuid": "loc-1", "name": "Cartagena"},
            ],
        ]
    )
    memory = GraphitiMemory(FakeGraphiti(driver, [edge()]))

    # ACT:
    facts = await memory.recall(EMAIL, "Where does the customer ship from?")

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
            CONTACT_LOOKUP,
            [
                {"uuid": "cust-1", "name": "Acme Corp"},
                {"uuid": "loc-1", "name": "Cartagena"},
            ],
        ]
    )
    memory = GraphitiMemory(FakeGraphiti(driver, [stale, edge()]))

    # ACT:
    facts = await memory.recall(EMAIL, "Where does the customer ship from?")

    # ASSERT:
    assert [f.uuid for f in facts] == ["edge-1"]


@pytest.mark.asyncio
async def test_recall_resolves_names_in_one_query():
    """All endpoint names come from a single Cypher query over the uuids."""
    # ARRANGE:
    driver = FakeDriver(
        [
            CONTACT_LOOKUP,
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
    facts = await memory.recall(EMAIL, "Where does the customer ship to?")

    # ASSERT:
    assert len(driver.calls) == 2  # contact lookup + one name-resolution query
    assert {f.target_name for f in facts} == {"Cartagena", "Miami"}


@pytest.mark.asyncio
async def test_recall_skips_name_query_when_all_edges_invalidated():
    """No surviving edges means no name-resolution query and an empty result."""
    # ARRANGE:
    stale = edge(invalid_at=datetime(2026, 2, 1, tzinfo=timezone.utc))
    driver = FakeDriver([CONTACT_LOOKUP])
    memory = GraphitiMemory(FakeGraphiti(driver, [stale]))

    # ACT:
    facts = await memory.recall(EMAIL, "Where does the customer ship from?")

    # ASSERT:
    assert facts == []
    assert len(driver.calls) == 1


@pytest.mark.asyncio
async def test_recall_scopes_every_lookup_to_the_group():
    """With a group_id, the anchor lookups, search, and name query all filter by it."""
    # ARRANGE: email lookup misses so the name fallback runs too
    driver = FakeDriver(
        [
            [],
            CONTACT_LOOKUP,
            [
                {"uuid": "cust-1", "name": "Acme Corp"},
                {"uuid": "loc-1", "name": "Cartagena"},
            ],
        ]
    )
    fake = FakeGraphiti(driver, [edge()])
    memory = GraphitiMemory(fake, group_id="eval")

    # ACT:
    await memory.recall(
        EMAIL, "Where does the customer ship from?", contact_name="Marta Restrepo"
    )

    # ASSERT: every Cypher query filters on the group
    for query, params in driver.calls:
        assert "group_id" in query
        assert params["group_id"] == "eval"
    # ASSERT: the search is scoped too
    assert fake.search_calls[0]["group_ids"] == ["eval"]


@pytest.mark.asyncio
async def test_recall_without_group_is_unscoped():
    """No group_id keeps the legacy behavior: no filters anywhere."""
    # ARRANGE:
    driver = FakeDriver(
        [CONTACT_LOOKUP, [{"uuid": "cust-1", "name": "Acme Corp"},
                          {"uuid": "loc-1", "name": "Cartagena"}]]
    )
    fake = FakeGraphiti(driver, [edge()])
    memory = GraphitiMemory(fake)

    # ACT:
    await memory.recall(EMAIL, "Where does the customer ship from?")

    # ASSERT:
    for query, _ in driver.calls:
        assert "group_id" not in query
    assert fake.search_calls[0]["group_ids"] is None


@pytest.mark.asyncio
async def test_learn_writes_episodes_into_the_group():
    """Episodes carry the memory's group_id so graphs never cross-pollute."""
    # ARRANGE:
    fake = FakeGraphiti(FakeDriver([]), [])
    memory = GraphitiMemory(fake, group_id="eval")

    # ACT:
    await memory.learn("018-acme-09", "From: ...", datetime(2023, 3, 6))

    # ASSERT:
    assert fake.episodes[0]["group_id"] == "eval"
    assert fake.episodes[0]["source_description"] == "customer email"


@pytest.mark.asyncio
async def test_learn_tags_the_episode_source():
    """An answered-question episode carries its own source tag, per the design."""
    # ARRANGE:
    memory = GraphitiMemory(fake := FakeGraphiti(FakeDriver([]), []))

    # ACT:
    await memory.learn(
        "018-acme-09-answers",
        "Q: ...\nA: ...",
        datetime(2023, 3, 6),
        source_description="answered question",
    )

    # ASSERT:
    assert fake.episodes[0]["source_description"] == "answered question"


@pytest.mark.asyncio
async def test_wipe_group_detach_deletes_only_that_group():
    """wipe_group removes every node in the group and reports how many existed."""
    # ARRANGE:
    driver = FakeDriver([[{"count": 5}]])

    # ACT:
    count = await wipe_group(driver, "eval")

    # ASSERT:
    query, params = driver.calls[0]
    assert "DETACH DELETE" in query
    assert "group_id" in query
    assert params["group_id"] == "eval"
    assert count == 5
