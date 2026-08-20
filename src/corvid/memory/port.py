from datetime import datetime
from typing import Protocol

from pydantic import BaseModel


class RecalledFact(BaseModel):
    fact: str  # the edge's prose rendering — for logs, never parsed
    uuid: str
    edge_name: str  # edge type from the ontology, e.g. SHIPS_FROM
    source_name: str
    target_name: str
    valid_at: datetime | None
    invalid_at: datetime | None = None


class Memory(Protocol):
    """A memory interface for recalling and learning facts."""

    async def recall(self, customer: str, question: str) -> list[RecalledFact]:
        """Recalls a fact from memory."""
        ...

    async def learn(
        self,
        name: str,
        body: str,
        date: datetime,
        source_description: str = "customer email",
    ) -> None:
        """Adds one episode to memory, tagged with where it came from."""
        ...


class FakeMemory:
    """A fake memory implementation for testing."""

    def __init__(self, facts: list[RecalledFact] | None = None):
        self.facts = facts or []
        self.episodes: list[dict] = []

    async def recall(self, customer: str, question: str) -> list[RecalledFact]:
        return self.facts

    async def learn(
        self,
        name: str,
        body: str,
        date: datetime,
        source_description: str = "customer email",
    ) -> None:
        self.episodes.append(
            {
                "name": name,
                "body": body,
                "date": date,
                "source_description": source_description,
            }
        )
