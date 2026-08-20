from datetime import datetime
from typing import Protocol

from pydantic import BaseModel


class RecalledFact(BaseModel):
    fact: str
    uuid: str
    valid_at: datetime | None


class Memory(Protocol):
    """A memory interface for recalling and learning facts."""

    async def recall(self, customer: str, question: str) -> list[RecalledFact]:
        """Recalls a fact from memory."""
        ...

    async def learn(self, name: str, body: str, date: datetime) -> None:
        """Adds one email episode to memory."""
        ...


class FakeMemory:
    """A fake memory implementation for testing."""

    def __init__(self, facts: list[RecalledFact] | None = None):
        self.facts = facts or []
        self.episodes: list[tuple[str, str, datetime]] = []

    async def recall(self, customer: str, question: str) -> list[RecalledFact]:
        return self.facts

    async def learn(self, name: str, body: str, date: datetime) -> None:
        self.episodes.append((name, body, date))
