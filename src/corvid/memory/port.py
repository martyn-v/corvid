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

    async def recall(
        self, email: str, question: str, contact_name: str | None = None
    ) -> list[RecalledFact]:
        """Recalls facts anchored on the sender's email address.

        The email is the only identifier guaranteed present in every
        message; the display-name is a fallback handle on the same contact.
        """
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
        self.recall_calls: list[dict] = []

    async def recall(
        self, email: str, question: str, contact_name: str | None = None
    ) -> list[RecalledFact]:
        self.recall_calls.append(
            {"email": email, "question": question, "contact_name": contact_name}
        )
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
