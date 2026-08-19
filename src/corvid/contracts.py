from datetime import datetime
from typing import Literal
from pydantic import BaseModel, Field


class Requester(BaseModel):
    name: str | None = Field(
        default=None,
        description="The requester's name as written in the email, e.g. John Doe",
    )
    email: str | None = Field(
        default=None,
        description="The requester's email address as written in the email, e.g. john.doe@example.org",
    )
    company: str | None = Field(
        default=None,
        description="The requester's company as written in the email, e.g. Acme Inc.",
    )


class Location(BaseModel):
    name: str | None = Field(
        default=None,
        description="The location name as written in the email, e.g. Cartagena, ",
    )


class QuoteRequest(BaseModel):
    requester: Requester
    origin: Location
    destination: Location

    def missing(self) -> list[str]:
        """Returns a list of missing required fields in the quote request."""
        missing_fields = []
        if not self.requester.name:
            missing_fields.append("requester.name")
        if not self.requester.email:
            missing_fields.append("requester.email")
        if not self.origin.name:
            missing_fields.append("origin.name")
        if not self.destination.name:
            missing_fields.append("destination.name")
        return missing_fields


class Provenance(BaseModel):
    source: Literal["email", "learned", "answered_question"]
    fact_uuid: str | None = None  # graphiti edge, when source == "learned"
    valid_at: datetime | None = None  # what memory believed — feeds the stale check
