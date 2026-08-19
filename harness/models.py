import datetime
from pydantic import BaseModel
from typing import Literal


class Contact(BaseModel):
    name: str
    email: str


class Location(BaseModel):
    locode: str
    name: str


class EmailStyle(BaseModel):
    language: Literal["en", "es"]
    tone: str


class PersonaChange(BaseModel):
    at_email: int
    origin: Location
    reason_in_email: str


class Persona(BaseModel):
    id: str
    company: str
    contact: Contact
    origin: Location
    destination: Location
    mode: Literal["ocean_fcl", "ocean_reefer"]
    commodity: str
    omit_origin: bool
    change: PersonaChange | None = None
    style: EmailStyle


class Range(BaseModel):
    min: int
    max: int


class GenerationVariables(BaseModel):
    pieces: Range
    weight_kg: Range


class Timeline(BaseModel):
    start_date: datetime.date
    start_offset_days: Range
    gap_days: Range


class RendererSettings(BaseModel):
    model: str
    temperature: float


class GenerationSettings(BaseModel):
    seed: int
    emails_per_persona: int
    omission_rate: float  # chance an omit_origin persona omits it, per email
    timeline: Timeline
    renderer: RendererSettings
    variables: GenerationVariables


class ConfigFile(BaseModel):
    personas: list[Persona]
    generation: GenerationSettings


class CaseDraft(BaseModel):
    """A case before global numbering; n is assigned after the date sort."""

    index: int
    persona: str
    date: datetime.date
    origin: Location
    destination: Location
    mode: Literal["ocean_fcl", "ocean_reefer"]
    commodity: str
    pieces: int
    weight_kg: int
    origin_omitted: bool
    change_reason: str | None = None


class Case(CaseDraft):
    n: int

    @property
    def key(self) -> str:
        """Names this case's email file; the world/agent filename contract."""
        return f"{self.n:03d}-{self.persona}-{self.index:02d}"
