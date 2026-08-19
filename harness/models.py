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


class GenerationSettings(BaseModel):
    seed: int
    emails_per_persona: int
    variables: GenerationVariables


class ConfigFile(BaseModel):
    personas: list[Persona]
    generation: GenerationSettings


class Fact(BaseModel):
    n: int | None
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
