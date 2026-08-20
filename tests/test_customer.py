import datetime

from harness.customer import answer_question
from harness.models import Case, Contact, EmailStyle, Location, Persona


def _persona() -> Persona:
    return Persona(
        id="acme-alimentos",
        company="ACME Alimentos SAS",
        contact=Contact(
            name="Marta Restrepo", email="marta.restrepo@acme-alimentos.example.com"
        ),
        origin=Location(locode="COBOG", name="Bogotá, Colombia"),
        destination=Location(locode="NLRTM", name="Rotterdam, Netherlands"),
        mode="ocean_fcl",
        commodity="packaged non-perishable foods",
        omit_origin=True,
        style=EmailStyle(language="en", tone="terse"),
    )


def _case() -> Case:
    return Case(
        index=9,
        persona="acme-alimentos",
        date=datetime.date(2023, 3, 6),
        origin=Location(locode="COBOG", name="Bogotá, Colombia"),
        destination=Location(locode="NLRTM", name="Rotterdam, Netherlands"),
        mode="ocean_fcl",
        commodity="packaged non-perishable foods",
        pieces=14,
        weight_kg=9385,
        origin_omitted=True,
        n=18,
    )


def test_answers_locations_from_the_case_as_city_names():
    """Locations come from the case (they follow persona changes), city name only."""
    assert answer_question(_case(), _persona(), "origin.name") == "Bogotá"
    assert answer_question(_case(), _persona(), "destination.name") == "Rotterdam"


def test_answers_requester_fields_from_the_persona():
    assert answer_question(_case(), _persona(), "requester.name") == "Marta Restrepo"
    assert (
        answer_question(_case(), _persona(), "requester.email")
        == "marta.restrepo@acme-alimentos.example.com"
    )
    assert (
        answer_question(_case(), _persona(), "requester.company")
        == "ACME Alimentos SAS"
    )


def test_unknown_field_goes_unanswered():
    assert answer_question(_case(), _persona(), "origin.code") is None
