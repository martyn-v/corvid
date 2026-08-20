import datetime

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from harness.models import Case, Contact, EmailStyle, Location, Persona
from harness.renderer import render_email, validate_email


class StubModel(BaseChatModel):
    """Stands in for the renderer LLM; replays canned outputs in order."""

    outputs: list[str]
    calls: int = 0

    @property
    def _llm_type(self) -> str:
        return "stub"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        self.calls += 1
        content = self.outputs[min(self.calls - 1, len(self.outputs) - 1)]
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=content))])

PERSONA = Persona(
    id="acme-alimentos",
    company="ACME Alimentos SAS",
    contact=Contact(name="Marta Restrepo", email="marta@acme.example.com"),
    origin=Location(locode="COBOG", name="Bogotá, Colombia"),
    destination=Location(locode="NLRTM", name="Rotterdam, Netherlands"),
    mode="ocean_fcl",
    commodity="packaged non-perishable foods",
    omit_origin=True,
    style=EmailStyle(language="en", tone="warm"),
)


def make_case(**overrides) -> Case:
    fields = dict(
        n=1,
        index=2,
        persona="acme-alimentos",
        date=datetime.date(2023, 1, 6),
        origin=PERSONA.origin,
        destination=PERSONA.destination,
        mode="ocean_fcl",
        commodity="packaged non-perishable foods",
        pieces=14,
        weight_kg=9385,
        origin_omitted=False,
    )
    fields.update(overrides)
    return Case(**fields)


def make_raw(
    *,
    origin_line="It ships from Bogotá, Colombia.",
    destination="Rotterdam, Netherlands",
    quantities="14 pieces weighing 9385 kg",
    signature="Marta Restrepo\nACME Alimentos SAS\nmarta@acme.example.com",
) -> str:
    return (
        f"Subject: Quote request to {destination}\n"
        "\n"
        "Hello,\n"
        "\n"
        f"We need a quote for {quantities} going to {destination}. {origin_line}\n"
        "\n"
        "Warm regards,\n"
        f"{signature}"
    )


def test_valid_email_has_no_problems():
    assert validate_email(make_raw(), make_case(), PERSONA) == []


def test_missing_destination_is_a_problem():
    raw = make_raw(destination="somewhere unnamed")
    assert any("destination" in p.lower() for p in validate_email(raw, make_case(), PERSONA))


def test_missing_stated_origin_is_a_problem():
    raw = make_raw(origin_line="")
    assert any("origin" in p.lower() for p in validate_email(raw, make_case(), PERSONA))


def test_omitted_origin_must_not_appear():
    raw = make_raw()  # mentions Bogotá
    case = make_case(origin_omitted=True)
    assert any("omitted" in p.lower() for p in validate_email(raw, case, PERSONA))


def test_omitted_origin_absent_is_valid():
    raw = make_raw(origin_line="")
    case = make_case(origin_omitted=True)
    assert validate_email(raw, case, PERSONA) == []


def test_transliterated_origin_counts_as_stated():
    """Models transliterate; 'Bogota' states the origin 'Bogotá'."""
    raw = make_raw(origin_line="It ships from Bogota.")
    assert validate_email(raw, make_case(), PERSONA) == []


def test_transliterated_origin_counts_as_a_leak_when_omitted():
    raw = make_raw(origin_line="It ships from Bogota.")
    case = make_case(origin_omitted=True)
    assert any("omitted" in p.lower() for p in validate_email(raw, case, PERSONA))


def test_city_match_is_case_insensitive():
    raw = make_raw(origin_line="It ships from BOGOTÁ.")
    assert validate_email(raw, make_case(), PERSONA) == []


def test_missing_pieces_is_a_problem():
    raw = make_raw(quantities="a shipment weighing 9385 kg")
    assert any("pieces" in p.lower() for p in validate_email(raw, make_case(), PERSONA))


def test_missing_weight_is_a_problem():
    raw = make_raw(quantities="14 pieces")
    assert any("weight" in p.lower() for p in validate_email(raw, make_case(), PERSONA))


def test_weight_with_thousands_separator_is_found():
    raw = make_raw(quantities="14 pieces weighing 9,385 kg")
    assert validate_email(raw, make_case(), PERSONA) == []


def test_spelled_out_pieces_count_as_stated():
    """Real render: 'twenty-one pieces of frozen seafood'."""
    raw = make_raw(quantities="fourteen pieces weighing 9385 kg")
    assert validate_email(raw, make_case(), PERSONA) == []


def test_spelled_out_weight_counts_as_stated():
    """Real render: 'nine thousand four hundred thirty-six kilograms'."""
    raw = make_raw(
        quantities="14 pieces weighing nine thousand three hundred eighty-five kilograms"
    )
    assert validate_email(raw, make_case(), PERSONA) == []


def test_spelled_out_numbers_follow_the_persona_language():
    persona = PERSONA.model_copy(deep=True)
    persona.style.language = "es"
    raw = make_raw(
        quantities="catorce piezas con un peso de nueve mil trescientos ochenta y cinco kg"
    )
    assert validate_email(raw, make_case(), persona) == []


def test_missing_sender_email_is_a_problem():
    raw = make_raw(signature="Marta Restrepo\nACME Alimentos SAS")
    problems = validate_email(raw, make_case(), PERSONA)
    assert any("marta@acme.example.com" in p for p in problems)


def test_missing_sender_name_is_a_problem():
    raw = make_raw(signature="ACME Alimentos SAS\nmarta@acme.example.com")
    problems = validate_email(raw, make_case(), PERSONA)
    assert any("Marta Restrepo" in p for p in problems)


def test_missing_company_is_a_problem():
    raw = make_raw(signature="Marta Restrepo\nmarta@acme.example.com")
    problems = validate_email(raw, make_case(), PERSONA)
    assert any("ACME Alimentos SAS" in p for p in problems)


def test_destination_leaked_as_origin_is_a_problem():
    raw = make_raw(
        origin_line="We ship from our location in Rotterdam, Netherlands."
    )
    problems = validate_email(raw, make_case(), PERSONA)
    assert any("departure" in p.lower() for p in problems)


def test_render_email_retries_until_valid():
    invalid = make_raw(quantities="14 pieces")  # weight missing
    model = StubModel(outputs=[invalid, make_raw()])
    email = render_email(model, make_case(), PERSONA)
    assert model.calls == 2
    assert "9385 kg" in email


def test_render_email_raises_listing_problems_when_attempts_run_out():
    model = StubModel(outputs=[make_raw(quantities="14 pieces")])
    with pytest.raises(ValueError, match="weight"):
        render_email(model, make_case(), PERSONA)
    assert model.calls == 3
