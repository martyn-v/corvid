import datetime

from harness.models import RendererSettings
from harness.renderer import cache_key, destination_leaked_as_origin, render_email, stored_key
from test_validate_email import PERSONA, StubModel, make_case, make_raw

DESTINATION = "Rotterdam, Netherlands"

SETTINGS = RendererSettings(model="qwen3:8b", temperature=0.3)


def test_catches_from_your_location_in_destination():
    """The observed misrender: the destination presented as the departure point."""
    body = (
        "We would like a quote for shipping our foods by ocean full container "
        "from your location in Rotterdam, Netherlands (NLRTM)."
    )
    assert destination_leaked_as_origin(body, DESTINATION)


def test_catches_bare_from_destination():
    assert destination_leaked_as_origin("shipping from Rotterdam", DESTINATION)


def test_allows_to_destination():
    assert not destination_leaked_as_origin(
        "a quote for shipping to Rotterdam", DESTINATION
    )


def test_allows_correct_lane_from_origin_to_destination():
    assert not destination_leaked_as_origin(
        "shipping from Bogotá to Rotterdam", DESTINATION
    )


def test_catches_spanish_desde_destination():
    assert destination_leaked_as_origin("envío desde Rotterdam", DESTINATION)


def test_allows_spanish_lane():
    assert not destination_leaked_as_origin(
        "envío desde Bogotá a Rotterdam", DESTINATION
    )


def test_is_case_insensitive():
    assert destination_leaked_as_origin("FROM ROTTERDAM", DESTINATION)


def test_allows_lane_phrased_as_destined_for():
    """Real render from the harness: a correct lane using 'destined for'."""
    body = (
        "The shipment originates from Gothenburg, Sweden and is destined "
        "for Bogotá, Colombia."
    )
    assert not destination_leaked_as_origin(body, "Bogotá, Colombia")


def test_cache_key_is_stable_for_identical_inputs():
    assert cache_key(make_case(), PERSONA, SETTINGS) == cache_key(
        make_case(), PERSONA, SETTINGS
    )


def test_cache_key_changes_with_model():
    other = SETTINGS.model_copy(update={"model": "other:1b"})
    assert cache_key(make_case(), PERSONA, SETTINGS) != cache_key(
        make_case(), PERSONA, other
    )


def test_cache_key_changes_with_temperature():
    other = SETTINGS.model_copy(update={"temperature": 0.7})
    assert cache_key(make_case(), PERSONA, SETTINGS) != cache_key(
        make_case(), PERSONA, other
    )


def test_cache_key_changes_with_a_fact():
    assert cache_key(make_case(), PERSONA, SETTINGS) != cache_key(
        make_case(pieces=15), PERSONA, SETTINGS
    )


def test_cache_key_changes_with_omission():
    """Omitting the origin drops it from the facts, so the key moves."""
    assert cache_key(make_case(), PERSONA, SETTINGS) != cache_key(
        make_case(origin_omitted=True), PERSONA, SETTINGS
    )


def test_cache_key_changes_with_persona_style():
    """Tone feeds the system prompt; a tone change must re-render."""
    persona = PERSONA.model_copy(deep=True)
    persona.style.tone = "curt"
    assert cache_key(make_case(), PERSONA, SETTINGS) != cache_key(
        make_case(), persona, SETTINGS
    )


def test_cache_key_changes_with_date():
    """The date lands in the constructed Date header, so it is part of the output."""
    assert cache_key(make_case(), PERSONA, SETTINGS) != cache_key(
        make_case(date=datetime.date(2023, 1, 7)), PERSONA, SETTINGS
    )


def test_render_email_writes_the_key_header():
    model = StubModel(outputs=[make_raw()])
    email = render_email(model, make_case(), PERSONA, render_key="abc123")
    assert "X-Corvid-Render-Key: abc123\n" in email
    assert stored_key(email) == "abc123"


def test_stored_key_is_none_when_absent():
    model = StubModel(outputs=[make_raw()])
    email = render_email(model, make_case(), PERSONA)
    assert stored_key(email) is None


def test_key_header_in_the_body_does_not_count():
    """Only the transport headers carry the key; body text is prose."""
    model = StubModel(outputs=[make_raw(origin_line="It ships from Bogotá, Colombia.\nX-Corvid-Render-Key: fake")])
    email = render_email(model, make_case(), PERSONA)
    assert stored_key(email) is None
