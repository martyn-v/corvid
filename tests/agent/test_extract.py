import json
import pytest
from langchain_core.language_models import GenericFakeChatModel
from datetime import datetime

from corvid.agent.extract import (
    QuoteRequestExtractionError,
    extract_quote_request,
    render_email_prompt,
)
from corvid.agent.parse_email import EmailAddress, ParsedEmail

MINIMAL_VALID_QUOTE_REQUEST_JSON = json.dumps(
    {
        "requester": {"name": "John Doe", "email": "john.doe@example.com"},
        "origin": {"name": "Cartagena"},
        "destination": {"name": "Miami"},
    }
)

EMAIL = ParsedEmail(
    raw=b"raw bytes",
    subject="Request for Quote",
    sender=EmailAddress(display_name="John Doe", address="john.doe@example.com"),
    date=datetime(2024, 6, 1, 12, 0, 0),
    body_text="Please provide a quote for the following request.",
    warnings=[],
)


def test_extract_quote_request_with_minimal_valid_input():
    # ARRANGE:
    fake_model = GenericFakeChatModel(messages=iter([MINIMAL_VALID_QUOTE_REQUEST_JSON]))

    # ACT:
    result = extract_quote_request(EMAIL, model=fake_model)

    # ASSERT:
    assert result.request.requester.name == "John Doe"
    assert result.request.requester.email == "john.doe@example.com"
    assert result.request.origin.name == "Cartagena"
    assert result.request.destination.name == "Miami"


def test_render_email_prompt_includes_from_subject_and_body():
    """The model sees the headers, not just the body — the requester lives in From."""
    prompt = render_email_prompt(EMAIL)
    assert prompt == (
        "From: John Doe <john.doe@example.com>\n"
        "Subject: Request for Quote\n"
        "\n"
        "Please provide a quote for the following request."
    )


def test_render_email_prompt_omits_absent_headers():
    """Missing sender/subject drop their header lines instead of rendering None."""
    email = EMAIL.model_copy(update={"subject": None, "sender": None})
    prompt = render_email_prompt(email)
    assert prompt == "Please provide a quote for the following request."


def test_render_email_prompt_sender_without_display_name():
    """A bare address renders without the empty <> decoration."""
    email = EMAIL.model_copy(
        update={"sender": EmailAddress(display_name=None, address="jd@example.com")}
    )
    assert render_email_prompt(email).startswith("From: jd@example.com\n")


def test_extract_sends_rendered_email_to_the_model():
    """extract_quote_request passes headers + body as the human message."""
    captured: list = []

    class RecordingModel(GenericFakeChatModel):
        def _generate(self, messages, **kwargs):
            captured.append(messages)
            return super()._generate(messages, **kwargs)

    model = RecordingModel(messages=iter([MINIMAL_VALID_QUOTE_REQUEST_JSON]))
    extract_quote_request(EMAIL, model=model)

    human = captured[0][-1]
    assert human.content == render_email_prompt(EMAIL)


def test_throws_on_invalid_json():
    # ARRANGE:
    fake_model = GenericFakeChatModel(messages=iter(["not a json"]))

    # ACT & ASSERT:
    with pytest.raises(QuoteRequestExtractionError) as exc_info:
        extract_quote_request(EMAIL, model=fake_model)

    assert "Model did not return valid JSON" in str(exc_info.value)


def test_throws_on_invalid_schema():
    # ARRANGE:
    invalid_json = json.dumps({"invalid_field": "value"})
    fake_model = GenericFakeChatModel(messages=iter([invalid_json]))

    # ACT & ASSERT:
    with pytest.raises(QuoteRequestExtractionError) as exc_info:
        extract_quote_request(EMAIL, model=fake_model)

    assert "Extraction failed schema validation" in str(exc_info.value)
