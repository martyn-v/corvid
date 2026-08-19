import json
import pytest
from langchain_core.language_models import GenericFakeChatModel
from datetime import datetime

from corvid.agent.extract import (
    QuoteRequestExtractionError,
    extract_quote_request,
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
