import email
from email import policy
from pathlib import Path

import pytest

from corvid.agent.parse_email import EmailParseError, parse_eml

CASES = {
    "001-acme-alimentos-01": {
        "subject": "Quote Request for Shipment from Bogotá to Rotterdam",
        "sender_display_name": "Marta Restrepo",
        "sender_address": "marta.restrepo@acme-alimentos.example.com",
        "date": "2023-01-06T09:00:00+00:00",
        "body_str": """Hi there,

I hope you are having a wonderful week. This is Marta Restrepo""",
    }
}

FIXTURES = Path("tests") / "fixtures" / "emails"


def load(name: str) -> bytes:
    return (FIXTURES / f"{name}.eml").read_bytes()


@pytest.mark.parametrize("name", CASES.keys(), ids=CASES.keys())
def test_parse_eml(name):
    raw = load(name)
    result = parse_eml(raw)

    if "subject" in CASES[name]:
        assert result.subject == CASES[name]["subject"]
    if "sender_display_name" in CASES[name]:
        assert result.sender is not None
        assert result.sender.display_name == CASES[name]["sender_display_name"]
    if "sender_address" in CASES[name]:
        assert result.sender is not None
        assert result.sender.address == CASES[name]["sender_address"]
    if "date" in CASES[name]:
        assert result.date is not None
        assert result.date.isoformat() == CASES[name]["date"]
    if "body_str" in CASES[name]:
        assert result.body_text.strip().startswith(CASES[name]["body_str"].strip())


def test_parse_eml_missing_subject():
    raw = load("001-acme-alimentos-01")
    msg = email.message_from_bytes(raw, policy=policy.default)

    # Remove the Subject header to simulate a missing subject
    del msg["Subject"]

    result = parse_eml(msg.as_bytes())
    assert "Missing Subject header" in result.warnings


def test_parse_eml_missing_date():
    raw = load("001-acme-alimentos-01")
    msg = email.message_from_bytes(raw, policy=policy.default)

    # Remove the Date header to simulate a missing date
    del msg["Date"]

    result = parse_eml(msg.as_bytes())
    assert "Missing Date header" in result.warnings


def test_parse_eml_invalid_date():
    raw = load("001-acme-alimentos-01")
    msg = email.message_from_bytes(raw, policy=policy.default)

    # Set an invalid Date header to simulate an invalid date
    msg.replace_header("Date", "invalid-date")

    result = parse_eml(msg.as_bytes())
    assert "Invalid Date header" in result.warnings


def test_parse_email_missing_from():
    raw = load("001-acme-alimentos-01")
    msg = email.message_from_bytes(raw, policy=policy.default)

    # Remove the From header to simulate a missing sender
    del msg["From"]

    result = parse_eml(msg.as_bytes())
    assert "Missing or invalid From header" in result.warnings


def test_parse_eml_empty_body():
    raw = load("001-acme-alimentos-01")
    msg = email.message_from_bytes(raw, policy=policy.default)

    # Blank the body while keeping the text/plain part in place
    msg.set_content("")

    result = parse_eml(msg.as_bytes())
    assert "Empty body" in result.warnings
    assert result.body_text.strip() == ""


def test_parse_eml_no_body():
    raw = load("001-acme-alimentos-01")
    msg = email.message_from_bytes(raw, policy=policy.default)

    # Re-type the only part so the email has no plain/html body part
    msg.replace_header("Content-Type", "application/octet-stream")

    with pytest.raises(EmailParseError) as excinfo:
        parse_eml(msg.as_bytes())
    assert "No suitable body part found in the email" in str(excinfo.value)
