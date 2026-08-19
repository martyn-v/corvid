from datetime import datetime
from pydantic import BaseModel
import email
from email import policy
from email.message import EmailMessage
from email.utils import parsedate_to_datetime


class EmailParseError(Exception):
    """Raised when there is an error parsing the .eml file."""


class EmailAddress(BaseModel):
    display_name: str | None
    address: str  # the actual email address, parsed


class ParsedEmail(BaseModel):
    raw: bytes
    subject: str | None = None
    sender: EmailAddress | None = None  # display name + address, parsed
    date: datetime | None = None
    body_text: str  # the cleaned, extraction-ready text
    warnings: list[str]  # everything weird, surfaced not swallowed


def _parse_sender(msg: EmailMessage, warnings: list[str]) -> EmailAddress | None:
    sender = msg.get("from")
    if sender is None or sender.addresses is None or len(sender.addresses) == 0:
        warnings.append("Missing or invalid From header")
        return None
    return EmailAddress(
        display_name=sender.addresses[0].display_name,
        address=sender.addresses[0].addr_spec,
    )


def _parse_subject(msg: EmailMessage, warnings: list[str]) -> str | None:
    subject = msg.get("subject")
    if subject is None:
        warnings.append("Missing Subject header")
    return subject


def _parse_date(msg: EmailMessage, warnings: list[str]) -> datetime | None:
    date = msg.get("date")
    if date is None:
        warnings.append("Missing Date header")
        return None
    try:
        return parsedate_to_datetime(date)
    except (ValueError, TypeError):
        warnings.append("Invalid Date header")
        return None


def _extract_body_text(msg: EmailMessage, warnings: list[str]) -> str:
    body = msg.get_body(preferencelist=("plain"))
    if not body:
        raise EmailParseError("No suitable body part found in the email")

    body_text = body.get_content().strip()

    if not body_text:
        warnings.append("Empty body")
    return body_text


def parse_eml(raw: bytes) -> ParsedEmail:
    """Parse a raw .eml file into a ParsedEmail.

    Args:
        raw: The raw bytes of the .eml file.

    Returns:
        A ParsedEmail object containing the parsed email data.
    """
    msg = email.message_from_bytes(raw, policy=policy.default)

    warnings: list[str] = []
    sender = _parse_sender(msg, warnings)
    subject = _parse_subject(msg, warnings)
    date = _parse_date(msg, warnings)
    body_text = _extract_body_text(msg, warnings)

    return ParsedEmail(
        raw=raw,
        subject=subject,
        sender=sender,
        date=date,
        body_text=body_text,
        warnings=warnings,
    )
