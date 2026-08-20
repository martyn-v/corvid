"""LEARN: render the episodes memory ingests — the raw email, and any answers."""

from pathlib import Path

from corvid.agent.ask import Exchange
from corvid.agent.extract import render_email_prompt
from corvid.agent.parse_email import ParsedEmail


def _sender(email: ParsedEmail) -> str:
    if email.sender is None:
        return "The customer"
    if email.sender.display_name:
        return f"{email.sender.display_name} <{email.sender.address}>"
    return email.sender.address


def email_episode(file_path: str, email: ParsedEmail) -> tuple[str, str]:
    """(name, body) for the raw-email episode: named for the file, headers included."""
    return Path(file_path).stem, render_email_prompt(email)


def answers_episode(
    file_path: str, email: ParsedEmail, asked: list[Exchange]
) -> tuple[str, str] | None:
    """(name, body) for the answered-questions episode, or None if nothing was answered."""
    answered = [e for e in asked if e.answer is not None]
    if not answered:
        return None
    dialogue = "\n".join(f"Q: {e.question}\nA: {e.answer}" for e in answered)
    body = f"{_sender(email)} answered questions about their quote request:\n{dialogue}"
    return f"{Path(file_path).stem}-answers", body
