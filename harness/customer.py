"""The simulated customer: answers the agent's questions from ground truth."""

from harness.models import Case, Persona


def _city(name: str) -> str:
    """Ground truth names read "Bogotá, Colombia"; the customer says the city."""
    return name.split(",")[0].strip()


def answer_question(case: Case, persona: Persona, path: str) -> str | None:
    """The ground-truth answer for a field, or None for fields the harness can't answer.

    Locations come from the case, not the persona, so answers follow
    scripted persona changes at the right episode.
    """
    answers = {
        "origin.name": _city(case.origin.name),
        "destination.name": _city(case.destination.name),
        "requester.name": persona.contact.name,
        "requester.email": persona.contact.email,
        "requester.company": persona.company,
    }
    return answers.get(path)
