"""ASK: questions to the customer for fields neither stated nor known."""

from pydantic import BaseModel

from corvid.contracts import Provenance, QuoteRequest, apply_fill

ASK_QUESTIONS = {
    "requester.name": "Could you share your name?",
    "requester.email": "What is the best email address to reach you?",
    "requester.company": "Which company is this quote for?",
    "origin.name": "Where will the shipment depart from?",
    "destination.name": "Where should the shipment be delivered?",
}


class Exchange(BaseModel):
    """One question asked of the customer, and the answer if one came back."""

    path: str
    question: str
    answer: str | None = None


def questions_for(request: QuoteRequest) -> dict[str, str]:
    """One question per missing required field, keyed by dot path."""
    return {
        path: ASK_QUESTIONS.get(path, f"Could you provide the {path}?")
        for path in request.missing()
    }


def apply_answers(
    request: QuoteRequest,
    provenance: dict[str, Provenance],
    questions: dict[str, str],
    answers: dict[str, str | None],
) -> list[Exchange]:
    """Fills answered fields with answered_question provenance, in place.

    Only fields that were actually asked about can be filled; unanswered
    questions are recorded with answer=None and leave their field missing.
    """
    exchanges = []
    for path, question in questions.items():
        answer = answers.get(path)
        if answer is not None:
            apply_fill(
                request, provenance, path, answer, Provenance(source="answered_question")
            )
        exchanges.append(Exchange(path=path, question=question, answer=answer))
    return exchanges
