"""Score an eval run's extracted quote request against a case's ground truth."""

import unicodedata
from typing import Literal

from corvid.contracts import Provenance, QuoteRequest
from harness.models import Case

Outcome = Literal["correct", "wrong", "missing", "hallucinated"]


def _city(name: str) -> str:
    """The comparable city part; ground truth names read "Bogotá, Colombia".

    Diacritics are stripped: extraction models transliterate ("Bogota"),
    and that is not an extraction error.
    """
    city = name.split(",")[0].strip().casefold()
    return "".join(
        c for c in unicodedata.normalize("NFKD", city) if not unicodedata.combining(c)
    )


def score_case(
    case: Case, request: QuoteRequest, provenance: dict[str, Provenance]
) -> dict[str, Outcome]:
    """Scores each location field the case has ground truth for.

    A field the email omitted but an email-sourced value filled anyway is
    hallucinated even when the value happens to match — there was nothing
    in the email to extract it from.
    """
    expected = {
        "origin.name": case.origin.name,
        "destination.name": case.destination.name,
    }
    omitted = {"origin.name"} if case.origin_omitted else set()

    scores: dict[str, Outcome] = {}
    for path, truth in expected.items():
        part = request
        for attr in path.split("."):
            part = getattr(part, attr)
        prov = provenance.get(path)
        if path in omitted and prov is not None and prov.source == "email":
            scores[path] = "hallucinated"
        elif part is None:
            scores[path] = "missing"
        elif _city(part) == _city(truth):
            scores[path] = "correct"
        else:
            scores[path] = "wrong"
    return scores
