"""Score an eval run's extracted quote request against a case's ground truth."""

from typing import Literal

from corvid.contracts import Provenance, QuoteRequest
from harness.models import Case, Persona
from harness.text import city as _city

Outcome = Literal["correct", "wrong", "missing", "hallucinated", "stale"]


def score_case(
    case: Case,
    request: QuoteRequest,
    provenance: dict[str, Provenance],
    superseded: dict[str, str] | None = None,
) -> dict[str, Outcome]:
    """Scores each location field the case has ground truth for.

    A field the email omitted but an email-sourced value filled anyway is
    hallucinated even when the value happens to match — there was nothing
    in the email to extract it from.

    `superseded` maps a path to the value that was true before a scripted
    change; a wrong fill matching it scores stale — the failure is recalling
    outdated truth, not inventing a value.
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
        elif superseded and path in superseded and _city(part) == _city(
            superseded[path]
        ):
            scores[path] = "stale"
        else:
            scores[path] = "wrong"
    return scores


def superseded_for(case: Case, persona: Persona) -> dict[str, str]:
    """Returns path → previously-true value for cases after a scripted change.

    Empty when the persona has no change or the case predates it; the
    persona's own origin is the pre-change truth (generate.py overrides it
    from the change email onward).
    """
    if persona.change is None or case.index < persona.change.at_email:
        return {}
    return {"origin.name": persona.origin.name}
