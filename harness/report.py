"""Render one eval case as a per-field report for eyeballing gaps."""

from corvid.contracts import Provenance, QuoteRequest
from harness.models import Case
from harness.score import Outcome

_MARKERS = {"correct": "✓", "wrong": "✗", "hallucinated": "⚠", "stale": "↺"}


def _value(request: QuoteRequest, path: str) -> object:
    part: object = request
    for attr in path.split("."):
        part = getattr(part, attr)
    return part


def render_case(
    case: Case,
    request: QuoteRequest,
    provenance: dict[str, Provenance],
    scores: dict[str, Outcome],
) -> str:
    """One line per field: marker, value, provenance source, expected on mismatch.

    Absent optional fields are skipped; absent required fields get a MISSING
    line so gaps stand out.
    """
    truths = {
        "origin.name": case.origin.name,
        "destination.name": case.destination.name,
    }
    required = set(QuoteRequest.required_dot_fields())
    width = max(len(path) for path in QuoteRequest.dot_fields())
    value_width = max(
        (len(str(v)) for p in QuoteRequest.dot_fields() if (v := _value(request, p))),
        default=0,
    )

    lines = [case.key]
    for path in QuoteRequest.dot_fields():
        value = _value(request, path)
        if value is None:
            if path in required:
                lines.append(f"  ∅ {path:<{width}}  — MISSING")
            continue
        marker = _MARKERS.get(scores.get(path, ""), "·")
        source = provenance[path].source if path in provenance else "?"
        line = f"  {marker} {path:<{width}}  {str(value):<{value_width}}  ({source})"
        if scores.get(path) == "wrong":
            line += f"  expected: {truths[path]}"
        elif scores.get(path) == "stale":
            line += f"  — superseded; expected: {truths[path]}"
        elif scores.get(path) == "hallucinated":
            line += f"  — email omitted this; expected empty or recalled {truths[path]}"
        lines.append(line)
    return "\n".join(lines)
