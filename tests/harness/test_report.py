import datetime

from corvid.contracts import Provenance, QuoteRequest
from harness.models import Case, Location
from harness.report import render_case


def _case(origin_omitted: bool = False) -> Case:
    return Case(
        index=1,
        persona="acme-alimentos",
        date=datetime.date(2023, 1, 6),
        origin=Location(locode="COBOG", name="Bogotá, Colombia"),
        destination=Location(locode="NLRTM", name="Rotterdam, Netherlands"),
        mode="ocean_fcl",
        commodity="packaged non-perishable foods",
        pieces=14,
        weight_kg=9385,
        origin_omitted=origin_omitted,
        n=1,
    )


def _request() -> QuoteRequest:
    return QuoteRequest.model_validate(
        {
            "requester": {"name": "John Doe", "company": "Acme Alimentos"},
            "origin": {"name": "Bogotá"},
            "destination": {"name": "Rotterdam"},
        }
    )


def _line(report: str, path: str) -> str:
    """The report line for a dot path; fails the test if absent."""
    matches = [line for line in report.splitlines() if f" {path} " in f"{line} "]
    assert matches, f"no line for {path} in:\n{report}"
    return matches[0]


def test_header_is_the_case_key():
    report = render_case(_case(), _request(), {}, {})
    assert report.splitlines()[0] == "001-acme-alimentos-01"


def test_present_field_shows_value_and_provenance_source():
    provenance = {"origin.name": Provenance(source="email")}
    report = render_case(_case(), _request(), provenance, {})
    line = _line(report, "origin.name")
    assert "Bogotá" in line
    assert "email" in line


def test_correct_score_marks_the_line():
    report = render_case(_case(), _request(), {}, {"origin.name": "correct"})
    assert _line(report, "origin.name").lstrip().startswith("✓")


def test_wrong_score_shows_expected_value():
    request = _request()
    request.origin.name = "Acme Alimentos"
    report = render_case(_case(), request, {}, {"origin.name": "wrong"})
    line = _line(report, "origin.name")
    assert line.lstrip().startswith("✗")
    assert "Bogotá, Colombia" in line


def test_hallucinated_score_says_email_omitted_it():
    report = render_case(
        _case(origin_omitted=True), _request(), {}, {"origin.name": "hallucinated"}
    )
    line = _line(report, "origin.name")
    assert line.lstrip().startswith("⚠")
    assert "omitted" in line


def test_missing_required_field_is_called_out():
    report = render_case(_case(), _request(), {}, {})
    assert "MISSING" in _line(report, "requester.email")


def test_absent_optional_field_has_no_line():
    report = render_case(_case(), _request(), {}, {})
    assert "origin.country" not in report
