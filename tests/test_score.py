import datetime

from corvid.contracts import Provenance, QuoteRequest
from harness.models import Case, Location
from harness.score import score_case


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


def _request(origin_name: str | None, destination_name: str | None) -> QuoteRequest:
    return QuoteRequest.model_validate(
        {
            "requester": {},
            "origin": {"name": origin_name},
            "destination": {"name": destination_name},
        }
    )


def test_correct_when_extracted_city_matches_ground_truth():
    """The extracted city matches the case even though ground truth appends the country."""
    scores = score_case(_case(), _request("Bogotá", "Rotterdam"), {})
    assert scores == {"origin.name": "correct", "destination.name": "correct"}


def test_match_is_case_insensitive():
    """Casing differences between email and ground truth are not extraction errors."""
    scores = score_case(_case(), _request("BOGOTÁ", "rotterdam"), {})
    assert scores == {"origin.name": "correct", "destination.name": "correct"}


def test_wrong_when_origin_holds_the_customer_name():
    """The observed failure mode: the customer name lands in origin.name."""
    scores = score_case(_case(), _request("Acme Alimentos", "Rotterdam"), {})
    assert scores["origin.name"] == "wrong"


def test_missing_when_field_is_empty():
    """An empty field scores missing, not wrong — it is still recoverable by recall."""
    scores = score_case(_case(), _request(None, "Rotterdam"), {})
    assert scores["origin.name"] == "missing"


def test_hallucinated_when_omitted_origin_was_filled_from_email():
    """If the email omitted the origin, an email-sourced value is invented — even a lucky one."""
    provenance = {"origin.name": Provenance(source="email")}
    scores = score_case(
        _case(origin_omitted=True), _request("Bogotá", "Rotterdam"), provenance
    )
    assert scores["origin.name"] == "hallucinated"


def test_omitted_origin_filled_from_memory_scores_normally():
    """An omitted origin recalled from the graph is the success path, not a hallucination."""
    provenance = {"origin.name": Provenance(source="learned")}
    scores = score_case(
        _case(origin_omitted=True), _request("Bogotá", "Rotterdam"), provenance
    )
    assert scores["origin.name"] == "correct"
