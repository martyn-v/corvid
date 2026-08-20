import datetime

from corvid.contracts import Provenance, QuoteRequest
from harness.models import Case, Contact, EmailStyle, Location, Persona, PersonaChange
from harness.score import score_case, superseded_for


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


def test_match_ignores_diacritics():
    """Local models transliterate accents away; "Bogota" still means Bogotá."""
    scores = score_case(_case(), _request("Bogota", "Rotterdam"), {})
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


def test_stale_when_wrong_fill_matches_the_superseded_value():
    """A fill using the pre-change origin after the change is stale, not plain wrong."""
    case = _case()
    case.origin = Location(locode="COCTG", name="Cartagena, Colombia")
    scores = score_case(
        case,
        _request("Bogotá", "Rotterdam"),
        {"origin.name": Provenance(source="learned")},
        superseded={"origin.name": "Bogotá, Colombia"},
    )
    assert scores["origin.name"] == "stale"


def test_wrong_when_fill_matches_neither_truth_nor_superseded():
    """The stale check only renames failures that match the old truth."""
    case = _case()
    case.origin = Location(locode="COCTG", name="Cartagena, Colombia")
    scores = score_case(
        case,
        _request("Medellín", "Rotterdam"),
        {},
        superseded={"origin.name": "Bogotá, Colombia"},
    )
    assert scores["origin.name"] == "wrong"


def test_correct_fill_stays_correct_when_a_superseded_value_exists():
    """Recalling the post-change origin is the success path; superseded is irrelevant."""
    case = _case()
    case.origin = Location(locode="COCTG", name="Cartagena, Colombia")
    scores = score_case(
        case,
        _request("Cartagena", "Rotterdam"),
        {},
        superseded={"origin.name": "Bogotá, Colombia"},
    )
    assert scores["origin.name"] == "correct"


def test_omitted_origin_filled_from_memory_scores_normally():
    """An omitted origin recalled from the graph is the success path, not a hallucination."""
    provenance = {"origin.name": Provenance(source="learned")}
    scores = score_case(
        _case(origin_omitted=True), _request("Bogotá", "Rotterdam"), provenance
    )
    assert scores["origin.name"] == "correct"


def _persona(change: PersonaChange | None) -> Persona:
    return Persona(
        id="acme-alimentos",
        company="Acme Alimentos",
        contact=Contact(name="Ana Ruiz", email="ana@acme.example"),
        origin=Location(locode="COBOG", name="Bogotá, Colombia"),
        destination=Location(locode="NLRTM", name="Rotterdam, Netherlands"),
        mode="ocean_fcl",
        commodity="packaged non-perishable foods",
        omit_origin=True,
        change=change,
        style=EmailStyle(language="es", tone="formal"),
    )


def _change() -> PersonaChange:
    return PersonaChange(
        at_email=4,
        origin=Location(locode="COCTG", name="Cartagena, Colombia"),
        reason_in_email="new plant",
    )


def test_superseded_empty_for_persona_without_change():
    assert superseded_for(_case(), _persona(change=None)) == {}


def test_superseded_empty_before_the_change_email():
    case = _case()
    case.index = 3
    assert superseded_for(case, _persona(change=_change())) == {}


def test_superseded_holds_old_origin_from_the_change_email_onward():
    case = _case()
    case.index = 4
    assert superseded_for(case, _persona(change=_change())) == {
        "origin.name": "Bogotá, Colombia"
    }
