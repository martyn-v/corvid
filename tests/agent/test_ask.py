from corvid.agent.ask import apply_answers, questions_for
from corvid.contracts import Provenance, QuoteRequest


def _request_missing_origin() -> QuoteRequest:
    return QuoteRequest.model_validate(
        {
            "requester": {
                "name": "Marta Restrepo",
                "email": "marta@acme.example.com",
                "company": "ACME Alimentos SAS",
            },
            "destination": {"name": "Rotterdam"},
        }
    )


def test_questions_for_covers_exactly_the_missing_fields():
    questions = questions_for(_request_missing_origin())
    assert set(questions) == {"origin.name"}
    assert "?" in questions["origin.name"]


def test_questions_for_complete_request_is_empty():
    request = _request_missing_origin()
    request.origin.name = "Bogotá"
    assert questions_for(request) == {}


def test_apply_answers_fills_with_answered_question_provenance():
    request = _request_missing_origin()
    provenance: dict[str, Provenance] = {}
    questions = questions_for(request)

    exchanges = apply_answers(request, provenance, questions, {"origin.name": "Bogotá"})

    assert request.origin.name == "Bogotá"
    assert provenance["origin.name"].source == "answered_question"
    assert len(exchanges) == 1
    assert exchanges[0].path == "origin.name"
    assert exchanges[0].question == questions["origin.name"]
    assert exchanges[0].answer == "Bogotá"


def test_apply_answers_records_unanswered_questions_without_filling():
    request = _request_missing_origin()
    provenance: dict[str, Provenance] = {}
    questions = questions_for(request)

    exchanges = apply_answers(request, provenance, questions, {})

    assert request.origin.name is None
    assert provenance == {}
    assert exchanges[0].answer is None


def test_apply_answers_ignores_answers_to_unasked_questions():
    """A resume payload can only fill fields the agent actually asked about."""
    request = _request_missing_origin()
    provenance: dict[str, Provenance] = {}

    exchanges = apply_answers(
        request, provenance, {}, {"destination.name": "Shanghai"}
    )

    assert request.destination.name == "Rotterdam"
    assert exchanges == []
