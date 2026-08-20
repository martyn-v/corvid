import pytest
from corvid.contracts import Provenance, QuoteRequest, apply_fill, present_fields


def test_dot_fields():
    """Test the dot_fields method of the QuoteRequest class."""
    assert QuoteRequest.dot_fields() == [
        "requester.name",
        "requester.email",
        "requester.company",
        "origin.name",
        "destination.name",
    ]


def test_dot_fields_covers_missing():
    """Every path missing() can report must be a known dot field."""
    empty = QuoteRequest.model_validate(
        {"requester": {}, "origin": {}, "destination": {}}
    )
    assert set(empty.missing()) <= set(QuoteRequest.dot_fields())


def _empty_request() -> QuoteRequest:
    return QuoteRequest.model_validate({"requester": {}, "origin": {}, "destination": {}})


def test_apply_fill_sets_field_and_records_provenance():
    """apply_fill() sets the value at the dot path and records provenance in one move."""
    request = _empty_request()
    prov: dict[str, Provenance] = {}

    apply_fill(request, prov, "origin.name", "Cartagena", Provenance(source="learned"))

    assert request.origin.name == "Cartagena"
    assert prov["origin.name"].source == "learned"
    assert "origin.name" not in request.missing()


def test_apply_fill_rejects_unknown_path():
    """apply_fill() refuses paths that are not fields of the request."""
    with pytest.raises(KeyError):
        apply_fill(
            _empty_request(), {}, "origin.typo", "x", Provenance(source="learned")
        )


def test_apply_fill_rejects_overwrite():
    """apply_fill() never silently overwrites a field that already has provenance."""
    request = _empty_request()
    prov = {"origin.name": Provenance(source="email")}

    with pytest.raises(ValueError):
        apply_fill(request, prov, "origin.name", "Bogota", Provenance(source="learned"))


def test_present_fields_only_counts_actual_values():
    """present_fields() reports non-None leaves, ignoring optionality."""
    request = _empty_request()
    assert present_fields(request) == set()

    request.requester.name = "John Doe"
    request.origin.name = "Cartagena"

    assert present_fields(request) == {"requester.name", "origin.name"}
