from corvid.memory.ontology import Contact, Location


def test_contact_carries_an_email_attribute():
    """The sender address anchors identity, so extraction must have a slot
    for it — same pattern as Location.locode."""
    field = Contact.model_fields["email"]
    assert field.default is None  # optional: not every mention has an address
    assert "email" in (field.description or "").lower()


def test_location_still_carries_its_locode():
    field = Location.model_fields["locode"]
    assert field.default is None
