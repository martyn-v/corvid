from harness.renderer import destination_leaked_as_origin

DESTINATION = "Rotterdam, Netherlands"


def test_catches_from_your_location_in_destination():
    """The observed misrender: the destination presented as the departure point."""
    body = (
        "We would like a quote for shipping our foods by ocean full container "
        "from your location in Rotterdam, Netherlands (NLRTM)."
    )
    assert destination_leaked_as_origin(body, DESTINATION)


def test_catches_bare_from_destination():
    assert destination_leaked_as_origin("shipping from Rotterdam", DESTINATION)


def test_allows_to_destination():
    assert not destination_leaked_as_origin(
        "a quote for shipping to Rotterdam", DESTINATION
    )


def test_allows_correct_lane_from_origin_to_destination():
    assert not destination_leaked_as_origin(
        "shipping from Bogotá to Rotterdam", DESTINATION
    )


def test_catches_spanish_desde_destination():
    assert destination_leaked_as_origin("envío desde Rotterdam", DESTINATION)


def test_allows_spanish_lane():
    assert not destination_leaked_as_origin(
        "envío desde Bogotá a Rotterdam", DESTINATION
    )


def test_is_case_insensitive():
    assert destination_leaked_as_origin("FROM ROTTERDAM", DESTINATION)


def test_allows_lane_phrased_as_destined_for():
    """Real render from the harness: a correct lane using 'destined for'."""
    body = (
        "The shipment originates from Gothenburg, Sweden and is destined "
        "for Bogotá, Colombia."
    )
    assert not destination_leaked_as_origin(body, "Bogotá, Colombia")
