from ml.severity import estimate_severity


def test_no_damage_has_no_severity() -> None:
    assert estimate_severity("no_damage", 0.99) == "none"


def test_high_confidence_pothole_is_high_severity() -> None:
    assert estimate_severity("pothole", 0.71) == "high"


def test_medium_confidence_damage_is_medium_severity() -> None:
    assert estimate_severity("longitudinal_crack", 0.65) == "medium"
