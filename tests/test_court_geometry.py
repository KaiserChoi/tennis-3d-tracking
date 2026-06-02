import pytest

from app.court import DEFAULT_COURT, CourtGeometry


pytestmark = pytest.mark.algorithm


def test_default_court_matches_project_coordinate_system():
    court = DEFAULT_COURT

    assert court.x_min == pytest.approx(-4.115)
    assert court.x_max == pytest.approx(4.115)
    assert court.y_min == pytest.approx(-11.885)
    assert court.y_max == pytest.approx(11.885)
    assert court.net_y == pytest.approx(0.0)


def test_contains_respects_configurable_margin():
    court = CourtGeometry(court_margin=0.15)

    assert court.contains(4.115, 11.885)
    assert court.contains(4.20, 11.90)
    assert not court.contains(4.30, 11.90)
    assert not court.contains(4.20, 12.10)

    assert not court.contains(4.20, 11.90, margin=0.0)


def test_side_for_y_splits_at_net():
    court = DEFAULT_COURT

    assert court.side_for_y(-0.001) == "near"
    assert court.side_for_y(0.0) == "far"
    assert court.side_for_y(0.001) == "far"
