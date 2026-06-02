import pytest

from app.detection_pairing import pair_by_capture_time


pytestmark = pytest.mark.algorithm


def det(name, ts):
    return {"id": name, "timestamp": ts}


def test_pair_by_capture_time_uses_nearest_unused_detection():
    first = [det("a", 10.00), det("b", 10.20)]
    second = [det("x", 10.19), det("y", 10.01)]

    result = pair_by_capture_time(first, second, match_window=0.05)

    assert [(a["id"], b["id"]) for a, b in result.pairs] == [
        ("a", "y"),
        ("b", "x"),
    ]
    assert result.remaining_first == []
    assert result.remaining_second == []


def test_pair_by_capture_time_keeps_unmatched_for_next_round():
    first = [det("a", 10.00), det("b", 20.00)]
    second = [det("x", 10.03)]

    result = pair_by_capture_time(first, second, match_window=0.05)

    assert [(a["id"], b["id"]) for a, b in result.pairs] == [("a", "x")]
    assert [d["id"] for d in result.remaining_first] == ["b"]
    assert result.remaining_second == []


def test_pair_by_capture_time_treats_exact_window_as_unmatched():
    result = pair_by_capture_time(
        [det("a", 10.00)],
        [det("x", 10.30)],
        match_window=0.30,
    )

    assert result.pairs == []
    assert [d["id"] for d in result.remaining_first] == ["a"]
    assert [d["id"] for d in result.remaining_second] == ["x"]


def test_pair_by_capture_time_prefers_capture_ts_over_timestamp():
    first = [{"id": "a", "timestamp": 1.0, "capture_ts": 10.0}]
    second = [{"id": "x", "timestamp": 10.0, "capture_ts": 1.0}]

    result = pair_by_capture_time(first, second, match_window=0.05)

    assert result.pairs == []


def test_pair_by_capture_time_caps_unmatched_queue_tail():
    first = [det(f"a{i}", float(i)) for i in range(40)]
    second = [det("x", 100.0)]

    result = pair_by_capture_time(
        first,
        second,
        match_window=0.05,
        max_queue_size=32,
        keep_tail_size=16,
    )

    assert result.pairs == []
    assert [d["id"] for d in result.remaining_first] == [f"a{i}" for i in range(24, 40)]
    assert [d["id"] for d in result.remaining_second] == ["x"]
