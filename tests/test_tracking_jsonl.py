import json

from app.tracking_jsonl import TrackingJsonlWriter


def test_tracking_jsonl_writer_numbers_only_frame_rows(tmp_path):
    writer = TrackingJsonlWriter(tmp_path)
    path = writer.make_path(ts="20260504_120000", label="unit")

    writer.open(path, reset_counter=True)
    writer.append({"event": "start"}, bump_frame_counter=False)
    writer.append({"x": 1.0}, bump_frame_counter=True)
    writer.append({"x": 2.0}, bump_frame_counter=True)
    writer.close()

    rows = [json.loads(line) for line in open(path, encoding="utf-8")]
    assert rows == [
        {"event": "start"},
        {"frame": 0, "x": 1.0},
        {"frame": 1, "x": 2.0},
    ]
    assert writer.frame_count == 2


def test_tracking_jsonl_writer_rotates_to_unique_paths(tmp_path):
    writer = TrackingJsonlWriter(tmp_path)
    first = writer.make_path(ts="20260504_120000", label="bg")

    writer.open(first, reset_counter=True)
    writer.append({"x": 1}, bump_frame_counter=True)
    second = writer.make_path(ts="20260504_120000", label="bg")
    writer.rotate(second, reset_counter=True)
    writer.append({"x": 2}, bump_frame_counter=True)
    writer.close()

    assert first != second
    assert writer.path == second
    assert writer.frame_count == 1


def test_tracking_jsonl_latest_prefers_completed_recording(tmp_path):
    writer = TrackingJsonlWriter(tmp_path)
    bg = writer.make_path(ts="20260504_120000", label="bg")
    rec = writer.make_path(ts="20260504_120001", label="rec")

    writer.open(bg, reset_counter=True)
    writer.close()
    writer.open(rec, reset_counter=True)
    writer.close()
    writer.last_completed_path = rec

    assert writer.latest_path(recording=False) == rec
    assert writer.latest_path(recording=True) == rec
