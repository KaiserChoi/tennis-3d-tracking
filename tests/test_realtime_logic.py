import asyncio
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.analytics import RallyStateMachine
from app.config import load_config
from app.orchestrator import Orchestrator


@pytest.fixture
def orch():
    orch = Orchestrator(load_config("config.yaml"))
    try:
        yield orch
    finally:
        try:
            orch._manager.shutdown()
        except Exception:
            pass


def test_rally_state_machine_pending_start_used_for_serving():
    sm = RallyStateMachine(serve_confirm_frames=3)
    pts = [
        {"x": 0.0, "y": -8.0, "z": 1.2, "timestamp": 1.00, "frame_index": 100},
        {"x": 0.1, "y": -8.1, "z": 1.1, "timestamp": 1.04, "frame_index": 101},
        {"x": 0.2, "y": -8.2, "z": 1.0, "timestamp": 1.08, "frame_index": 102},
    ]

    for pt in pts:
        sm.update(pt)

    assert sm.get_state_dict()["state"] == "serving"
    assert sm._rally_start_time == pytest.approx(1.00)
    assert sm._rally_start_frame == 100


def test_rally_state_machine_pending_start_used_for_midflight_rally():
    sm = RallyStateMachine(serve_confirm_frames=3)
    pts = [
        {"x": 0.0, "y": -1.0, "z": 1.2, "timestamp": 2.00, "frame_index": 200},
        {"x": 0.1, "y": 0.8, "z": 1.1, "timestamp": 2.04, "frame_index": 201},
    ]

    for pt in pts:
        sm.update(pt)

    assert sm.get_state_dict()["state"] == "rally"
    assert sm._stroke_count == 1
    assert sm._rally_start_time == pytest.approx(2.00)
    assert sm._rally_start_frame == 200


def test_reset_live_analytics_clears_sg_buffer(orch):
    orch._sg_buffer.append(
        {
            "x": 1.0,
            "y": 2.0,
            "z": 0.5,
            "timestamp": 10.0,
            "capture_ts": 10.0,
            "frame_index": 1,
        }
    )

    orch.reset_live_analytics()

    assert orch._sg_buffer == []


def test_orchestrator_uses_bounce_detection_config():
    config = load_config("config.yaml")
    config.bounce_detection.hybrid.min_seg_len = 6
    config.bounce_detection.hybrid.min_dense = 5
    config.bounce_detection.hybrid.max_gap_s = 0.9
    config.bounce_detection.hybrid.z_max = 0.9
    config.bounce_detection.hybrid.min_speed = 2.0
    config.bounce_detection.hybrid.v_window = 6
    config.bounce_detection.hybrid.half_wins = [4, 6]
    config.bounce_detection.smoothing.max_frame_gap = 6
    config.bounce_detection.smoothing.max_gap_s = 0.9

    orch = Orchestrator(config)
    try:
        assert orch._hybrid_bounce._min_seg_len == 6
        assert orch._hybrid_bounce._min_dense == 5
        assert orch._hybrid_bounce._max_gap_s == pytest.approx(0.9)
        assert orch._hybrid_bounce._z_max == pytest.approx(0.9)
        assert orch._hybrid_bounce._min_speed == pytest.approx(2.0)
        assert orch._hybrid_bounce._v_window == 6
        assert orch._hybrid_bounce._half_wins == (4, 6)
        assert orch._sg_max_gap == 6
        assert orch._sg_max_gap_s == pytest.approx(0.9)
    finally:
        orch._manager.shutdown()


def test_live_bounce_history_keeps_true_total_after_rollover(orch):
    orch._LIVE_BOUNCE_HISTORY_LIMIT = 3

    for i in range(5):
        orch._record_live_bounce_locked({
            "timestamp": float(i),
            "x": float(i),
            "y": 0.0,
            "z": 0.0,
            "in_court": True,
            "frame_index": i,
        })

    analytics = orch.get_live_analytics()

    assert analytics["total_bounces"] == 5
    assert [b["frame_index"] for b in analytics["recent_bounces"]] == [2, 3, 4]
    assert [b["sequence"] for b in analytics["recent_bounces"]] == [3, 4, 5]


def _accepted_yolo_bounce(frame=42, x=1.25, y=-3.5):
    return {
        "timestamp": 1000.0,
        "capture_ts": 1000.0,
        "x": x,
        "y": y,
        "z": 0.0,
        "in_court": True,
        "frame": frame,
        "frame_index": frame,
        "camera": "cam68",
        "camera_name": "cam68",
        "bounce_mode": "mono_cam68",
        "source": "yolo_fuzzy_single_cam",
        "speed_kmh": 57,
    }


def test_3d_push_enabled_queues_yolo_bounce_from_record_fanout(orch):
    orch._bounce_mode = "mono_cam68"
    orch._ws_enabled = True

    orch._record_live_bounce_locked(_accepted_yolo_bounce())

    assert orch._live_bounces[-1]["sequence"] == orch._ws_bounce_queue[-1]["sequence"]
    assert orch._ws_bounce_queue[-1]["source"] == "yolo_fuzzy_single_cam"
    assert orch._ws_bounce_queue[-1]["bounce_mode"] == "mono_cam68"
    assert orch._ws_bounce_queue[-1]["camera_name"] == "cam68"
    assert orch._ws_bounce_queue[-1]["frame_index"] == 42
    analytics = orch.get_live_analytics()
    assert analytics["ws_pending_bounces"] == 1
    assert analytics["ws_last_queued_sequence"] == orch._live_bounces[-1]["sequence"]
    assert analytics["ws_last_sent_sequence"] is None
    assert analytics["ws_sent_count"] == 0
    assert analytics["ws_error_count"] == 0
    assert analytics["ws_last_error"] is None


def test_dashboard_status_limits_events_without_trimming_history(orch):
    for i in range(60):
        orch._record_live_bounce_locked(_accepted_yolo_bounce(frame=i))
        orch._record_live_hit_locked({"timestamp": float(i), "x": 0.0, "y": 0.0})
        orch._record_live_speed_event_locked({"timestamp": float(i), "speed_kmh": 80})

    status = orch.get_dashboard_status(event_limit=40)
    analytics = status["analytics"]
    encoded = json.dumps(status, default=str).encode("utf-8")

    assert len(analytics["recent_bounces"]) == 40
    assert analytics["recent_bounces"][0]["frame_index"] == 20
    assert analytics["total_bounces"] == 60
    assert analytics["recent_hits"] == []
    assert analytics["recent_speed_events"] == []
    assert analytics["completed_rallies"] == []
    assert len(orch._live_bounces) == 60
    assert "latest_detections" not in status
    assert len(encoded) < 50 * 1024


def test_3d_push_disabled_keeps_yolo_bounce_out_of_ws_queue(orch):
    orch._bounce_mode = "mono_cam68"
    orch._ws_enabled = False

    orch._record_live_bounce_locked(_accepted_yolo_bounce())

    assert len(orch._live_bounces) == 1
    assert len(orch._ws_bounce_queue) == 0


def test_yolo_accepted_bounce_uses_record_fanout_for_ws_queue(orch, monkeypatch):
    def fake_detect_single_camera_events(*_args, **_kwargs):
        return {
            "bounces": [
                {
                    "frame_index": 10,
                    "x": 1.25,
                    "y": -3.5,
                    "pixel_x": 960,
                    "pixel_y": 540,
                    "in_court": True,
                    "source": "yolo_fuzzy_single_cam",
                }
            ],
            "hits": [],
            "speed_events": [],
        }

    import app.pipeline.yolo_bounce_filter as yolo_bounce_filter

    monkeypatch.setattr(
        yolo_bounce_filter,
        "detect_single_camera_events",
        fake_detect_single_camera_events,
    )
    monkeypatch.setattr(orch, "_event_homography_for_camera", lambda _cam: None)
    orch.config.model.detector_type = "yolo_roadmap"
    orch._bounce_mode = "mono_cam68"
    orch._ws_enabled = True

    det = {
        "frame_index": 20,
        "timestamp": 1000.0,
        "capture_ts": 1000.0,
        "pixel_x": 960,
        "pixel_y": 540,
        "x": 1.25,
        "y": -3.5,
    }
    with orch._analytics_lock:
        emitted = orch._run_yolo_fuzzy_single_cam_locked("cam68", det)

    assert emitted is not None
    assert orch._live_bounces[-1]["source"] == "yolo_fuzzy_single_cam"
    assert orch._live_bounces[-1]["sequence"] == orch._ws_bounce_queue[-1]["sequence"]
    assert orch._ws_bounce_queue[-1]["frame_index"] == 10


def test_enable_3d_push_does_not_replay_existing_live_bounces(orch, monkeypatch):
    monkeypatch.setattr(orch, "_ws_push_loop", lambda: None)
    orch._record_live_bounce_locked(_accepted_yolo_bounce(frame=10))
    assert len(orch._live_bounces) == 1
    assert len(orch._ws_bounce_queue) == 0

    orch.enable_3d_display()

    assert len(orch._live_bounces) == 1
    assert len(orch._ws_bounce_queue) == 0

    orch._record_live_bounce_locked(_accepted_yolo_bounce(frame=11))

    assert len(orch._live_bounces) == 2
    assert len(orch._ws_bounce_queue) == 1
    assert orch._ws_bounce_queue[0]["frame_index"] == 11


def test_3d_push_queue_uses_remote_units_with_raw_coordinates(orch):
    orch._ws_enabled = True

    orch._record_live_bounce_locked(_accepted_yolo_bounce(x=1.25, y=-3.5))
    queued = orch._ws_bounce_queue[-1]

    assert queued["raw_x"] == pytest.approx(1.25)
    assert queued["raw_y"] == pytest.approx(-3.5)
    assert queued["x"] == pytest.approx(queued["raw_x"] * 10)
    assert queued["y"] == pytest.approx(queued["raw_y"] * 10)


def test_3d_push_send_success_pops_queue_and_records_sequence(orch):
    class FakeWebSocket:
        def __init__(self):
            self.messages = []

        async def send(self, message):
            self.messages.append(json.loads(message))

    orch._ws_enabled = True
    orch._record_live_bounce_locked(_accepted_yolo_bounce(frame=88))
    sequence = orch._ws_bounce_queue[0]["sequence"]
    ws = FakeWebSocket()

    sent = asyncio.run(orch._send_ws_bounce_once(ws))

    assert sent is True
    assert len(orch._ws_bounce_queue) == 0
    assert orch._ws_last_sent_sequence == sequence
    assert orch._ws_sent_count == 1
    assert ws.messages[0]["msg"]["data"]["bounce"]["x"] == pytest.approx(12.5)


def test_3d_push_send_failure_keeps_queue_for_retry(orch):
    class FailingWebSocket:
        async def send(self, _message):
            raise ConnectionError("disconnected")

    class WorkingWebSocket:
        def __init__(self):
            self.messages = []

        async def send(self, message):
            self.messages.append(json.loads(message))

    orch._ws_enabled = True
    orch._record_live_bounce_locked(_accepted_yolo_bounce(frame=89))
    sequence = orch._ws_bounce_queue[0]["sequence"]

    with pytest.raises(ConnectionError):
        asyncio.run(orch._send_ws_bounce_once(FailingWebSocket()))

    assert len(orch._ws_bounce_queue) == 1
    assert orch._ws_bounce_queue[0]["sequence"] == sequence
    assert orch._ws_last_sent_sequence is None

    ws = WorkingWebSocket()
    assert asyncio.run(orch._send_ws_bounce_once(ws)) is True
    assert len(orch._ws_bounce_queue) == 0
    assert orch._ws_last_sent_sequence == sequence
    assert ws.messages[0]["msg"]["data"]["bounce"]["x"] == pytest.approx(12.5)


def test_post_filter_f2_allows_quick_but_distant_bounce(orch):
    orch._live_bounces = [
        {"timestamp": 10.0, "x": -3.0, "y": -8.0, "side": "near", "in_court": True}
    ]

    ok, reason = orch._post_filter_bounce(
        {"timestamp": 10.2, "x": 3.0, "y": -8.0, "side": "near", "in_court": True}
    )

    assert ok is True
    assert reason == "accepted"


def test_post_filter_f2_rejects_quick_nearby_repeat(orch):
    orch._live_bounces = [
        {"timestamp": 10.0, "x": -3.0, "y": -8.0, "side": "near", "in_court": True}
    ]

    ok, reason = orch._post_filter_bounce(
        {"timestamp": 10.2, "x": -2.2, "y": -8.3, "side": "near", "in_court": True}
    )

    assert ok is False
    assert reason == "f2_min_interval"


def test_live_detectors_respect_bounce_toggle_and_reset_buffers(orch, monkeypatch):
    pt = {
        "x": 0.0,
        "y": -4.0,
        "z": 1.0,
        "timestamp": 20.0,
        "capture_ts": 20.0,
        "frame_index": 10,
    }
    calls = {"peak": 0, "hybrid": 0}

    def fake_peak_update(_point):
        calls["peak"] += 1
        return None

    def fake_pop_pending():
        return []

    def fake_smooth(point, cam_dets):
        return point, cam_dets

    def fake_hybrid_update(_point, _cam_dets):
        calls["hybrid"] += 1
        return None

    monkeypatch.setattr(orch._bounce_detector, "update", fake_peak_update)
    monkeypatch.setattr(orch._bounce_detector, "pop_pending", fake_pop_pending)
    monkeypatch.setattr(orch, "_smooth_latest", fake_smooth)
    monkeypatch.setattr(orch._hybrid_bounce, "update", fake_hybrid_update)

    orch._sg_buffer.append({"x": 9.0, "y": 9.0, "z": 9.0, "timestamp": 9.0})
    orch.set_bounce_detection_enabled(False)
    assert orch._sg_buffer == []

    with orch._analytics_lock:
        smoothed_pt, hbounce = orch._run_live_bounce_detectors_locked(pt, {})
    assert smoothed_pt == pt
    assert hbounce is None
    assert calls == {"peak": 0, "hybrid": 0}

    orch.set_bounce_detection_enabled(True)
    with orch._analytics_lock:
        smoothed_pt, hbounce = orch._run_live_bounce_detectors_locked(pt, {})
    assert smoothed_pt == pt
    assert hbounce is None
    assert calls == {"peak": 1, "hybrid": 1}
