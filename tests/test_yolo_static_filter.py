from app.pipeline.inference import YoloRoadmapDetector


def make_detector():
    det = YoloRoadmapDetector.__new__(YoloRoadmapDetector)
    det.move_threshold = 5.0
    det.static_frame_limit = 4
    det.static_zone_radius = 20.0
    det.static_zone_ttl_frames = 8
    det.static_zone_max = 4
    det.static_release_speed = 8.0
    det.static_release_displacement = 18.0
    det.static_release_frames = 2
    det.static_starvation_frames = 90
    det._frame_counter = 0
    det._track_history = {}
    det._last_seen = {}
    det._static_zones = {}
    det._next_static_zone_id = 1
    det._next_pseudo_track_id = 1
    det._static_starvation_count = 0
    det._static_fail_open_until = 0
    det._static_stats = {
        "raw_detections": 0,
        "kept_detections": 0,
        "static_blocked": 0,
        "static_zones_created": 0,
        "static_zones_expired": 0,
        "motion_released": 0,
        "fail_open_kept": 0,
        "untracked_kept": 0,
        "pseudo_tracked": 0,
    }
    det._track_available = True
    return det


def update(det, track_id, x, y):
    det._frame_counter += 1
    state = det._update_track_state(track_id, float(x), float(y), 12.0, 12.0)
    return det._apply_static_gate(state, float(x), float(y))


def test_static_zone_blocks_then_expires_when_object_disappears():
    det = make_detector()

    blocked = []
    for _ in range(6):
        keep, status, zone = update(det, 7, 100, 100)
        blocked.append((keep, status, zone))

    assert det.get_runtime_stats()["active_static_zones"] == 1
    assert blocked[-1][0] is False
    assert blocked[-1][1] == "static_blocked"

    for _ in range(det.static_zone_ttl_frames + 1):
        det._frame_counter += 1
        det._expire_static_zones()

    assert det.get_runtime_stats()["active_static_zones"] == 0


def test_motion_through_static_zone_is_released_without_deleting_zone():
    det = make_detector()

    for _ in range(6):
        update(det, 7, 100, 100)

    assert det.get_runtime_stats()["active_static_zones"] == 1

    update(det, 9, 92, 100)
    update(det, 9, 106, 100)
    keep, status, zone = update(det, 9, 120, 100)

    assert keep is True
    assert status == "motion_released"
    assert zone is not None
    assert det.get_runtime_stats()["active_static_zones"] == 1
    assert det.get_runtime_stats()["motion_released"] >= 1


def test_pseudo_tracks_link_untracked_static_detections():
    det = make_detector()
    claimed = set()

    first = det._assign_pseudo_track(100.0, 100.0, claimed)
    det._update_track_state(first, 100.0, 100.0, 12.0, 12.0)

    det._frame_counter += 1
    claimed = set()
    second = det._assign_pseudo_track(102.0, 101.0, claimed)

    assert first == second
