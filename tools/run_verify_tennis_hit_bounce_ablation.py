"""Run verify_tennis.py hit/bounce chain ablations for cam68.

This runner keeps yolo_roadmap.verify_tennis as the module under test.  It
wraps the original CourtCalibrator, QueueTracker, TrajectoryStitcher, and
TrajectoryAnalyzer with deterministic inputs so the hit/bounce separation can
be reviewed without the interactive calibration UI.

Outputs are written to:
  reports/verify_tennis_hit_bounce_ablation_<timestamp>/

Each HB variant gets one MP4 plus one JSON, and ablation_summary.json contains
the compact comparison table.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.pipeline.yolo_bounce_filter import COURT_Y_MAX, COURT_Y_MIN, SINGLES_X_MAX, SINGLES_X_MIN
from yolo_roadmap.verify_tennis import Config, CourtCalibrator, QueueTracker, TrajectoryAnalyzer, TrajectoryStitcher


DEFAULT_FRAMES_DIR = Path(r"D:\tennis-dataset\1001\clip11\cam68_20260404_075325_2min")
DEFAULT_BALL_MODEL = Path(r"D:\tennis\tennis-3d-tracking\yolo_roadmap\best.pt")
DEFAULT_PERSON_MODEL = Path(r"D:\tennis\tennis-3d-tracking\yolo11n.pt")
DEFAULT_HOMOGRAPHY = Path(r"D:\tennis\tennis-3d-tracking\src\homography_matrices.json")
DEFAULT_OUT_ROOT = Path(r"D:\tennis\tennis-3d-tracking\reports")

CAMERA = "cam68"
MINIMAP_FLIP = "rotate180"
MINIMAP_SHOW_HITS = True
COURT_X_MIN = -10.97 / 2.0
COURT_X_MAX = 10.97 / 2.0
HIT_SUPPRESS_RADIUS_FRAMES = 3


@dataclass(frozen=True)
class VariantSpec:
    variant: str
    video_name: str
    description: str
    stitching: bool = True
    static_zone: bool = True
    top_crossing_only: bool = False
    enable_top_hit: bool = True
    top_distance_gate: bool = True
    enable_bottom_hit: bool = True
    bottom_distance_mode: str = "dynamic"  # "dynamic" or "fixed_base"
    suppress_hit_window: bool = True
    reuse_variant: str | None = None


@dataclass
class DetectionPass:
    image_paths: list[Path]
    ball_by_frame: dict[int, list[tuple[float, float, float, float, float, float]]] = field(default_factory=dict)
    ball_boxes_by_frame: dict[int, list[dict[str, Any]]] = field(default_factory=dict)
    player_boxes_by_frame: dict[int, list[dict[str, Any]]] = field(default_factory=dict)
    player_history: dict[int, list[dict[str, Any]]] = field(default_factory=dict)
    raw_ball_boxes: int = 0
    raw_person_boxes: int = 0


@dataclass
class VariantResult:
    spec: VariantSpec
    per_frame: dict[int, dict[str, Any]]
    final_bounces: dict[int, dict[str, Any]]
    raw_bounce_candidates: dict[int, dict[str, Any]]
    hits: dict[int, dict[str, Any]]
    crossings: list[dict[str, Any]]
    top_hit_candidates: list[dict[str, Any]]
    bottom_hit_candidates: list[dict[str, Any]]
    suppressed_bounces: list[dict[str, Any]]
    stats: dict[str, Any]
    render: dict[str, Any] | None = None
    json_path: str | None = None


class NoStaticZoneQueueTracker(QueueTracker):
    """QueueTracker variant that disables static-zone absorption only."""

    def process_frame(self, frame_idx, detections):  # type: ignore[override]
        saved_radius = self.static_lock_radius
        saved_persistence = self.zone_persistence
        try:
            self.static_lock_radius = -1.0
            self.zone_persistence = 0
            super().process_frame(frame_idx, detections)
            self.static_zones.clear()
        finally:
            self.static_lock_radius = saved_radius
            self.zone_persistence = saved_persistence


def _safe_frame_index(path: Path) -> int:
    try:
        return int(path.stem)
    except ValueError:
        return -1


def _load_image_paths(frames_dir: Path, max_frames: int | None) -> list[Path]:
    image_paths = sorted(frames_dir.glob("*.jpg"), key=_safe_frame_index)
    if max_frames is not None:
        image_paths = image_paths[:max_frames]
    if not image_paths:
        raise RuntimeError(f"No JPG frames found in {frames_dir}")
    return image_paths


def _extract_video_frames(video_path: Path, frames_dir: Path, max_frames: int | None) -> tuple[list[Path], float]:
    frames_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
    image_paths: list[Path] = []
    frame_index = 0
    try:
        while True:
            if max_frames is not None and frame_index >= max_frames:
                break
            ok, frame = cap.read()
            if not ok:
                break
            out_path = frames_dir / f"{frame_index:05d}.jpg"
            if not cv2.imwrite(str(out_path), frame):
                raise RuntimeError(f"Cannot write extracted frame: {out_path}")
            image_paths.append(out_path)
            frame_index += 1
    finally:
        cap.release()
    if not image_paths:
        raise RuntimeError(f"No frames extracted from {video_path}")
    return image_paths, fps


def _mask_for_yolo(frame: np.ndarray) -> np.ndarray:
    frame_for_infer = frame.copy()
    frame_for_infer[0:41, 0:603] = 0
    return frame_for_infer


def _load_homography_matrices(path: Path, camera: str) -> tuple[np.ndarray, np.ndarray]:
    data = json.loads(path.read_text(encoding="utf-8"))
    cam = data[camera]
    return (
        np.array(cam["H_image_to_world"], dtype=np.float32),
        np.array(cam["H_world_to_image"], dtype=np.float32),
    )


def _make_calibrator(width: int, height: int, homography_path: Path, camera: str) -> CourtCalibrator:
    h_image_to_world, h_world_to_image = _load_homography_matrices(homography_path, camera)
    calibrator = CourtCalibrator(width, height)
    calibrator.H_pixel_to_real = h_image_to_world
    calibrator.H_real_to_pixel = h_world_to_image
    calibrator.map1 = None
    calibrator.map2 = None
    calibrator.net_offset_px = Config.NET_OFFSET_PX_DEFAULT
    calibrator.speed_line_offset_px = Config.SPEED_LINE_OFFSET_DEFAULT
    _set_net_and_speed_lines(calibrator)
    return calibrator


def _project_world(calibrator: CourtCalibrator, x: float, y: float) -> tuple[float, float]:
    pt = np.array([[[x, y]]], dtype=np.float32)
    out = cv2.perspectiveTransform(pt, calibrator.H_real_to_pixel)
    return float(out[0][0][0]), float(out[0][0][1])


def _normalized_line(p1: tuple[float, float], p2: tuple[float, float]) -> np.ndarray:
    line = np.cross([p1[0], p1[1], 1.0], [p2[0], p2[1], 1.0])
    norm = np.linalg.norm(line[:2])
    return line / norm if norm > 0 else line


def _set_net_and_speed_lines(calibrator: CourtCalibrator) -> None:
    p1 = _project_world(calibrator, -5.485, 0.0)
    p2 = _project_world(calibrator, 5.485, 0.0)
    p1_net = (p1[0], p1[1] - calibrator.net_offset_px)
    p2_net = (p2[0], p2[1] - calibrator.net_offset_px)
    p1_spd = (p1_net[0], p1_net[1] + calibrator.speed_line_offset_px)
    p2_spd = (p2_net[0], p2_net[1] + calibrator.speed_line_offset_px)
    calibrator.net_line_eq = _normalized_line(p1_net, p2_net)
    calibrator.speed_line_eq = _normalized_line(p1_spd, p2_spd)


def _detection_tuple_to_box(det: tuple[float, float, float, float, float, float], frame_index: int) -> dict[str, Any]:
    cx, cy, x1, y1, x2, y2 = [float(v) for v in det]
    return {
        "frame_index": int(frame_index),
        "pixel_x": cx,
        "pixel_y": cy,
        "bbox": [x1, y1, x2, y2],
    }


def _detect_ball_objects(model: Any, frame: np.ndarray, *, conf: float, imgsz: int, device: str) -> list[tuple[float, float, float, float, float, float]]:
    results = model(frame, conf=conf, imgsz=imgsz, device=device, half=(device != "cpu"), verbose=False)
    detections: list[tuple[float, float, float, float, float, float]] = []
    if not results or results[0].boxes is None:
        return detections
    for box in results[0].boxes:
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        detections.append((float(cx), float(cy), float(x1), float(y1), float(x2), float(y2)))
    return detections


def _parse_person_results(results: Any, frame_index: int, calibrator: CourtCalibrator) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    person_boxes: list[dict[str, Any]] = []
    current_players: list[dict[str, Any]] = []
    if not results or results[0].boxes is None or len(results[0].boxes) == 0:
        return person_boxes, current_players

    boxes = results[0].boxes.xyxy.cpu().numpy()
    if results[0].boxes.id is not None:
        track_ids = results[0].boxes.id.int().cpu().tolist()
    else:
        track_ids = list(range(len(boxes)))

    for box, track_id in zip(boxes, track_ids):
        px1, py1, px2, py2 = [float(v) for v in box[:4]]
        feet_x = (px1 + px2) / 2.0
        feet_y = py2
        real_coords = calibrator.pixel_to_real(feet_x, feet_y)
        rx, ry = real_coords if real_coords else (0.0, 0.0)
        hit_anchor_x = feet_x
        hit_anchor_y = py1 + (py2 - py1) * 0.3
        box_data = {
            "frame_index": int(frame_index),
            "id": int(track_id),
            "bbox": [px1, py1, px2, py2],
            "anchor_x": float(hit_anchor_x),
            "anchor_y": float(hit_anchor_y),
            "feet_x": float(feet_x),
            "feet_y": float(feet_y),
            "rx": float(rx),
            "ry": float(ry),
        }
        person_boxes.append(box_data)
        current_players.append({
            "id": int(track_id),
            "cx": float(hit_anchor_x),
            "cy": float(hit_anchor_y),
            "rx": float(rx),
            "ry": float(ry),
            "bbox": [px1, py1, px2, py2],
        })
    return person_boxes, current_players


def _run_detection_pass(
    *,
    image_paths: list[Path],
    ball_model_path: Path,
    person_model_path: Path,
    calibrator: CourtCalibrator,
    device: str,
    conf_ball: float,
    conf_person: float,
    imgsz_ball: int,
    imgsz_person: int,
    mask_osd: bool,
    progress_every: int,
) -> DetectionPass:
    from ultralytics import YOLO

    ball_model = YOLO(str(ball_model_path))
    person_model = YOLO(str(person_model_path))
    data = DetectionPass(image_paths=image_paths)
    start = time.time()

    for seq, image_path in enumerate(image_paths, start=1):
        frame_index = _safe_frame_index(image_path)
        frame = cv2.imread(str(image_path))
        if frame is None:
            data.ball_by_frame[frame_index] = []
            data.ball_boxes_by_frame[frame_index] = []
            data.player_boxes_by_frame[frame_index] = []
            data.player_history[frame_index] = []
            continue

        infer_frame = _mask_for_yolo(frame) if mask_osd else frame
        ball_dets = _detect_ball_objects(
            ball_model,
            infer_frame,
            conf=conf_ball,
            imgsz=imgsz_ball,
            device=device,
        )
        data.ball_by_frame[frame_index] = ball_dets
        data.ball_boxes_by_frame[frame_index] = [_detection_tuple_to_box(det, frame_index) for det in ball_dets]
        data.raw_ball_boxes += len(ball_dets)

        person_results = person_model.track(
            frame,
            conf=conf_person,
            imgsz=imgsz_person,
            classes=[0],
            persist=True,
            tracker="bytetrack.yaml",
            device=device,
            half=(device != "cpu"),
            verbose=False,
        )
        person_boxes, current_players = _parse_person_results(person_results, frame_index, calibrator)
        data.player_boxes_by_frame[frame_index] = person_boxes
        data.player_history[frame_index] = current_players
        data.raw_person_boxes += len(person_boxes)

        if progress_every > 0 and seq % progress_every == 0:
            print(
                f"detection pass {seq}/{len(image_paths)} frames "
                f"ball_boxes={data.raw_ball_boxes} person_boxes={data.raw_person_boxes} "
                f"elapsed={time.time() - start:.1f}s",
                flush=True,
            )

    return data


def _event_from_bounce(frame: int, px: float, py: float, angle: float, calibrator: CourtCalibrator) -> dict[str, Any] | None:
    real_coords = calibrator.pixel_to_real(px, py)
    if not real_coords:
        return None
    rx, ry = real_coords
    in_court = SINGLES_X_MIN <= rx <= SINGLES_X_MAX and COURT_Y_MIN <= ry <= COURT_Y_MAX
    return {
        "frame_index": int(frame),
        "pixel_x": float(px),
        "pixel_y": float(py),
        "x": float(rx),
        "y": float(ry),
        "world_x": float(rx),
        "world_y": float(ry),
        "angle": float(angle),
        "in_court": bool(in_court),
        "type": "IN" if in_court else "OUT",
    }


def _dedupe_event(events: dict[int, dict[str, Any]], new_frame: int, new_event: dict[str, Any]) -> int:
    keys_to_delete: list[int] = []
    for existing_frame, data in events.items():
        if abs(new_frame - existing_frame) <= Config.CLEAN_TIME_FRAMES:
            dist = math.hypot(float(new_event["x"]) - float(data["x"]), float(new_event["y"]) - float(data["y"]))
            if dist <= Config.CLEAN_SPACE_METERS:
                keys_to_delete.append(existing_frame)
    for key in keys_to_delete:
        del events[key]
    return len(keys_to_delete)


def _hit_event(
    *,
    frame: int,
    source: str,
    player: dict[str, Any],
    ball_px: float,
    ball_py: float,
    distance_px: float,
    angle: float | None = None,
    threshold_px: float | None = None,
    crossing_frame: int | None = None,
) -> dict[str, Any]:
    return {
        "frame_index": int(frame),
        "source": source,
        "pixel_x": float(player["cx"]),
        "pixel_y": float(player["cy"]),
        "ball_pixel_x": float(ball_px),
        "ball_pixel_y": float(ball_py),
        "x": float(player["rx"]),
        "y": float(player["ry"]),
        "world_x": float(player["rx"]),
        "world_y": float(player["ry"]),
        "player_id": int(player.get("id", -1)),
        "distance_px": float(distance_px),
        "angle": None if angle is None else float(angle),
        "threshold_px": None if threshold_px is None else float(threshold_px),
        "crossing_frame": crossing_frame,
    }


def _find_top_hit_candidate(
    *,
    crossing_frame: int,
    ball_history: dict[int, tuple[float, float]],
    player_history: dict[int, list[dict[str, Any]]],
) -> dict[str, Any] | None:
    best: dict[str, Any] | None = None
    search_end = max(0, crossing_frame - Config.TOP_HIT_LOOKBACK_FRAMES)
    for frame in range(crossing_frame, search_end - 1, -1):
        if frame not in ball_history or frame not in player_history:
            continue
        bx, by = ball_history[frame]
        for player in player_history[frame]:
            if abs(float(player["rx"])) > SINGLES_X_MAX + Config.ROI_SIDE_MARGIN:
                continue
            if float(player["ry"]) >= 0:
                continue
            dist = math.hypot(float(bx) - float(player["cx"]), float(by) - float(player["cy"]))
            if best is None or dist < float(best["distance_px"]):
                best = {
                    "frame_index": int(frame),
                    "crossing_frame": int(crossing_frame),
                    "ball_pixel_x": float(bx),
                    "ball_pixel_y": float(by),
                    "player": player,
                    "distance_px": float(dist),
                    "threshold_px": float(Config.HIT_DIST_PX_TOP_MAX),
                }
    return best


def _find_bottom_hit_candidate(
    *,
    bounce_frame: int,
    bx: float,
    by: float,
    angle: float,
    ry: float,
    player_history: dict[int, list[dict[str, Any]]],
    distance_mode: str,
) -> dict[str, Any] | None:
    if ry <= 0 or angle <= Config.HIT_ANGLE_THR:
        return None
    best: dict[str, Any] | None = None
    for frame in [bounce_frame, bounce_frame - 1, bounce_frame + 1, bounce_frame - 2, bounce_frame + 2]:
        if frame not in player_history:
            continue
        for player in player_history[frame]:
            if abs(float(player["rx"])) > SINGLES_X_MAX + Config.ROI_SIDE_MARGIN:
                continue
            if float(player["ry"]) <= 0:
                continue
            if distance_mode == "fixed_base":
                dynamic_dist_thr = float(Config.HIT_DIST_PX_BASE)
            else:
                ry_ratio = min(max(float(player["ry"]), 0.0), COURT_Y_MAX) / COURT_Y_MAX
                dynamic_dist_thr = Config.HIT_DIST_PX_NET + (Config.HIT_DIST_PX_BASE - Config.HIT_DIST_PX_NET) * ry_ratio
            dist = math.hypot(float(bx) - float(player["cx"]), float(by) - float(player["cy"]))
            candidate = {
                "frame_index": int(bounce_frame),
                "player_frame": int(frame),
                "ball_pixel_x": float(bx),
                "ball_pixel_y": float(by),
                "player": player,
                "distance_px": float(dist),
                "threshold_px": float(dynamic_dist_thr),
                "angle": float(angle),
            }
            if best is None or dist < float(best["distance_px"]):
                best = candidate
    return best


def _suppress_near_hit(
    *,
    final_bounces: dict[int, dict[str, Any]],
    hit_frame: int,
    suppressed_bounces: list[dict[str, Any]],
    reason: str,
) -> None:
    for frame in sorted(list(final_bounces)):
        if abs(frame - hit_frame) <= HIT_SUPPRESS_RADIUS_FRAMES:
            event = final_bounces.pop(frame)
            suppressed = dict(event)
            suppressed["suppressed_by_hit_frame"] = int(hit_frame)
            suppressed["suppression_reason"] = reason
            suppressed_bounces.append(suppressed)


def _maybe_add_final_bounce(
    *,
    event: dict[str, Any],
    final_bounces: dict[int, dict[str, Any]],
    hits: dict[int, dict[str, Any]],
    suppressed_bounces: list[dict[str, Any]],
    suppress_hit_window: bool,
) -> int:
    frame = int(event["frame_index"])
    if suppress_hit_window:
        for hit_frame in hits:
            if abs(frame - hit_frame) <= HIT_SUPPRESS_RADIUS_FRAMES:
                suppressed = dict(event)
                suppressed["suppressed_by_hit_frame"] = int(hit_frame)
                suppressed["suppression_reason"] = "existing_hit_window"
                suppressed_bounces.append(suppressed)
                return 0
    cleaned = _dedupe_event(final_bounces, frame, event)
    final_bounces[frame] = event
    return cleaned


def _run_variant(
    *,
    spec: VariantSpec,
    detection: DetectionPass,
    calibrator: CourtCalibrator,
    progress_every: int,
) -> VariantResult:
    tracker: QueueTracker
    if spec.static_zone:
        tracker = QueueTracker()
    else:
        tracker = NoStaticZoneQueueTracker()
    stitcher = TrajectoryStitcher()
    analyzer = TrajectoryAnalyzer()

    per_frame: dict[int, dict[str, Any]] = {}
    ball_history: dict[int, tuple[float, float]] = {}
    raw_bounce_candidates: dict[int, dict[str, Any]] = {}
    final_bounces: dict[int, dict[str, Any]] = {}
    hits: dict[int, dict[str, Any]] = {}
    crossings: list[dict[str, Any]] = []
    top_hit_candidates: list[dict[str, Any]] = []
    bottom_hit_candidates: list[dict[str, Any]] = []
    suppressed_bounces: list[dict[str, Any]] = []
    processed_crossing_keys: set[tuple[int, str]] = set()
    processed_bounce_frames: set[int] = set()

    static_boxes = 0
    moving_boxes = 0
    trajectory_points = 0
    stitched_points_max = 0
    cleaned_duplicate_bounces = 0
    cleaned_duplicate_hits = 0
    start = time.time()

    for seq, image_path in enumerate(detection.image_paths, start=1):
        frame_index = _safe_frame_index(image_path)
        ball_dets = detection.ball_by_frame.get(frame_index, [])
        tracker.process_frame(frame_idx=frame_index, detections=ball_dets)
        if spec.stitching:
            before_stitched = sum(1 for q in tracker.queues for item in q["history"] if len(item) > 2 and item[2])
            tracker.queues = stitcher.stitch_queues(tracker.queues)
            after_stitched = sum(1 for q in tracker.queues for item in q["history"] if len(item) > 2 and item[2])
            stitched_points_max = max(stitched_points_max, after_stitched, before_stitched)

        for q in tracker.queues:
            if not q["is_static"]:
                for item in q["history"]:
                    f_idx, det, _is_stitched = item
                    ball_history[int(f_idx)] = (float(det[0]), float(det[1]))
        old_key = frame_index - 150
        ball_history.pop(old_key, None)

        moving_dets, static_dets, moving_segments, frame_bounces, frame_crossings = tracker.get_render_data(frame_index, analyzer, calibrator)
        static_boxes += len(static_dets)
        moving_boxes += len(moving_dets)
        trajectory_points += len(moving_dets)

        per_frame[frame_index] = {
            "raw_ball_boxes": detection.ball_boxes_by_frame.get(frame_index, []),
            "player_boxes": detection.player_boxes_by_frame.get(frame_index, []),
            "moving_boxes": [_detection_tuple_to_box(tuple(det), frame_index) for det in moving_dets],
            "static_boxes": [_detection_tuple_to_box(tuple(det), frame_index) for det in static_dets],
            "moving_segments": [
                {
                    "p1": [float(pt1[0]), float(pt1[1])],
                    "p2": [float(pt2[0]), float(pt2[1])],
                    "stitched": bool(is_stitched),
                }
                for pt1, pt2, is_stitched in moving_segments
            ],
            "crossings": [],
        }

        for crossing in frame_crossings:
            direction = str(crossing["direction"])
            c_frame = int(crossing["frame"])
            key = (c_frame, direction)
            if key in processed_crossing_keys:
                continue
            processed_crossing_keys.add(key)
            crossing_event = {
                "frame_index": c_frame,
                "direction": direction,
                "speed_px": float(crossing.get("speed_px", 0.0)),
            }
            crossings.append(crossing_event)
            if c_frame == frame_index:
                per_frame[frame_index]["crossings"].append(crossing_event)

            if direction != "top_down" or not spec.enable_top_hit:
                continue
            candidate = _find_top_hit_candidate(
                crossing_frame=c_frame,
                ball_history=ball_history,
                player_history=detection.player_history,
            )
            if candidate is None:
                continue
            top_hit_candidates.append({
                "frame_index": candidate["frame_index"],
                "crossing_frame": candidate["crossing_frame"],
                "distance_px": candidate["distance_px"],
                "threshold_px": candidate["threshold_px"],
                "accepted": bool((not spec.top_distance_gate) or candidate["distance_px"] < Config.HIT_DIST_PX_TOP_MAX),
            })
            if spec.top_distance_gate and candidate["distance_px"] >= Config.HIT_DIST_PX_TOP_MAX:
                continue
            hit = _hit_event(
                frame=int(candidate["frame_index"]),
                source="top_down_lookback",
                player=candidate["player"],
                ball_px=float(candidate["ball_pixel_x"]),
                ball_py=float(candidate["ball_pixel_y"]),
                distance_px=float(candidate["distance_px"]),
                threshold_px=None if not spec.top_distance_gate else float(Config.HIT_DIST_PX_TOP_MAX),
                crossing_frame=c_frame,
            )
            cleaned_duplicate_hits += _dedupe_event(hits, int(hit["frame_index"]), hit)
            hits[int(hit["frame_index"])] = hit
            if spec.suppress_hit_window:
                _suppress_near_hit(
                    final_bounces=final_bounces,
                    hit_frame=int(hit["frame_index"]),
                    suppressed_bounces=suppressed_bounces,
                    reason="top_hit_window",
                )

        for b_frame, bx, by, angle in frame_bounces:
            b_frame = int(b_frame)
            if b_frame in processed_bounce_frames:
                continue
            processed_bounce_frames.add(b_frame)
            event = _event_from_bounce(b_frame, float(bx), float(by), float(angle), calibrator)
            if event is None:
                continue
            raw_bounce_candidates[b_frame] = event

            is_hit = False
            hit_event: dict[str, Any] | None = None
            bottom_candidate = _find_bottom_hit_candidate(
                bounce_frame=b_frame,
                bx=float(bx),
                by=float(by),
                angle=float(angle),
                ry=float(event["y"]),
                player_history=detection.player_history,
                distance_mode=spec.bottom_distance_mode,
            )
            if bottom_candidate is not None:
                accepted = spec.enable_bottom_hit and bottom_candidate["distance_px"] <= bottom_candidate["threshold_px"]
                bottom_hit_candidates.append({
                    "frame_index": b_frame,
                    "player_frame": bottom_candidate["player_frame"],
                    "angle": bottom_candidate["angle"],
                    "distance_px": bottom_candidate["distance_px"],
                    "threshold_px": bottom_candidate["threshold_px"],
                    "accepted": bool(accepted),
                })
                if accepted:
                    is_hit = True
                    hit_event = _hit_event(
                        frame=b_frame,
                        source="bottom_reversal_player_anchor",
                        player=bottom_candidate["player"],
                        ball_px=float(bx),
                        ball_py=float(by),
                        distance_px=float(bottom_candidate["distance_px"]),
                        angle=float(angle),
                        threshold_px=float(bottom_candidate["threshold_px"]),
                    )

            if is_hit and hit_event is not None:
                cleaned_duplicate_hits += _dedupe_event(hits, b_frame, hit_event)
                hits[b_frame] = hit_event
                if spec.suppress_hit_window:
                    _suppress_near_hit(
                        final_bounces=final_bounces,
                        hit_frame=b_frame,
                        suppressed_bounces=suppressed_bounces,
                        reason="bottom_hit_window",
                    )
                continue

            cleaned_duplicate_bounces += _maybe_add_final_bounce(
                event=event,
                final_bounces=final_bounces,
                hits=hits,
                suppressed_bounces=suppressed_bounces,
                suppress_hit_window=spec.suppress_hit_window,
            )

        if progress_every > 0 and seq % progress_every == 0:
            print(
                f"{spec.variant} {seq}/{len(detection.image_paths)} frames "
                f"raw_bounce={len(raw_bounce_candidates)} hits={len(hits)} final_bounce={len(final_bounces)}",
                flush=True,
            )

    top_down_crossings = [c for c in crossings if c["direction"] == "top_down"]
    bottom_reversal_candidates = [b for b in raw_bounce_candidates.values() if float(b["y"]) > 0]
    accepted_top_hits = [h for h in hits.values() if h["source"] == "top_down_lookback"]
    accepted_bottom_hits = [h for h in hits.values() if h["source"] == "bottom_reversal_player_anchor"]
    in_count = sum(1 for b in final_bounces.values() if bool(b.get("in_court", False)))
    out_count = len(final_bounces) - in_count
    stats = {
        "variant": spec.variant,
        "frames": len(detection.image_paths),
        "camera": CAMERA,
        "minimap_flip": MINIMAP_FLIP,
        "minimap_show_hits": MINIMAP_SHOW_HITS,
        "raw_ball_boxes": int(detection.raw_ball_boxes),
        "raw_person_boxes": int(detection.raw_person_boxes),
        "kept_ball_boxes": int(moving_boxes),
        "static_ball_boxes": int(static_boxes),
        "trajectory_points": int(trajectory_points),
        "stitched_points": int(stitched_points_max),
        "stitching_enabled": bool(spec.stitching),
        "static_zone_enabled": bool(spec.static_zone),
        "net_crossings": len(crossings),
        "top_down_crossings": len(top_down_crossings),
        "top_hit_candidates": len(top_hit_candidates),
        "top_hits": len(accepted_top_hits),
        "bottom_reversal_candidates": len(bottom_reversal_candidates),
        "bottom_hit_candidates": len(bottom_hit_candidates),
        "bottom_hits": len(accepted_bottom_hits),
        "raw_bounce_candidates": len(raw_bounce_candidates),
        "suppressed_bounces_by_hit_window": len(suppressed_bounces),
        "cleaned_duplicate_bounces": int(cleaned_duplicate_bounces),
        "cleaned_duplicate_hits": int(cleaned_duplicate_hits),
        "final_bounce_count": len(final_bounces),
        "in_count": int(in_count),
        "out_count": int(out_count),
        "hit_frames": sorted(int(f) for f in hits),
        "suppressed_bounce_frames": sorted(int(b["frame_index"]) for b in suppressed_bounces),
        "final_bounce_frames": sorted(int(f) for f in final_bounces),
        "runtime_seconds": round(time.time() - start, 2),
        "config": {
            "top_hit_lookback_frames": Config.TOP_HIT_LOOKBACK_FRAMES,
            "top_hit_distance_px": None if not spec.top_distance_gate else Config.HIT_DIST_PX_TOP_MAX,
            "bottom_hit_angle_thr": Config.HIT_ANGLE_THR,
            "bottom_hit_distance_mode": spec.bottom_distance_mode,
            "hit_suppress_radius_frames": HIT_SUPPRESS_RADIUS_FRAMES if spec.suppress_hit_window else 0,
            "clean_time_frames": Config.CLEAN_TIME_FRAMES,
            "clean_space_meters": Config.CLEAN_SPACE_METERS,
        },
    }
    return VariantResult(
        spec=spec,
        per_frame=per_frame,
        final_bounces=final_bounces,
        raw_bounce_candidates=raw_bounce_candidates,
        hits=hits,
        crossings=crossings,
        top_hit_candidates=top_hit_candidates,
        bottom_hit_candidates=bottom_hit_candidates,
        suppressed_bounces=suppressed_bounces,
        stats=stats,
    )


def _world_to_minimap(
    x: float,
    y: float,
    *,
    origin: tuple[int, int],
    size: tuple[int, int],
    pad: int = 18,
) -> tuple[int, int]:
    # Display mapping only. Source court coordinates stay unchanged.
    if MINIMAP_FLIP == "rotate180":
        x = COURT_X_MIN + COURT_X_MAX - x
        y = COURT_Y_MIN + COURT_Y_MAX - y

    ox, oy = origin
    mw, mh = size
    sx = (mw - pad * 2) / (COURT_X_MAX - COURT_X_MIN)
    sy = (mh - pad * 2) / (COURT_Y_MAX - COURT_Y_MIN)
    scale = min(sx, sy)
    court_w = (COURT_X_MAX - COURT_X_MIN) * scale
    court_h = (COURT_Y_MAX - COURT_Y_MIN) * scale
    left = ox + (mw - court_w) / 2.0
    top = oy + (mh - court_h) / 2.0
    px = int(round(left + (x - COURT_X_MIN) * scale))
    py = int(round(top + (COURT_Y_MAX - y) * scale))
    return px, py


def _draw_minimap_base(image: np.ndarray, *, origin: tuple[int, int], size: tuple[int, int]) -> None:
    ox, oy = origin
    mw, mh = size
    cv2.rectangle(image, (ox, oy), (ox + mw, oy + mh), (18, 48, 30), -1)
    cv2.rectangle(image, (ox, oy), (ox + mw, oy + mh), (70, 105, 74), 1, cv2.LINE_AA)

    def pt(x: float, y: float) -> tuple[int, int]:
        return _world_to_minimap(x, y, origin=origin, size=size)

    white = (232, 242, 232)
    doubles_x_min = -10.97 / 2.0
    doubles_x_max = 10.97 / 2.0
    for xmin, xmax in ((doubles_x_min, doubles_x_max), (SINGLES_X_MIN, SINGLES_X_MAX)):
        cv2.rectangle(image, pt(xmin, COURT_Y_MAX), pt(xmax, COURT_Y_MIN), white, 1, cv2.LINE_AA)
    cv2.line(image, pt(doubles_x_min, 0.0), pt(doubles_x_max, 0.0), (180, 220, 180), 1, cv2.LINE_AA)
    for sy in (-6.40, 6.40):
        cv2.line(image, pt(SINGLES_X_MIN, sy), pt(SINGLES_X_MAX, sy), white, 1, cv2.LINE_AA)
    cv2.line(image, pt(0.0, -6.40), pt(0.0, 6.40), white, 1, cv2.LINE_AA)


def _draw_minimap_bounces(
    image: np.ndarray,
    *,
    origin: tuple[int, int],
    size: tuple[int, int],
    bounces: list[dict[str, Any]],
    current_frame: int,
) -> None:
    visible = [b for b in bounces if int(b["frame_index"]) <= current_frame][-80:]
    if not visible:
        return
    latest = visible[-1]
    for bounce in visible:
        p = _world_to_minimap(float(bounce["x"]), float(bounce["y"]), origin=origin, size=size)
        is_latest = bounce is latest
        if bool(bounce.get("in_court", False)):
            if is_latest:
                cv2.circle(image, p, 12, (230, 255, 230), 1, cv2.LINE_AA)
                cv2.circle(image, p, 8, (50, 255, 110), 1, cv2.LINE_AA)
            cv2.circle(image, p, 5 if is_latest else 4, (50, 255, 110), -1, cv2.LINE_AA)
        else:
            if is_latest:
                cv2.circle(image, p, 12, (45, 65, 245), 1, cv2.LINE_AA)
            cv2.circle(image, p, 6 if is_latest else 5, (45, 65, 245), 2, cv2.LINE_AA)


def _draw_minimap_hits(
    image: np.ndarray,
    *,
    origin: tuple[int, int],
    size: tuple[int, int],
    hits: list[dict[str, Any]],
    current_frame: int,
) -> None:
    visible = [h for h in hits if int(h["frame_index"]) <= current_frame][-80:]
    if not visible:
        return
    latest = visible[-1]
    for hit in visible:
        p = _world_to_minimap(float(hit["x"]), float(hit["y"]), origin=origin, size=size)
        is_latest = hit is latest
        color = (0, 165, 255)
        if is_latest:
            cv2.circle(image, p, 11, (0, 210, 255), 1, cv2.LINE_AA)
        cv2.drawMarker(
            image,
            p,
            color,
            markerType=cv2.MARKER_DIAMOND,
            markerSize=11 if is_latest else 9,
            thickness=2,
            line_type=cv2.LINE_AA,
        )
        cv2.drawMarker(
            image,
            p,
            (0, 220, 255),
            markerType=cv2.MARKER_TILTED_CROSS,
            markerSize=9 if is_latest else 7,
            thickness=1,
            line_type=cv2.LINE_AA,
        )


def _draw_text(
    image: np.ndarray,
    text: str,
    org: tuple[int, int],
    *,
    scale: float = 0.54,
    color: tuple[int, int, int] = (245, 245, 245),
    thickness: int = 1,
) -> None:
    cv2.putText(image, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thickness + 3, cv2.LINE_AA)
    cv2.putText(image, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def _draw_box(
    image: np.ndarray,
    box: dict[str, Any],
    *,
    scale: float,
    color: tuple[int, int, int],
    thickness: int,
    min_size: int,
) -> None:
    bbox = box.get("bbox") or []
    px = int(round(float(box["pixel_x"]) * scale))
    py = int(round(float(box["pixel_y"]) * scale))
    if len(bbox) == 4:
        x1, y1, x2, y2 = [int(round(float(v) * scale)) for v in bbox]
    else:
        x1, y1, x2, y2 = px - 4, py - 4, px + 4, py + 4
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    half_w = max(min_size // 2, abs(x2 - x1) // 2)
    half_h = max(min_size // 2, abs(y2 - y1) // 2)
    h, w = image.shape[:2]
    x1 = max(0, cx - half_w)
    x2 = min(w - 1, cx + half_w)
    y1 = max(0, cy - half_h)
    y2 = min(h - 1, cy + half_h)
    if thickness >= 3:
        cv2.rectangle(image, (x1 - 1, y1 - 1), (x2 + 1, y2 + 1), (0, 0, 0), thickness + 2, cv2.LINE_AA)
    cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness, cv2.LINE_AA)
    cv2.circle(image, (px, py), 5 if thickness >= 2 else 3, (0, 0, 0), -1, cv2.LINE_AA)
    cv2.circle(image, (px, py), 3 if thickness >= 2 else 2, color, -1, cv2.LINE_AA)


def _draw_person(image: np.ndarray, person: dict[str, Any], *, scale: float) -> None:
    x1, y1, x2, y2 = [int(round(float(v) * scale)) for v in person["bbox"]]
    ax = int(round(float(person["anchor_x"]) * scale))
    ay = int(round(float(person["anchor_y"]) * scale))
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 220, 120), 2, cv2.LINE_AA)
    cv2.circle(image, (ax, ay), 6, (0, 0, 0), -1, cv2.LINE_AA)
    cv2.circle(image, (ax, ay), 4, (0, 220, 255), -1, cv2.LINE_AA)


def _draw_segments(image: np.ndarray, segments: list[dict[str, Any]], *, scale: float) -> None:
    for seg in segments:
        p1 = (int(round(seg["p1"][0] * scale)), int(round(seg["p1"][1] * scale)))
        p2 = (int(round(seg["p2"][0] * scale)), int(round(seg["p2"][1] * scale)))
        color = (0, 165, 255) if seg.get("stitched") else (255, 0, 255)
        cv2.line(image, p1, p2, color, 2 if not seg.get("stitched") else 1, cv2.LINE_AA)


def _recent_events(events: dict[int, dict[str, Any]], current_frame: int, linger: int) -> list[dict[str, Any]]:
    return [
        event for frame, event in sorted(events.items())
        if 0 <= current_frame - int(frame) <= linger
    ]


def _render_video(
    *,
    result: VariantResult,
    image_paths: list[Path],
    output_path: Path,
    fps: float,
    display_scale: float,
    panel_width: int,
    linger_frames: int,
) -> dict[str, Any]:
    first = cv2.imread(str(image_paths[0]))
    if first is None:
        raise RuntimeError(f"Cannot read first frame: {image_paths[0]}")
    h, w = first.shape[:2]
    frame_w = int(round(w * display_scale))
    frame_h = int(round(h * display_scale))
    if frame_h % 2:
        frame_h += 1
    out_w = frame_w + panel_width
    if out_w % 2:
        out_w += 1

    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (out_w, frame_h))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer: {output_path}")

    sorted_bounces = [result.final_bounces[k] for k in sorted(result.final_bounces)]
    sorted_hits = [result.hits[k] for k in sorted(result.hits)]
    minimap_origin = (16, 18)
    minimap_size = (panel_width - 32, frame_h - 36)

    try:
        for seq, image_path in enumerate(image_paths, start=1):
            frame_index = _safe_frame_index(image_path)
            frame = cv2.imread(str(image_path))
            if frame is None:
                frame = np.zeros((h, w, 3), dtype=np.uint8)
            left = cv2.resize(frame, (frame_w, frame_h), interpolation=cv2.INTER_AREA)
            frame_data = result.per_frame.get(frame_index, {})

            _draw_segments(left, frame_data.get("moving_segments", []), scale=display_scale)
            for box in frame_data.get("raw_ball_boxes", []):
                _draw_box(left, box, scale=display_scale, color=(110, 110, 110), thickness=1, min_size=16)
            for box in frame_data.get("static_boxes", []):
                _draw_box(left, box, scale=display_scale, color=(80, 80, 220), thickness=1, min_size=18)
            for box in frame_data.get("moving_boxes", []):
                _draw_box(left, box, scale=display_scale, color=(0, 255, 255), thickness=4, min_size=34)
            for person in frame_data.get("player_boxes", []):
                _draw_person(left, person, scale=display_scale)

            for crossing in frame_data.get("crossings", []):
                color = (255, 80, 255) if crossing["direction"] == "top_down" else (255, 180, 40)
                _draw_text(left, f"CROSS {crossing['direction']}", (16, 90), scale=0.55, color=color, thickness=2)

            for hit in _recent_events(result.hits, frame_index, linger_frames):
                hp = (int(round(float(hit["pixel_x"]) * display_scale)), int(round(float(hit["pixel_y"]) * display_scale)))
                bp = (int(round(float(hit["ball_pixel_x"]) * display_scale)), int(round(float(hit["ball_pixel_y"]) * display_scale)))
                cv2.circle(left, hp, 22, (0, 165, 255), 3, cv2.LINE_AA)
                cv2.circle(left, hp, 5, (0, 165, 255), -1, cv2.LINE_AA)
                cv2.line(left, hp, bp, (0, 165, 255), 1, cv2.LINE_AA)
                _draw_text(left, "HIT", (hp[0] + 16, max(24, hp[1] - 12)), scale=0.55, color=(0, 190, 255), thickness=2)

            for bounce in _recent_events(result.final_bounces, frame_index, linger_frames):
                bp = (int(round(float(bounce["pixel_x"]) * display_scale)), int(round(float(bounce["pixel_y"]) * display_scale)))
                color = (50, 255, 110) if bool(bounce.get("in_court", False)) else (45, 65, 245)
                cv2.circle(left, bp, 18, color, 3, cv2.LINE_AA)
                cv2.circle(left, bp, 5, color, -1, cv2.LINE_AA)
                _draw_text(left, "BOUNCE", (bp[0] + 18, max(24, bp[1] - 12)), scale=0.5, color=color, thickness=2)

            _draw_text(left, result.spec.variant, (16, 30), scale=0.62, color=(255, 255, 255), thickness=2)
            _draw_text(
                left,
                f"frame {frame_index:05d}  raw {len(result.raw_bounce_candidates)}  hits {len(result.hits)}  final {len(result.final_bounces)}",
                (16, 58),
                scale=0.5,
                color=(225, 255, 225),
            )

            canvas = np.zeros((frame_h, out_w, 3), dtype=np.uint8)
            canvas[:, :frame_w] = left
            panel = canvas[:, frame_w:]
            panel[:] = (15, 18, 16)
            _draw_minimap_base(panel, origin=minimap_origin, size=minimap_size)
            _draw_minimap_bounces(
                panel,
                origin=minimap_origin,
                size=minimap_size,
                bounces=sorted_bounces,
                current_frame=frame_index,
            )
            if MINIMAP_SHOW_HITS:
                _draw_minimap_hits(
                    panel,
                    origin=minimap_origin,
                    size=minimap_size,
                    hits=sorted_hits,
                    current_frame=frame_index,
                )
            writer.write(canvas)
            if seq % 500 == 0:
                print(f"rendered {result.spec.variant} {seq}/{len(image_paths)} frames", flush=True)
    finally:
        writer.release()

    return {"video": str(output_path), "width": out_w, "height": frame_h, "frames": len(image_paths), "fps": fps}


def _result_to_json(result: VariantResult) -> dict[str, Any]:
    return {
        "variant": result.spec.variant,
        "description": result.spec.description,
        "video_name": result.spec.video_name,
        "stats": result.stats,
        "net_crossings": result.crossings,
        "top_hit_candidates": result.top_hit_candidates,
        "bottom_hit_candidates": result.bottom_hit_candidates,
        "hits": [result.hits[k] for k in sorted(result.hits)],
        "raw_bounce_candidates": [result.raw_bounce_candidates[k] for k in sorted(result.raw_bounce_candidates)],
        "suppressed_bounces": result.suppressed_bounces,
        "final_bounces": [result.final_bounces[k] for k in sorted(result.final_bounces)],
        "render": result.render,
    }


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_readme(path: Path, results: list[VariantResult]) -> None:
    lines = [
        "# verify_tennis.py Hit/Bounce Ablation",
        "",
        f"Minimap flip is {MINIMAP_FLIP}; it shows bounces and hits in every video.",
        "",
        "| Variant | Video | Purpose | Final Bounce | Hits | Suppressed |",
        "|---|---|---|---:|---:|---:|",
    ]
    for result in results:
        s = result.stats
        lines.append(
            f"| {result.spec.variant} | `{result.spec.video_name}` | {result.spec.description} | "
            f"{s['final_bounce_count']} | {len(s['hit_frames'])} | {s['suppressed_bounces_by_hit_window']} |"
        )
    lines.append("")
    lines.append("Use `ablation_summary.json` for frame-level comparisons and per-module counters.")
    path.write_text("\n".join(lines), encoding="utf-8")


def _build_variants(selected: set[str] | None) -> list[VariantSpec]:
    variants = [
        VariantSpec(
            "HB0_full_hit_bounce_chain",
            "HB0_full_hit_bounce_chain.mp4",
            "Full Module1+2+3+4 chain with top/bottom HIT and plus/minus 3-frame HIT suppression.",
        ),
        VariantSpec(
            "HB1_no_hit_suppression",
            "HB1_no_hit_suppression.mp4",
            "Same as HB0 but HIT does not suppress nearby BOUNCE candidates.",
            suppress_hit_window=False,
        ),
        VariantSpec(
            "HB2_top_down_crossing_only",
            "HB2_top_down_crossing_only.mp4",
            "Count and show top_down crossings without top-hit lookback or bottom-hit separation.",
            enable_top_hit=False,
            enable_bottom_hit=False,
            suppress_hit_window=False,
            top_crossing_only=True,
        ),
        VariantSpec(
            "HB3_top_hit_lookback_50",
            "HB3_top_hit_lookback_50.mp4",
            "Enable only top_down 50-frame lookback HIT separation plus HIT window suppression.",
            enable_bottom_hit=False,
        ),
        VariantSpec(
            "HB4_top_hit_distance_off",
            "HB4_top_hit_distance_off.mp4",
            "Top_down lookback accepts nearest upper-half player without the 50px distance gate.",
            enable_bottom_hit=False,
            top_distance_gate=False,
        ),
        VariantSpec(
            "HB5_bottom_bounce_raw",
            "HB5_bottom_bounce_raw.mp4",
            "Treat lower-half y-reversal candidates as raw bounces without player HIT separation.",
            enable_top_hit=False,
            enable_bottom_hit=False,
            suppress_hit_window=False,
        ),
        VariantSpec(
            "HB6_bottom_hit_angle_45",
            "HB6_bottom_hit_angle_45.mp4",
            "Enable lower-half angle>45 HIT filtering with fixed 250px player-anchor distance.",
            enable_top_hit=False,
            bottom_distance_mode="fixed_base",
        ),
        VariantSpec(
            "HB7_bottom_player_anchor_dynamic",
            "HB7_bottom_player_anchor_dynamic.mp4",
            "Enable lower-half angle>45 HIT filtering with 100px-250px dynamic player-anchor distance.",
            enable_top_hit=False,
            bottom_distance_mode="dynamic",
        ),
        VariantSpec(
            "HB8_no_stitching",
            "HB8_no_stitching.mp4",
            "Full hit/bounce chain with TrajectoryStitcher disabled.",
            stitching=False,
        ),
        VariantSpec(
            "HB9_tracker_no_static_zone",
            "HB9_tracker_no_static_zone.mp4",
            "Full hit/bounce chain with QueueTracker static-zone absorption disabled.",
            static_zone=False,
        ),
        VariantSpec(
            "HB10_final_minimap_only",
            "HB10_final_minimap_only.mp4",
            "Final review video rendered from HB0 events; no event recomputation.",
            reuse_variant="HB0_full_hit_bounce_chain",
        ),
    ]
    if selected is None:
        return variants
    selected = set(selected)
    if "HB10" in selected or "HB10_final_minimap_only" in selected:
        selected.add("HB0")
        selected.add("HB0_full_hit_bounce_chain")
    return [variant for variant in variants if variant.variant in selected or variant.variant.split("_", 1)[0] in selected]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run verify_tennis.py hit/bounce ablations.")
    parser.add_argument("--camera", default=CAMERA, choices=("cam66", "cam68"))
    parser.add_argument("--minimap-flip", choices=("auto", "none", "rotate180"), default="auto")
    parser.add_argument("--minimap-hide-hits", action="store_true")
    parser.add_argument("--frames-dir", type=Path, default=DEFAULT_FRAMES_DIR)
    parser.add_argument("--video", type=Path, default=None)
    parser.add_argument("--ball-model", type=Path, default=DEFAULT_BALL_MODEL)
    parser.add_argument("--person-model", type=Path, default=DEFAULT_PERSON_MODEL)
    parser.add_argument("--homography", type=Path, default=DEFAULT_HOMOGRAPHY)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--max-frames", type=int, default=1500)
    parser.add_argument("--fps", type=float, default=25.0)
    parser.add_argument("--display-scale", type=float, default=0.58)
    parser.add_argument("--panel-width", type=int, default=300)
    parser.add_argument("--linger-frames", type=int, default=15)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--conf-ball", type=float, default=Config.CONF_BALL)
    parser.add_argument("--conf-person", type=float, default=Config.CONF_PERSON)
    parser.add_argument("--imgsz-ball", type=int, default=Config.IMGSZ_BALL)
    parser.add_argument("--imgsz-person", type=int, default=Config.IMGSZ_PERSON)
    parser.add_argument("--mask-osd", action="store_true", default=True)
    parser.add_argument("--no-mask-osd", action="store_false", dest="mask_osd")
    parser.add_argument("--skip-videos", action="store_true")
    parser.add_argument("--variants", nargs="*", default=None, help="Optional HB ids or full variant names.")
    parser.add_argument("--progress-every", type=int, default=250)
    return parser.parse_args()


def main() -> None:
    global CAMERA, MINIMAP_FLIP, MINIMAP_SHOW_HITS
    args = parse_args()
    CAMERA = args.camera
    MINIMAP_FLIP = "none" if args.minimap_flip == "auto" and CAMERA == "cam66" else "rotate180" if args.minimap_flip == "auto" else args.minimap_flip
    MINIMAP_SHOW_HITS = not args.minimap_hide_hits
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = args.out_root / f"verify_tennis_hit_bounce_ablation_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.video is not None:
        print(f"extracting_video={args.video}", flush=True)
        image_paths, detected_fps = _extract_video_frames(args.video, out_dir / "_input_frames", args.max_frames)
        if args.fps <= 0:
            args.fps = detected_fps
    else:
        image_paths = _load_image_paths(args.frames_dir, args.max_frames)

    first = cv2.imread(str(image_paths[0]))
    if first is None:
        raise RuntimeError(f"Cannot read first frame: {image_paths[0]}")
    h, w = first.shape[:2]
    calibrator = _make_calibrator(w, h, args.homography, CAMERA)

    print(f"output_dir={out_dir}", flush=True)
    detection = _run_detection_pass(
        image_paths=image_paths,
        ball_model_path=args.ball_model,
        person_model_path=args.person_model,
        calibrator=calibrator,
        device=args.device,
        conf_ball=args.conf_ball,
        conf_person=args.conf_person,
        imgsz_ball=args.imgsz_ball,
        imgsz_person=args.imgsz_person,
        mask_osd=args.mask_osd,
        progress_every=args.progress_every,
    )

    selected = set(args.variants) if args.variants else None
    variants = _build_variants(selected)
    results_by_name: dict[str, VariantResult] = {}
    ordered_results: list[VariantResult] = []

    for spec in variants:
        if spec.reuse_variant:
            if spec.reuse_variant not in results_by_name:
                raise RuntimeError(f"{spec.variant} needs {spec.reuse_variant}, but it was not selected/run")
            base = results_by_name[spec.reuse_variant]
            result = VariantResult(
                spec=spec,
                per_frame=base.per_frame,
                final_bounces=base.final_bounces,
                raw_bounce_candidates=base.raw_bounce_candidates,
                hits=base.hits,
                crossings=base.crossings,
                top_hit_candidates=base.top_hit_candidates,
                bottom_hit_candidates=base.bottom_hit_candidates,
                suppressed_bounces=base.suppressed_bounces,
                stats={**base.stats, "variant": spec.variant, "reused_from": spec.reuse_variant},
            )
        else:
            print(f"running {spec.variant}", flush=True)
            result = _run_variant(
                spec=spec,
                detection=detection,
                calibrator=calibrator,
                progress_every=args.progress_every,
            )

        if not args.skip_videos:
            video_path = out_dir / spec.video_name
            result.render = _render_video(
                result=result,
                image_paths=image_paths,
                output_path=video_path,
                fps=args.fps,
                display_scale=args.display_scale,
                panel_width=args.panel_width,
                linger_frames=args.linger_frames,
            )
        json_path = out_dir / f"{Path(spec.video_name).stem}.json"
        result.json_path = str(json_path)
        _write_json(json_path, _result_to_json(result))
        results_by_name[spec.variant] = result
        ordered_results.append(result)

    summary = {
        "output_dir": str(out_dir),
        "camera": CAMERA,
        "minimap_flip": MINIMAP_FLIP,
        "minimap_show_hits": MINIMAP_SHOW_HITS,
        "inputs": {
            "video": None if args.video is None else str(args.video),
            "frames_dir": str(args.frames_dir),
            "ball_model": str(args.ball_model),
            "person_model": str(args.person_model),
            "homography": str(args.homography),
            "max_frames": args.max_frames,
            "mask_osd": bool(args.mask_osd),
        },
        "variants": [
            {
                **result.stats,
                "description": result.spec.description,
                "video": None if result.render is None else result.render["video"],
                "json": result.json_path,
                "added_vs_HB0": [],
                "removed_vs_HB0": [],
            }
            for result in ordered_results
        ],
    }

    hb0 = next((r for r in ordered_results if r.spec.variant == "HB0_full_hit_bounce_chain"), None)
    if hb0 is not None:
        hb0_frames = set(int(f) for f in hb0.final_bounces)
        for item in summary["variants"]:
            result_frames = set(int(f) for f in item.get("final_bounce_frames", []))
            item["added_vs_HB0"] = sorted(result_frames - hb0_frames)
            item["removed_vs_HB0"] = sorted(hb0_frames - result_frames)

    _write_json(out_dir / "ablation_summary.json", summary)
    _write_readme(out_dir / "README.md", ordered_results)
    print(json.dumps({"output_dir": str(out_dir), "summary": str(out_dir / "ablation_summary.json")}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
