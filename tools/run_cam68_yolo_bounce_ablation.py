"""Run cam68 YOLO-to-bounce ablations with box, trajectory, and minimap video.

Outputs:
  A0_yolo_raw_box_trajectory.mp4
  A1_yolo_original_static_trajectory.mp4
  A2_yolo_dashboard_static_trajectory.mp4
  A3_dashboard_integrated_replay.json
  A3_dashboard_integrated_replay_video.mp4
  A4_dashboard_integrated_minimap_video.mp4
  ablation_summary.json

The minimap is intentionally bounce-only and fixed to rotate180 display
mapping. Detection/bounce data stay in the original court coordinate system.
YOLO boxes are drawn with a minimum visible display size so tiny ball boxes do
not disappear in review videos.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from app.config import load_config
from app.orchestrator import Orchestrator
from app.pipeline.homography import HomographyTransformer
from app.pipeline.inference import YoloRoadmapDetector
from app.pipeline.yolo_bounce_filter import (
    COURT_Y_MAX,
    COURT_Y_MIN,
    SINGLES_X_MAX,
    SINGLES_X_MIN,
    detect_single_camera_bounces,
)


DEFAULT_FRAMES_DIR = Path(r"D:\tennis-dataset\1001\clip11\cam68_20260404_075325_2min")
DEFAULT_MODEL = Path(r"D:\tennis\tennis-3d-tracking\yolo_roadmap\best.pt")
DEFAULT_HOMOGRAPHY = Path(r"D:\tennis\tennis-3d-tracking\src\homography_matrices.json")
DEFAULT_OUT_ROOT = Path(r"D:\tennis\tennis-3d-tracking\reports")

CAMERA = "cam68"
MINIMAP_FLIP = "rotate180"
COURT_X_MIN = -10.97 / 2.0
COURT_X_MAX = 10.97 / 2.0


@dataclass
class VariantRun:
    variant: str
    pipeline: str
    detections: list[dict[str, Any]]
    per_frame: dict[int, dict[str, Any]]
    bounces: list[dict[str, Any]]
    stats: dict[str, Any]


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


def _mask_for_yolo(frame: np.ndarray) -> np.ndarray:
    frame_for_infer = frame.copy()
    frame_for_infer[0:41, 0:603] = 0
    return frame_for_infer


def _boxes_from_result(result: Any) -> list[dict[str, Any]]:
    if result is None or result.boxes is None or len(result.boxes) == 0:
        return []
    boxes = result.boxes
    xywh = boxes.xywh.cpu().numpy()
    confs = boxes.conf.cpu().numpy()
    classes = boxes.cls.cpu().numpy() if boxes.cls is not None else np.zeros(len(xywh))
    track_ids = (
        boxes.id.int().cpu().numpy().tolist()
        if boxes.id is not None
        else [None] * len(xywh)
    )

    out: list[dict[str, Any]] = []
    for box, conf, cls_id, track_id in zip(xywh, confs, classes, track_ids):
        x, y, w, h = [float(v) for v in box]
        out.append({
            "pixel_x": x,
            "pixel_y": y,
            "bbox": [x - w / 2.0, y - h / 2.0, x + w / 2.0, y + h / 2.0],
            "yolo_conf": float(conf),
            "confidence": float(conf),
            "track_id": int(track_id) if track_id is not None else None,
            "class_id": int(cls_id),
            "width": float(w),
            "height": float(h),
        })
    out.sort(key=lambda item: float(item["yolo_conf"]), reverse=True)
    return out


def _box_to_detection(
    box: dict[str, Any],
    *,
    frame_index: int,
    homography: HomographyTransformer,
    camera: str,
    source: str,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    wx, wy = homography.pixel_to_world(float(box["pixel_x"]), float(box["pixel_y"]))
    det = {
        "camera_name": camera,
        "frame_index": int(frame_index),
        "timestamp": int(frame_index) / 25.0,
        "capture_ts": int(frame_index) / 25.0,
        "pixel_x": float(box["pixel_x"]),
        "pixel_y": float(box["pixel_y"]),
        "x": float(wx),
        "y": float(wy),
        "world_x": float(wx),
        "world_y": float(wy),
        "confidence": float(box.get("confidence", box.get("yolo_conf", 0.0)) or 0.0),
        "yolo_conf": float(box.get("yolo_conf", box.get("confidence", 0.0)) or 0.0),
        "bbox": list(box.get("bbox", [])),
        "track_id": box.get("track_id"),
        "source": source,
    }
    for key in (
        "pseudo_track_id",
        "static_count",
        "static_status",
        "static_zone_id",
        "class_id",
    ):
        if box.get(key) is not None:
            det[key] = box[key]
    if extra:
        det.update(extra)
    return det


def _run_track_model(
    *,
    image_paths: list[Path],
    model_path: Path,
    device: str,
    conf: float,
    imgsz: int,
    homography: HomographyTransformer,
    progress_every: int,
) -> tuple[VariantRun, VariantRun]:
    from ultralytics import YOLO

    model = YOLO(str(model_path))
    track_history: dict[int, list[float]] = {}

    a0_detections: list[dict[str, Any]] = []
    a1_detections: list[dict[str, Any]] = []
    a0_per_frame: dict[int, dict[str, Any]] = {}
    a1_per_frame: dict[int, dict[str, Any]] = {}

    raw_boxes = 0
    a1_active_boxes = 0
    a1_masked_boxes = 0
    a1_missing_track_id = 0

    start = time.time()
    for seq, image_path in enumerate(image_paths, start=1):
        frame_index = _safe_frame_index(image_path)
        frame = cv2.imread(str(image_path))
        if frame is None:
            a0_per_frame[frame_index] = {"boxes": [], "masked_boxes": [], "selected": None}
            a1_per_frame[frame_index] = {"boxes": [], "masked_boxes": [], "selected": None}
            continue

        results = model.track(
            _mask_for_yolo(frame),
            persist=True,
            conf=conf,
            imgsz=imgsz,
            device=device,
            verbose=False,
        )
        boxes = _boxes_from_result(results[0] if results else None)
        raw_boxes += len(boxes)

        selected_a0 = None
        if boxes:
            selected_a0 = _box_to_detection(
                boxes[0],
                frame_index=frame_index,
                homography=homography,
                camera=CAMERA,
                source="yolo_raw_top_conf",
            )
            a0_detections.append(selected_a0)
        a0_per_frame[frame_index] = {
            "boxes": [selected_a0] if selected_a0 else [],
            "masked_boxes": [],
            "selected": selected_a0,
        }

        active: list[dict[str, Any]] = []
        masked: list[dict[str, Any]] = []
        for box in boxes:
            track_id = box.get("track_id")
            if track_id is None:
                a1_missing_track_id += 1
                masked.append(
                    _box_to_detection(
                        box,
                        frame_index=frame_index,
                        homography=homography,
                        camera=CAMERA,
                        source="yolo_original_static",
                        extra={"static_status": "missing_track_id", "active": False},
                    )
                )
                continue

            x = float(box["pixel_x"])
            y = float(box["pixel_y"])
            if track_id in track_history:
                last_x, last_y, static_count = track_history[track_id]
                distance = float(math.hypot(x - last_x, y - last_y))
                static_count = static_count + 1.0 if distance < 5.0 else 0.0
            else:
                distance = float("inf")
                static_count = 0.0

            track_history[track_id] = [x, y, static_count]
            is_active = static_count < 3.0
            det = _box_to_detection(
                box,
                frame_index=frame_index,
                homography=homography,
                camera=CAMERA,
                source="yolo_original_static",
                extra={
                    "static_count": int(static_count),
                    "static_status": "active" if is_active else "static_masked",
                    "distance_px": None if math.isinf(distance) else round(distance, 3),
                    "active": bool(is_active),
                },
            )
            if is_active:
                active.append(det)
                a1_active_boxes += 1
            else:
                masked.append(det)
                a1_masked_boxes += 1

        selected_a1 = max(active, key=lambda det: float(det["yolo_conf"])) if active else None
        if selected_a1 is not None:
            a1_detections.append(selected_a1)
        a1_per_frame[frame_index] = {
            "boxes": active,
            "masked_boxes": masked,
            "selected": selected_a1,
        }

        if progress_every > 0 and seq % progress_every == 0:
            elapsed = time.time() - start
            print(
                f"A0/A1 processed {seq}/{len(image_paths)} frames "
                f"raw={raw_boxes} a0={len(a0_detections)} a1={len(a1_detections)} "
                f"masked={a1_masked_boxes} elapsed={elapsed:.1f}s",
                flush=True,
            )

    a0_bounces = _detect_bounces(a0_detections)
    a1_bounces = _detect_bounces(a1_detections)
    a1_segment_stats = _segment_stats([int(det["frame_index"]) for det in a1_detections], len(image_paths))

    a0 = VariantRun(
        variant="A0_yolo_raw_box_trajectory",
        pipeline="YOLO.track raw top-confidence box -> Homography -> detect_single_camera_bounces",
        detections=a0_detections,
        per_frame=a0_per_frame,
        bounces=a0_bounces,
        stats={
            "raw_boxes": raw_boxes,
            "kept_boxes": len(a0_detections),
            "trajectory_points": len(a0_detections),
            "static_blocked": 0,
            "runtime_seconds": round(time.time() - start, 2),
            "conf": conf,
            "imgsz": imgsz,
        },
    )
    a1 = VariantRun(
        variant="A1_yolo_original_static_trajectory",
        pipeline="YOLO.track -> original track_id static_count mask(move<5px, static>=3) -> Homography -> detect_single_camera_bounces",
        detections=a1_detections,
        per_frame=a1_per_frame,
        bounces=a1_bounces,
        stats={
            "raw_boxes": raw_boxes,
            "kept_boxes": a1_active_boxes,
            "trajectory_points": len(a1_detections),
            "static_blocked": a1_masked_boxes,
            "missing_track_id_boxes": a1_missing_track_id,
            "move_threshold_px": 5.0,
            "static_frame_limit": 3,
            "segment_stats": a1_segment_stats,
            "runtime_seconds": round(time.time() - start, 2),
            "conf": conf,
            "imgsz": imgsz,
        },
    )
    if len(a1_bounces) <= 1:
        a1.stats["static_explanation"] = (
            "Original static filtering leaves too few continuous active "
            "trajectory segments for the fuzzy bounce window."
        )
    return a0, a1


def _run_dashboard_static(
    *,
    image_paths: list[Path],
    model_path: Path,
    device: str,
    conf: float,
    imgsz: int,
    homography: HomographyTransformer,
    progress_every: int,
) -> VariantRun:
    detector = YoloRoadmapDetector(
        model_path=str(model_path),
        frames_in=1,
        frames_out=1,
        device=device,
        conf=conf,
        imgsz=imgsz,
        static_frame_limit=20,
    )
    detections: list[dict[str, Any]] = []
    per_frame: dict[int, dict[str, Any]] = {}
    start = time.time()

    for seq, image_path in enumerate(image_paths, start=1):
        frame_index = _safe_frame_index(image_path)
        frame = cv2.imread(str(image_path))
        if frame is None:
            per_frame[frame_index] = {"boxes": [], "masked_boxes": [], "selected": None}
            continue

        blobs = detector.infer([_mask_for_yolo(frame)])[0]
        selected = None
        converted: list[dict[str, Any]] = []
        for blob in blobs:
            det = _box_to_detection(
                {
                    "pixel_x": blob["pixel_x"],
                    "pixel_y": blob["pixel_y"],
                    "bbox": blob.get("bbox", []),
                    "confidence": blob.get("blob_sum", blob.get("yolo_conf", 0.0)),
                    "yolo_conf": blob.get("yolo_conf", blob.get("blob_sum", 0.0)),
                    "track_id": blob.get("track_id"),
                    "pseudo_track_id": blob.get("pseudo_track_id"),
                    "static_count": blob.get("static_count"),
                    "static_status": blob.get("static_status"),
                    "static_zone_id": blob.get("static_zone_id"),
                },
                frame_index=frame_index,
                homography=homography,
                camera=CAMERA,
                source="yolo_dashboard_static",
            )
            converted.append(det)
        if converted:
            selected = max(converted, key=lambda det: float(det["yolo_conf"]))
            detections.append(selected)
        per_frame[frame_index] = {
            "boxes": converted,
            "masked_boxes": [],
            "selected": selected,
        }

        if progress_every > 0 and seq % progress_every == 0:
            stats = detector.get_runtime_stats()
            elapsed = time.time() - start
            print(
                f"A2 processed {seq}/{len(image_paths)} frames "
                f"raw={stats.get('raw_detections', 0)} kept={stats.get('kept_detections', 0)} "
                f"top={len(detections)} blocked={stats.get('static_blocked', 0)} elapsed={elapsed:.1f}s",
                flush=True,
            )

    final_stats = detector.get_runtime_stats()
    bounces = _detect_bounces(detections)
    return VariantRun(
        variant="A2_yolo_dashboard_static_trajectory",
        pipeline="YoloRoadmapDetector(static zones, pseudo track, fail-open) -> selected kept box -> Homography -> detect_single_camera_bounces",
        detections=detections,
        per_frame=per_frame,
        bounces=bounces,
        stats={
            "raw_boxes": int(final_stats.get("raw_detections", 0)),
            "kept_boxes": int(final_stats.get("kept_detections", 0)),
            "trajectory_points": len(detections),
            "static_blocked": int(final_stats.get("static_blocked", 0)),
            "static_zones_created": int(final_stats.get("static_zones_created", 0)),
            "static_zones_expired": int(final_stats.get("static_zones_expired", 0)),
            "motion_released": int(final_stats.get("motion_released", 0)),
            "fail_open_kept": int(final_stats.get("fail_open_kept", 0)),
            "pseudo_tracked": int(final_stats.get("pseudo_tracked", 0)),
            "dashboard_static_frame_limit": 20,
            "conf": conf,
            "imgsz": imgsz,
            "runtime_seconds": round(time.time() - start, 2),
            "detector_final_stats": final_stats,
        },
    )


def _detect_bounces(detections: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = detect_single_camera_bounces(
        detections,
        camera_name=CAMERA,
        max_gap=3,
        smooth_window=3,
        filter_window=3,
        angle_thresh=10.0,
        momentum_thresh=15.0,
        tolerance=2,
    )
    return list(result.get("bounces", []))


def _segment_stats(frames: list[int], total_frames: int) -> dict[str, Any]:
    frames = sorted(set(int(f) for f in frames))
    if not frames:
        return {
            "active_frames": 0,
            "frames_without_active": total_frames,
            "segments": 0,
            "longest_segment": 0,
            "segments_ge_10": 0,
            "avg_gap": 0.0,
            "max_gap": 0,
            "gaps_gt_9": 0,
        }
    segments: list[int] = []
    current = 1
    gaps: list[int] = []
    for prev, curr in zip(frames, frames[1:]):
        gap = curr - prev
        if gap == 1:
            current += 1
        else:
            segments.append(current)
            current = 1
            gaps.append(gap)
    segments.append(current)
    return {
        "active_frames": len(frames),
        "frames_without_active": max(0, total_frames - len(frames)),
        "segments": len(segments),
        "longest_segment": max(segments),
        "segments_ge_10": sum(1 for size in segments if size >= 10),
        "avg_gap": round(float(np.mean(gaps)), 2) if gaps else 0.0,
        "max_gap": max(gaps) if gaps else 0,
        "gaps_gt_9": sum(1 for gap in gaps if gap > 9),
    }


def _run_dashboard_integrated_replay(
    *,
    detections: list[dict[str, Any]],
    config_path: Path,
    homography_path: Path,
    model_path: Path,
    fps: float,
    output_path: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    cfg = load_config(str(config_path))
    cfg.model.path = str(model_path)
    cfg.model.frames_in = 1
    cfg.model.frames_out = 1
    cfg.model.detector_type = "yolo_roadmap"
    cfg.homography.path = str(homography_path)

    orch = Orchestrator(cfg)
    orch._video_test_detections[CAMERA] = [dict(det) for det in detections]
    event_result = orch.compute_single_cam_bounces(CAMERA)
    event_bounces = list(event_result.get("bounces", []))

    with orch._analytics_lock:
        orch._bounce_mode = "mono_cam68"
        orch._ws_enabled = True
        orch._reset_ws_push_telemetry_locked()
        orch._live_bounces.clear()
        orch._total_live_bounces = 0

        for event in event_bounces:
            frame_index = int(event.get("frame_index", event.get("frame", 0)) or 0)
            bd = orch._normalize_live_bounce_dict(
                {
                    **event,
                    "camera_name": CAMERA,
                    "camera": CAMERA,
                    "bounce_mode": "mono_cam68",
                    "source": "yolo_dashboard_integrated_replay",
                    "timestamp": frame_index / fps,
                    "capture_ts": frame_index / fps,
                    "speed_kmh": event.get("speed_kmh", 0),
                },
                fallback_ts=frame_index / fps,
            )
            orch._record_live_bounce_locked(bd)

        minimap_events = [dict(item) for item in orch._live_bounces]
        ws_queue = [dict(item) for item in orch._ws_bounce_queue]
        ws_messages = [json.loads(orch._build_ws_bounce_message(item)) for item in ws_queue]
        orch._ws_enabled = False

    minimap_frames = [int(b.get("frame_index", b.get("frame", -1))) for b in minimap_events]
    push_frames = [int(b.get("frame_index", -1)) for b in ws_queue]
    checks = {
        "yolo_box_entered_dashboard_selection": bool(detections),
        "trajectory_points_entered_bounce": int(event_result.get("detections", 0)) == len(detections),
        "bounce_events_entered_minimap": len(minimap_events) == len(event_bounces),
        "push_enabled_audit": True,
        "push_payloads_queued": len(ws_queue),
        "minimap_and_push_counts_match": len(minimap_events) == len(ws_queue),
        "minimap_and_push_frame_lists_match": minimap_frames == push_frames,
        "minimap_frames": minimap_frames,
        "push_frames": push_frames,
        "ws_protocol_units": "x/y are meters*10 in ws_queue and websocket messages; raw_x/raw_y remain court meters",
    }

    replay = {
        "variant": "A3_dashboard_integrated_replay",
        "camera": CAMERA,
        "dashboard_method": "Orchestrator.compute_single_cam_bounces + _record_live_bounce_locked",
        "source_variant": "A2_yolo_dashboard_static_trajectory",
        "model": str(model_path),
        "homography": str(homography_path),
        "input_selected_boxes": len(detections),
        "event_result": event_result,
        "minimap_events": minimap_events,
        "ws_queue": ws_queue,
        "ws_messages": ws_messages,
        "checks": checks,
    }
    output_path.write_text(json.dumps(replay, ensure_ascii=False, indent=2), encoding="utf-8")
    return minimap_events, replay


def _world_to_minimap(
    x: float,
    y: float,
    *,
    origin: tuple[int, int],
    size: tuple[int, int],
    pad: int = 18,
) -> tuple[int, int]:
    # Fixed rotate180 display mapping. Do not mutate source court coords.
    x = COURT_X_MIN + COURT_X_MAX - x
    y = COURT_Y_MIN + COURT_Y_MAX - y

    ox, oy = origin
    mw, mh = size
    sx = (mw - pad * 2) / (COURT_X_MAX - COURT_X_MIN)
    sy = (mh - pad * 2) / (COURT_Y_MAX - COURT_Y_MIN)
    s = min(sx, sy)
    court_w = (COURT_X_MAX - COURT_X_MIN) * s
    court_h = (COURT_Y_MAX - COURT_Y_MIN) * s
    left = ox + (mw - court_w) / 2.0
    top = oy + (mh - court_h) / 2.0
    px = int(round(left + (x - COURT_X_MIN) * s))
    py = int(round(top + (COURT_Y_MAX - y) * s))
    return px, py


def _draw_minimap_base(
    image: np.ndarray,
    *,
    origin: tuple[int, int],
    size: tuple[int, int],
) -> None:
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
    visible = [
        b for b in bounces
        if int(b.get("frame_index", b.get("frame", -1)) or -1) <= current_frame
    ][-80:]
    if not visible:
        return
    latest = visible[-1]
    for bounce in visible:
        x = float(bounce.get("x", bounce.get("world_x", 0.0)))
        y = float(bounce.get("y", bounce.get("world_y", 0.0)))
        p = _world_to_minimap(x, y, origin=origin, size=size)
        is_latest = bounce is latest
        in_court = bool(bounce.get("in_court", bounce.get("type", "IN") != "OUT"))
        if in_court:
            if is_latest:
                cv2.circle(image, p, 12, (230, 255, 230), 1, cv2.LINE_AA)
                cv2.circle(image, p, 8, (50, 255, 110), 1, cv2.LINE_AA)
            cv2.circle(image, p, 5 if is_latest else 4, (50, 255, 110), -1, cv2.LINE_AA)
        else:
            if is_latest:
                cv2.circle(image, p, 12, (45, 65, 245), 1, cv2.LINE_AA)
            cv2.circle(image, p, 6 if is_latest else 5, (45, 65, 245), 2, cv2.LINE_AA)


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


def _draw_trail(image: np.ndarray, trail: deque[tuple[int, int]]) -> None:
    pts = list(trail)
    if len(pts) < 2:
        return
    for i in range(1, len(pts)):
        alpha = i / max(1, len(pts) - 1)
        color = (0, int(120 + 110 * alpha), 255)
        cv2.line(image, pts[i - 1], pts[i], color, 2, cv2.LINE_AA)


def _draw_detection_box(
    image: np.ndarray,
    det: dict[str, Any],
    *,
    scale: float,
    color: tuple[int, int, int],
    masked: bool = False,
    selected: bool = False,
) -> None:
    bbox = det.get("bbox") or []
    px = int(round(float(det["pixel_x"]) * scale))
    py = int(round(float(det["pixel_y"]) * scale))
    if len(bbox) == 4:
        x1, y1, x2, y2 = [int(round(float(v) * scale)) for v in bbox]
    else:
        x1, y1, x2, y2 = px - 4, py - 4, px + 4, py + 4

    min_size = 16 if masked else 28
    if selected:
        min_size = 34
    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2
    half_w = max(min_size // 2, abs(x2 - x1) // 2)
    half_h = max(min_size // 2, abs(y2 - y1) // 2)
    h, w = image.shape[:2]
    x1 = max(0, cx - half_w)
    x2 = min(w - 1, cx + half_w)
    y1 = max(0, cy - half_h)
    y2 = min(h - 1, cy + half_h)

    thickness = 1 if masked else 3
    if selected:
        thickness = 4
        # A subtle black underlay makes the YOLO box visible on white court lines.
        cv2.rectangle(image, (x1 - 1, y1 - 1), (x2 + 1, y2 + 1), (0, 0, 0), thickness + 2, cv2.LINE_AA)
    cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness, cv2.LINE_AA)
    cv2.circle(image, (px, py), 3 if masked else 6, (0, 0, 0), -1, cv2.LINE_AA)
    cv2.circle(image, (px, py), 2 if masked else 4, color, -1, cv2.LINE_AA)
    if selected:
        cv2.line(image, (px - 12, py), (px + 12, py), color, 2, cv2.LINE_AA)
        cv2.line(image, (px, py - 12), (px, py + 12), color, 2, cv2.LINE_AA)


def _render_variant_video(
    *,
    variant: VariantRun,
    image_paths: list[Path],
    output_path: Path,
    fps: float,
    display_scale: float,
    panel_width: int,
    trail_len: int,
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

    det_by_frame = {int(det["frame_index"]): det for det in variant.detections}
    bounce_by_frame: dict[int, list[dict[str, Any]]] = {}
    for bounce in variant.bounces:
        bounce_by_frame.setdefault(int(bounce.get("frame_index", bounce.get("frame", -1))), []).append(bounce)

    trail: deque[tuple[int, int]] = deque(maxlen=trail_len)
    recent_bounce_pixels: deque[tuple[int, tuple[int, int], bool]] = deque(maxlen=10)
    minimap_origin = (16, 18)
    minimap_size = (panel_width - 32, frame_h - 36)

    try:
        for seq, image_path in enumerate(image_paths, start=1):
            frame_index = _safe_frame_index(image_path)
            img = cv2.imread(str(image_path))
            if img is None:
                img = np.zeros((h, w, 3), dtype=np.uint8)
            left = cv2.resize(img, (frame_w, frame_h), interpolation=cv2.INTER_AREA)

            selected = det_by_frame.get(frame_index) or frame_data.get("selected")
            if selected is not None:
                px = int(round(float(selected["pixel_x"]) * display_scale))
                py = int(round(float(selected["pixel_y"]) * display_scale))
                trail.append((px, py))
            _draw_trail(left, trail)

            frame_data = variant.per_frame.get(frame_index, {})
            for det in frame_data.get("masked_boxes", []) or []:
                _draw_detection_box(left, det, scale=display_scale, color=(125, 125, 125), masked=True)

            selected_key = (
                int(selected["frame_index"]),
                round(float(selected["pixel_x"]), 3),
                round(float(selected["pixel_y"]), 3),
            ) if selected is not None else None
            for det in frame_data.get("boxes", []) or []:
                det_key = (
                    int(det["frame_index"]),
                    round(float(det["pixel_x"]), 3),
                    round(float(det["pixel_y"]), 3),
                )
                is_selected = selected_key is not None and det_key == selected_key
                color = (0, 255, 255) if is_selected else (0, 190, 255)
                _draw_detection_box(left, det, scale=display_scale, color=color, masked=False, selected=is_selected)

            for bounce in bounce_by_frame.get(frame_index, []):
                bx = int(round(float(bounce["pixel_x"]) * display_scale))
                by = int(round(float(bounce["pixel_y"]) * display_scale))
                recent_bounce_pixels.append((frame_index, (bx, by), bool(bounce.get("in_court", True))))

            for bfi, xy, in_court in list(recent_bounce_pixels):
                age = frame_index - bfi
                if age < 0 or age > linger_frames:
                    continue
                color = (50, 255, 110) if in_court else (45, 65, 245)
                cv2.circle(left, xy, 18, color, 3, cv2.LINE_AA)
                cv2.drawMarker(left, xy, (255, 255, 255), markerType=cv2.MARKER_CROSS, markerSize=24, thickness=2)

            # Keep the selected YOLO box on top of trajectory and bounce rings.
            if selected is not None:
                _draw_detection_box(
                    left,
                    selected,
                    scale=display_scale,
                    color=(0, 255, 255),
                    masked=False,
                    selected=True,
                )

            _draw_text(left, variant.variant, (16, 30), scale=0.62, color=(255, 255, 255), thickness=2)
            _draw_text(
                left,
                f"frame {frame_index:05d}  traj {len([d for d in variant.detections if int(d['frame_index']) <= frame_index])}/{len(variant.detections)}  bounces {len([b for b in variant.bounces if int(b.get('frame_index', b.get('frame', -1))) <= frame_index])}/{len(variant.bounces)}",
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
                bounces=variant.bounces,
                current_frame=frame_index,
            )

            writer.write(canvas)
            if seq % 500 == 0:
                print(f"rendered {variant.variant} {seq}/{len(image_paths)} frames", flush=True)
    finally:
        writer.release()

    return {"video": str(output_path), "width": out_w, "height": frame_h, "frames": len(image_paths), "fps": fps}


def _variant_summary(variant: VariantRun, *, frames: int, render: dict[str, Any] | None = None) -> dict[str, Any]:
    bounces = variant.bounces
    in_count = sum(1 for b in bounces if bool(b.get("in_court", b.get("type") != "OUT")))
    out_count = len(bounces) - in_count
    summary = {
        "variant": variant.variant,
        "pipeline": variant.pipeline,
        "frames": frames,
        "camera": CAMERA,
        "minimap_flip": MINIMAP_FLIP,
        "raw_boxes": int(variant.stats.get("raw_boxes", 0)),
        "kept_boxes": int(variant.stats.get("kept_boxes", 0)),
        "trajectory_points": int(variant.stats.get("trajectory_points", len(variant.detections))),
        "static_blocked": int(variant.stats.get("static_blocked", 0)),
        "bounce_count": len(bounces),
        "in_count": in_count,
        "out_count": out_count,
        "bounce_frames": [int(b.get("frame_index", b.get("frame", -1))) for b in bounces],
        "stats": variant.stats,
    }
    if render is not None:
        summary["render"] = render
    return summary


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run cam68 YOLO bounce ablation videos.")
    parser.add_argument("--frames-dir", type=Path, default=DEFAULT_FRAMES_DIR)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--homography", type=Path, default=DEFAULT_HOMOGRAPHY)
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--max-frames", type=int, default=1500)
    parser.add_argument("--fps", type=float, default=25.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--display-scale", type=float, default=0.5)
    parser.add_argument("--panel-width", type=int, default=360)
    parser.add_argument("--trail-len", type=int, default=28)
    parser.add_argument("--linger-frames", type=int, default=45)
    parser.add_argument("--progress-every", type=int, default=250)
    parser.add_argument("--skip-render", action="store_true")
    return parser.parse_args()


def run(args: argparse.Namespace) -> dict[str, Any]:
    if not args.frames_dir.exists():
        raise FileNotFoundError(f"frames dir not found: {args.frames_dir}")
    if not args.model.exists():
        raise FileNotFoundError(f"model not found: {args.model}")
    if not args.homography.exists():
        raise FileNotFoundError(f"homography not found: {args.homography}")
    if not args.config.exists():
        raise FileNotFoundError(f"config not found: {args.config}")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = args.out_root / f"cam68_yolo_bounce_ablation_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    image_paths = _load_image_paths(args.frames_dir, args.max_frames)
    homography = HomographyTransformer(str(args.homography), CAMERA)

    print(f"Output: {out_dir}", flush=True)
    print(f"Frames: {len(image_paths)} from {args.frames_dir}", flush=True)

    a0, a1 = _run_track_model(
        image_paths=image_paths,
        model_path=args.model,
        device=args.device,
        conf=args.conf,
        imgsz=args.imgsz,
        homography=homography,
        progress_every=args.progress_every,
    )
    a2 = _run_dashboard_static(
        image_paths=image_paths,
        model_path=args.model,
        device=args.device,
        conf=args.conf,
        imgsz=args.imgsz,
        homography=homography,
        progress_every=args.progress_every,
    )

    a3_path = out_dir / "A3_dashboard_integrated_replay.json"
    a3_bounces, a3_replay = _run_dashboard_integrated_replay(
        detections=a2.detections,
        config_path=args.config,
        homography_path=args.homography,
        model_path=args.model,
        fps=args.fps,
        output_path=a3_path,
    )

    a3_video = VariantRun(
        variant="A3_dashboard_integrated_replay",
        pipeline="Dashboard integrated replay events rendered with A2 selected YOLO boxes; same event source is audited for minimap and 3D push",
        detections=a2.detections,
        per_frame=a2.per_frame,
        bounces=a3_bounces,
        stats={
            **a2.stats,
            "raw_boxes": a2.stats.get("raw_boxes", 0),
            "kept_boxes": a2.stats.get("kept_boxes", 0),
            "trajectory_points": len(a2.detections),
            "static_blocked": a2.stats.get("static_blocked", 0),
            "source_replay": str(a3_path),
            "push_payloads_queued": len(a3_replay.get("ws_queue", [])),
            "minimap_push_frame_lists_match": bool(
                a3_replay.get("checks", {}).get("minimap_and_push_frame_lists_match")
            ),
        },
    )

    a4 = VariantRun(
        variant="A4_dashboard_integrated_minimap_video",
        pipeline="Final dashboard-effect video rendered from A3 dashboard integrated minimap/live_bounces events with A2 selected YOLO boxes",
        detections=a2.detections,
        per_frame=a2.per_frame,
        bounces=a3_bounces,
        stats={
            **a2.stats,
            "raw_boxes": a2.stats.get("raw_boxes", 0),
            "kept_boxes": a2.stats.get("kept_boxes", 0),
            "trajectory_points": len(a2.detections),
            "static_blocked": a2.stats.get("static_blocked", 0),
            "source_replay": str(a3_path),
        },
    )

    renders: dict[str, dict[str, Any]] = {}
    if not args.skip_render:
        render_specs = [
            (a0, out_dir / "A0_yolo_raw_box_trajectory.mp4"),
            (a1, out_dir / "A1_yolo_original_static_trajectory.mp4"),
            (a2, out_dir / "A2_yolo_dashboard_static_trajectory.mp4"),
            (a3_video, out_dir / "A3_dashboard_integrated_replay_video.mp4"),
            (a4, out_dir / "A4_dashboard_integrated_minimap_video.mp4"),
        ]
        for variant, output_path in render_specs:
            renders[variant.variant] = _render_variant_video(
                variant=variant,
                image_paths=image_paths,
                output_path=output_path,
                fps=args.fps,
                display_scale=args.display_scale,
                panel_width=args.panel_width,
                trail_len=args.trail_len,
                linger_frames=args.linger_frames,
            )

    summaries = [
        _variant_summary(a0, frames=len(image_paths), render=renders.get(a0.variant)),
        _variant_summary(a1, frames=len(image_paths), render=renders.get(a1.variant)),
        _variant_summary(a2, frames=len(image_paths), render=renders.get(a2.variant)),
        {
            **_variant_summary(a3_video, frames=len(image_paths), render=renders.get(a3_video.variant)),
            "replay_json": str(a3_path),
            "checks": a3_replay.get("checks", {}),
        },
        _variant_summary(a4, frames=len(image_paths), render=renders.get(a4.variant)),
    ]

    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "frames_dir": str(args.frames_dir),
        "model": str(args.model),
        "homography": str(args.homography),
        "config": str(args.config),
        "max_frames": args.max_frames,
        "frames": len(image_paths),
        "camera": CAMERA,
        "minimap_flip": MINIMAP_FLIP,
        "minimap_style": "bounce_only_green_in_red_hollow_out_latest_highlight",
        "bounce_params": {
            "max_gap": 3,
            "smooth_window": 3,
            "filter_window": 3,
            "angle_thresh": 10.0,
            "momentum_thresh": 15.0,
            "tolerance": 2,
        },
        "variants": summaries,
        "outputs": {
            "out_dir": str(out_dir),
            "A0_video": str(out_dir / "A0_yolo_raw_box_trajectory.mp4"),
            "A1_video": str(out_dir / "A1_yolo_original_static_trajectory.mp4"),
            "A2_video": str(out_dir / "A2_yolo_dashboard_static_trajectory.mp4"),
            "A3_replay_json": str(a3_path),
            "A3_video": str(out_dir / "A3_dashboard_integrated_replay_video.mp4"),
            "A4_video": str(out_dir / "A4_dashboard_integrated_minimap_video.mp4"),
            "summary": str(out_dir / "ablation_summary.json"),
        },
    }
    _write_json(out_dir / "ablation_summary.json", summary)
    print(json.dumps({
        "out_dir": str(out_dir),
        "A0_bounces": len(a0.bounces),
        "A1_bounces": len(a1.bounces),
        "A2_bounces": len(a2.bounces),
        "A3_bounces": len(a3_bounces),
        "summary": str(out_dir / "ablation_summary.json"),
    }, ensure_ascii=False, indent=2), flush=True)
    return summary


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
