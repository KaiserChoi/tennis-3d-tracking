"""Render a single-camera YOLO/static-filter/fuzzy-bounce image sequence.

This helper is for no-GT review clips. It follows the current dashboard
single-camera path:

JPG -> OSD mask -> YoloRoadmapDetector(static zones) -> top candidate ->
HomographyTransformer -> detect_single_camera_bounces -> review video.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from collections import deque
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from app.pipeline.homography import HomographyTransformer
from app.pipeline.inference import YoloRoadmapDetector
from app.pipeline.yolo_bounce_filter import (
    COURT_Y_MAX,
    COURT_Y_MIN,
    SINGLES_X_MAX,
    SINGLES_X_MIN,
    detect_single_camera_bounces,
)


PYTHON_CAM68_DEFAULT = Path(r"D:\tennis-dataset\1001\clip11\cam68_20260404_075325_2min")


def _draw_text(
    image: np.ndarray,
    text: str,
    org: tuple[int, int],
    *,
    scale: float = 0.58,
    color: tuple[int, int, int] = (245, 245, 245),
    thickness: int = 1,
) -> None:
    cv2.putText(image, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thickness + 3, cv2.LINE_AA)
    cv2.putText(image, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def _safe_frame_index(path: Path) -> int:
    try:
        return int(path.stem)
    except ValueError:
        return -1


def _stats_delta(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "raw_detections",
        "kept_detections",
        "static_blocked",
        "motion_released",
        "fail_open_kept",
        "untracked_kept",
        "pseudo_tracked",
    ]
    out = {
        key: int(after.get(key, 0)) - int(before.get(key, 0))
        for key in keys
    }
    out["active_static_zones"] = int(after.get("active_static_zones", 0))
    out["static_starvation"] = int(after.get("static_starvation", 0))
    out["static_fail_open_remaining"] = int(after.get("static_fail_open_remaining", 0))
    out["zones"] = after.get("zones", [])
    return out


def _make_video_writer(path: Path, fps: float, size: tuple[int, int]) -> cv2.VideoWriter:
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    if not writer.isOpened():
        raise RuntimeError(f"Could not open VideoWriter: {path}")
    return writer


def _world_to_minimap(
    x: float,
    y: float,
    *,
    origin: tuple[int, int],
    size: tuple[int, int],
    pad: int = 18,
) -> tuple[int, int]:
    ox, oy = origin
    mw, mh = size
    x_min = -10.97 / 2.0
    x_max = 10.97 / 2.0
    y_min = COURT_Y_MIN
    y_max = COURT_Y_MAX
    sx = (mw - pad * 2) / (x_max - x_min)
    sy = (mh - pad * 2) / (y_max - y_min)
    s = min(sx, sy)
    court_w = (x_max - x_min) * s
    court_h = (y_max - y_min) * s
    left = ox + (mw - court_w) / 2.0
    top = oy + (mh - court_h) / 2.0
    px = int(round(left + (x - x_min) * s))
    py = int(round(top + (y_max - y) * s))
    return px, py


def _draw_minimap_base(
    image: np.ndarray,
    *,
    origin: tuple[int, int],
    size: tuple[int, int],
) -> None:
    ox, oy = origin
    mw, mh = size
    cv2.rectangle(image, (ox, oy), (ox + mw, oy + mh), (24, 50, 32), -1)
    cv2.rectangle(image, (ox, oy), (ox + mw, oy + mh), (70, 110, 80), 1)

    def pt(x: float, y: float) -> tuple[int, int]:
        return _world_to_minimap(x, y, origin=origin, size=size)

    white = (230, 245, 230)
    line = 1
    doubles_x_min = -10.97 / 2.0
    doubles_x_max = 10.97 / 2.0
    for xmin, xmax in ((doubles_x_min, doubles_x_max), (SINGLES_X_MIN, SINGLES_X_MAX)):
        cv2.rectangle(image, pt(xmin, COURT_Y_MAX), pt(xmax, COURT_Y_MIN), white, line, cv2.LINE_AA)
    cv2.line(image, pt(doubles_x_min, 0.0), pt(doubles_x_max, 0.0), (180, 220, 180), line, cv2.LINE_AA)
    for sy in (-6.40, 6.40):
        cv2.line(image, pt(SINGLES_X_MIN, sy), pt(SINGLES_X_MAX, sy), white, line, cv2.LINE_AA)
    cv2.line(image, pt(0.0, -6.40), pt(0.0, 6.40), white, line, cv2.LINE_AA)


def _draw_minimap_points(
    image: np.ndarray,
    *,
    origin: tuple[int, int],
    size: tuple[int, int],
    current_det: dict[str, Any] | None,
    trail_world: deque[tuple[float, float]],
    bounces: list[dict[str, Any]],
    current_frame: int,
) -> None:
    pts = list(trail_world)
    if len(pts) >= 2:
        for i in range(1, len(pts)):
            alpha = i / max(1, len(pts) - 1)
            color = (0, int(130 + 90 * alpha), int(200 + 45 * alpha))
            p0 = _world_to_minimap(pts[i - 1][0], pts[i - 1][1], origin=origin, size=size)
            p1 = _world_to_minimap(pts[i][0], pts[i][1], origin=origin, size=size)
            cv2.line(image, p0, p1, color, 2, cv2.LINE_AA)

    for bd in bounces:
        if int(bd.get("frame_index", -1)) > current_frame:
            continue
        p = _world_to_minimap(float(bd["x"]), float(bd["y"]), origin=origin, size=size)
        color = (255, 0, 255) if bd.get("in_court") else (60, 140, 255)
        cv2.circle(image, p, 5, color, -1, cv2.LINE_AA)
        cv2.circle(image, p, 9, (255, 255, 255), 1, cv2.LINE_AA)

    if current_det is not None:
        p = _world_to_minimap(float(current_det["x"]), float(current_det["y"]), origin=origin, size=size)
        cv2.circle(image, p, 4, (0, 255, 255), -1, cv2.LINE_AA)


def _draw_trail(image: np.ndarray, trail_px: deque[tuple[int, int]]) -> None:
    pts = list(trail_px)
    if len(pts) < 2:
        return
    for i in range(1, len(pts)):
        alpha = i / max(1, len(pts) - 1)
        color = (0, int(110 + 120 * alpha), 255)
        cv2.line(image, pts[i - 1], pts[i], color, 2, cv2.LINE_AA)


def _draw_zones(
    image: np.ndarray,
    stats: dict[str, Any] | None,
    *,
    scale: float,
) -> None:
    if not stats:
        return
    for zone in stats.get("zones", []):
        x = int(round(float(zone["x"]) * scale))
        y = int(round(float(zone["y"]) * scale))
        r = int(round(float(zone["radius"]) * scale))
        cv2.circle(image, (x, y), max(4, r), (80, 80, 255), 1, cv2.LINE_AA)
        _draw_text(image, f"S{zone.get('id')}", (x + 6, y - 6), scale=0.42, color=(150, 180, 255))


def _draw_bounce_marker(
    image: np.ndarray,
    xy: tuple[int, int],
    label: str,
    *,
    color: tuple[int, int, int] = (255, 0, 255),
) -> None:
    cv2.circle(image, xy, 22, color, 4, cv2.LINE_AA)
    cv2.drawMarker(image, xy, (255, 255, 255), markerType=cv2.MARKER_CROSS, markerSize=30, thickness=2)
    _draw_text(image, label, (xy[0] + 20, max(28, xy[1] - 18)), scale=0.64, color=(255, 215, 255), thickness=2)


def _write_detections_csv(path: Path, detections: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "frame_index",
        "pixel_x",
        "pixel_y",
        "world_x",
        "world_y",
        "confidence",
        "yolo_conf",
        "track_id",
        "pseudo_track_id",
        "static_count",
        "static_status",
        "static_zone_id",
        "bbox",
        "source",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for det in detections:
            writer.writerow({
                "frame_index": det["frame_index"],
                "pixel_x": round(float(det["pixel_x"]), 3),
                "pixel_y": round(float(det["pixel_y"]), 3),
                "world_x": round(float(det["x"]), 5),
                "world_y": round(float(det["y"]), 5),
                "confidence": round(float(det["confidence"]), 5),
                "yolo_conf": round(float(det.get("yolo_conf", det["confidence"])), 5),
                "track_id": det.get("track_id"),
                "pseudo_track_id": det.get("pseudo_track_id"),
                "static_count": det.get("static_count", 0),
                "static_status": det.get("static_status"),
                "static_zone_id": det.get("static_zone_id"),
                "bbox": json.dumps(det.get("bbox", []), separators=(",", ":")),
                "source": det.get("source", "yolo_roadmap"),
            })


def _run_yolo_pass(
    *,
    frames_dir: Path,
    camera: str,
    homography_path: Path,
    model_path: Path,
    device: str,
    conf: float,
    max_frames: int | None,
    progress_every: int,
) -> tuple[list[Path], list[dict[str, Any]], dict[int, dict[str, Any]], dict[str, Any], float]:
    image_paths = sorted(frames_dir.glob("*.jpg"), key=_safe_frame_index)
    if max_frames is not None:
        image_paths = image_paths[:max_frames]
    if not image_paths:
        raise RuntimeError(f"No JPG frames found in {frames_dir}")

    homography = HomographyTransformer(str(homography_path), camera)
    detector = YoloRoadmapDetector(
        model_path=str(model_path),
        frames_in=1,
        frames_out=1,
        device=device,
        conf=conf,
    )

    detections: list[dict[str, Any]] = []
    per_frame_stats: dict[int, dict[str, Any]] = {}
    start = time.time()

    for seq, image_path in enumerate(image_paths, start=1):
        frame_index = _safe_frame_index(image_path)
        frame = cv2.imread(str(image_path))
        before = detector.get_runtime_stats()
        if frame is None:
            per_frame_stats[frame_index] = {"read_error": True}
            continue

        frame_for_infer = frame.copy()
        frame_for_infer[0:41, 0:603] = 0
        blobs = detector.infer([frame_for_infer])[0]
        after = detector.get_runtime_stats()
        frame_stats = _stats_delta(before, after)
        frame_stats["kept_candidates"] = len(blobs)
        frame_stats["image"] = image_path.name
        per_frame_stats[frame_index] = frame_stats

        if blobs:
            top = blobs[0]
            wx, wy = homography.pixel_to_world(float(top["pixel_x"]), float(top["pixel_y"]))
            detections.append({
                "camera_name": camera,
                "frame_index": frame_index,
                "pixel_x": float(top["pixel_x"]),
                "pixel_y": float(top["pixel_y"]),
                "x": float(wx),
                "y": float(wy),
                "world_x": float(wx),
                "world_y": float(wy),
                "confidence": float(top.get("yolo_conf", top.get("blob_sum", 0.0))),
                "yolo_conf": float(top.get("yolo_conf", top.get("blob_sum", 0.0))),
                "bbox": top.get("bbox", []),
                "track_id": top.get("track_id"),
                "pseudo_track_id": top.get("pseudo_track_id"),
                "static_count": int(top.get("static_count", 0) or 0),
                "static_status": top.get("static_status"),
                "static_zone_id": top.get("static_zone_id"),
                "source": top.get("source", "yolo_roadmap"),
            })

        if progress_every > 0 and seq % progress_every == 0:
            elapsed = time.time() - start
            print(f"processed {seq}/{len(image_paths)} frames, detections={len(detections)}, elapsed={elapsed:.1f}s", flush=True)

    runtime = time.time() - start
    return image_paths, detections, per_frame_stats, detector.get_runtime_stats(), runtime


def _render_review_video(
    *,
    image_paths: list[Path],
    output_path: Path,
    detections: list[dict[str, Any]],
    per_frame_stats: dict[int, dict[str, Any]],
    bounces: list[dict[str, Any]],
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
    out_w = frame_w + panel_width
    if out_w % 2:
        out_w += 1
    if frame_h % 2:
        frame_h += 1

    det_by_frame = {int(det["frame_index"]): det for det in detections}
    bounce_by_frame: dict[int, list[dict[str, Any]]] = {}
    for bd in bounces:
        bounce_by_frame.setdefault(int(bd["frame_index"]), []).append(bd)

    writer = _make_video_writer(output_path, fps, (out_w, frame_h))
    trail_px: deque[tuple[int, int]] = deque(maxlen=trail_len)
    trail_world: deque[tuple[float, float]] = deque(maxlen=trail_len)
    recent_bounces: deque[tuple[int, tuple[int, int], str, bool]] = deque(maxlen=16)

    try:
        for seq, image_path in enumerate(image_paths, start=1):
            frame_index = _safe_frame_index(image_path)
            img = cv2.imread(str(image_path))
            if img is None:
                img = np.zeros((h, w, 3), dtype=np.uint8)
            if display_scale != 1.0:
                left = cv2.resize(img, (frame_w, frame_h), interpolation=cv2.INTER_AREA)
            else:
                left = img.copy()
            if left.shape[0] != frame_h:
                left = cv2.resize(left, (frame_w, frame_h), interpolation=cv2.INTER_AREA)

            stats = per_frame_stats.get(frame_index, {})
            det = det_by_frame.get(frame_index)
            if det is not None:
                px = int(round(float(det["pixel_x"]) * display_scale))
                py = int(round(float(det["pixel_y"]) * display_scale))
                trail_px.append((px, py))
                trail_world.append((float(det["x"]), float(det["y"])))

            _draw_zones(left, stats, scale=display_scale)
            _draw_trail(left, trail_px)

            if det is not None:
                px = int(round(float(det["pixel_x"]) * display_scale))
                py = int(round(float(det["pixel_y"]) * display_scale))
                cv2.circle(left, (px, py), 5, (0, 255, 255), -1, cv2.LINE_AA)
                cv2.circle(left, (px, py), 11, (0, 100, 255), 2, cv2.LINE_AA)
                bbox = det.get("bbox") or []
                if len(bbox) == 4:
                    x1, y1, x2, y2 = [int(round(float(v) * display_scale)) for v in bbox]
                    cv2.rectangle(left, (x1, y1), (x2, y2), (0, 210, 255), 1, cv2.LINE_AA)
                _draw_text(
                    left,
                    f"conf {float(det.get('yolo_conf', 0.0)):.2f} {det.get('static_status') or ''}",
                    (px + 12, min(frame_h - 12, py + 18)),
                    scale=0.48,
                    color=(220, 255, 220),
                )
            else:
                _draw_trail(left, trail_px)

            for bd in bounce_by_frame.get(frame_index, []):
                bx = int(round(float(bd["pixel_x"]) * display_scale))
                by = int(round(float(bd["pixel_y"]) * display_scale))
                label = f"BOUNCE {bd.get('type', '')} #{bd.get('sequence', len(recent_bounces) + 1)}"
                recent_bounces.append((frame_index, (bx, by), label, bool(bd.get("in_court"))))

            for bfi, xy, label, in_court in list(recent_bounces):
                age = frame_index - bfi
                if age < 0 or age > linger_frames:
                    continue
                color = (255, 0, 255) if in_court else (60, 140, 255)
                _draw_bounce_marker(left, xy, label, color=color)

            _draw_text(
                left,
                f"cam68 YOLO static fuzzy | frame {frame_index:05d}/{_safe_frame_index(image_paths[-1]):05d} | bounces {len([b for b in bounces if int(b['frame_index']) <= frame_index])}/{len(bounces)}",
                (18, 32),
                scale=0.62,
                color=(255, 255, 255),
                thickness=2,
            )
            _draw_text(
                left,
                f"raw {stats.get('raw_detections', 0)} kept {stats.get('kept_detections', 0)} blocked {stats.get('static_blocked', 0)} zones {stats.get('active_static_zones', 0)} failOpen {stats.get('static_fail_open_remaining', 0)}",
                (18, 62),
                scale=0.54,
                color=(225, 255, 225),
            )

            canvas = np.zeros((frame_h, out_w, 3), dtype=np.uint8)
            canvas[:, :frame_w] = left
            panel_x = frame_w
            panel = canvas[:, panel_x:]
            panel[:] = (18, 18, 18)
            cv2.rectangle(panel, (0, 0), (panel_width - 1, frame_h - 1), (55, 55, 55), 1)
            _draw_text(panel, "YOLO + Static + Fuzzy", (16, 34), scale=0.64, color=(255, 255, 255), thickness=2)
            _draw_text(panel, f"frame: {frame_index:05d}", (16, 66), color=(220, 220, 220))
            _draw_text(panel, f"detections: {len(detections)}", (16, 92), color=(220, 220, 220))
            _draw_text(panel, f"bounces: {len([b for b in bounces if int(b['frame_index']) <= frame_index])}/{len(bounces)}", (16, 118), color=(255, 210, 255))
            _draw_text(panel, f"raw/kept: {stats.get('raw_detections', 0)}/{stats.get('kept_detections', 0)}", (16, 144), color=(220, 255, 220))
            _draw_text(panel, f"blocked: {stats.get('static_blocked', 0)}", (16, 170), color=(170, 190, 255))
            _draw_text(panel, f"zones: {stats.get('active_static_zones', 0)}", (16, 196), color=(170, 190, 255))

            minimap_origin = (16, 230)
            minimap_size = (max(260, panel_width - 32), min(280, frame_h - 330))
            _draw_text(panel, "Mini Map", (16, minimap_origin[1] - 12), scale=0.56, color=(245, 245, 245), thickness=2)
            _draw_minimap_base(panel, origin=minimap_origin, size=minimap_size)
            _draw_minimap_points(
                panel,
                origin=minimap_origin,
                size=minimap_size,
                current_det=det,
                trail_world=trail_world,
                bounces=bounces,
                current_frame=frame_index,
            )

            y0 = minimap_origin[1] + minimap_size[1] + 30
            _draw_text(panel, "Recent Bounces", (16, y0), scale=0.56, color=(245, 245, 245), thickness=2)
            for i, bd in enumerate([b for b in bounces if int(b["frame_index"]) <= frame_index][-6:]):
                text = f"#{bd.get('sequence')} f{int(bd['frame_index']):05d} {bd.get('type')} a{float(bd.get('angle', 0.0)):.1f}"
                _draw_text(panel, text, (16, y0 + 28 + i * 24), scale=0.48, color=(230, 210, 255))

            writer.write(canvas)
            if seq % 500 == 0:
                print(f"rendered {seq}/{len(image_paths)} frames", flush=True)
    finally:
        writer.release()

    return {
        "width": out_w,
        "height": frame_h,
        "frames": len(image_paths),
        "fps": fps,
    }


def _make_contact_sheet(
    *,
    image_paths: list[Path],
    detections: list[dict[str, Any]],
    bounces: list[dict[str, Any]],
    output_path: Path,
    display_scale: float,
    window: int,
) -> bool:
    if not bounces:
        return False
    det_by_frame = {int(det["frame_index"]): det for det in detections}
    path_by_frame = {_safe_frame_index(path): path for path in image_paths}
    tiles: list[np.ndarray] = []
    for bd in bounces[:24]:
        frame_index = int(bd["frame_index"])
        frame_path = path_by_frame.get(frame_index)
        if frame_path is None:
            continue
        img = cv2.imread(str(frame_path))
        if img is None:
            continue
        det = det_by_frame.get(frame_index)
        if det is not None:
            x = int(round(float(det["pixel_x"])))
            y = int(round(float(det["pixel_y"])))
        else:
            x = int(round(float(bd["pixel_x"])))
            y = int(round(float(bd["pixel_y"])))
        h, w = img.shape[:2]
        x1 = max(0, x - window)
        y1 = max(0, y - window)
        x2 = min(w, x + window)
        y2 = min(h, y + window)
        crop = img[y1:y2, x1:x2].copy()
        if crop.size == 0:
            continue
        crop = cv2.resize(crop, (320, 220), interpolation=cv2.INTER_AREA)
        cx = int(round((x - x1) * 320 / max(1, x2 - x1)))
        cy = int(round((y - y1) * 220 / max(1, y2 - y1)))
        cv2.circle(crop, (cx, cy), 20, (255, 0, 255), 3, cv2.LINE_AA)
        cv2.drawMarker(crop, (cx, cy), (255, 255, 255), markerType=cv2.MARKER_CROSS, markerSize=28, thickness=2)
        _draw_text(crop, f"#{bd.get('sequence')} f{frame_index:05d} {bd.get('type')}", (10, 24), scale=0.55, color=(255, 230, 255), thickness=2)
        tiles.append(crop)

    if not tiles:
        return False
    cols = min(4, len(tiles))
    rows = int(np.ceil(len(tiles) / cols))
    sheet = np.zeros((rows * 220, cols * 320, 3), dtype=np.uint8)
    for idx, tile in enumerate(tiles):
        r = idx // cols
        c = idx % cols
        sheet[r * 220:(r + 1) * 220, c * 320:(c + 1) * 320] = tile
    if display_scale != 1.0:
        sheet = cv2.resize(
            sheet,
            (int(round(sheet.shape[1] * display_scale)), int(round(sheet.shape[0] * display_scale))),
            interpolation=cv2.INTER_AREA,
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return bool(cv2.imwrite(str(output_path), sheet))


def run(args: argparse.Namespace) -> dict[str, Any]:
    frames_dir = args.frames_dir
    output = args.output
    summary_path = args.summary
    csv_path = args.csv
    contact_sheet_path = args.contact_sheet
    if output is None:
        output = frames_dir.parent / f"{args.camera}_yolo_static_fuzzy_bounce.mp4"
    if summary_path is None:
        summary_path = output.with_suffix(".summary.json")
    if csv_path is None:
        csv_path = output.with_suffix(".detections.csv")
    if contact_sheet_path is None:
        contact_sheet_path = output.with_suffix(".contact_sheet.jpg")

    image_paths, detections, per_frame_stats, final_stats, yolo_runtime = _run_yolo_pass(
        frames_dir=frames_dir,
        camera=args.camera,
        homography_path=args.homography,
        model_path=args.model,
        device=args.device,
        conf=args.conf,
        max_frames=args.max_frames,
        progress_every=args.progress_every,
    )
    bounce_result = detect_single_camera_bounces(
        detections,
        camera_name=args.camera,
        max_gap=args.max_gap,
        smooth_window=args.smooth_window,
        filter_window=args.filter_window,
        angle_thresh=args.angle_thresh,
        momentum_thresh=args.momentum_thresh,
        tolerance=args.tolerance,
    )
    bounces = bounce_result.get("bounces", [])

    _write_detections_csv(csv_path, detections)
    render_info = _render_review_video(
        image_paths=image_paths,
        output_path=output,
        detections=detections,
        per_frame_stats=per_frame_stats,
        bounces=bounces,
        fps=args.fps,
        display_scale=args.display_scale,
        panel_width=args.panel_width,
        trail_len=args.trail_len,
        linger_frames=args.linger_frames,
    )
    contact_sheet_written = _make_contact_sheet(
        image_paths=image_paths,
        detections=detections,
        bounces=bounces,
        output_path=contact_sheet_path,
        display_scale=1.0,
        window=args.contact_window,
    )

    frames_with_detection = len(detections)
    raw_total = int(final_stats.get("raw_detections", 0))
    kept_total = int(final_stats.get("kept_detections", 0))
    summary = {
        "pipeline": "JPG -> OSD mask -> YoloRoadmapDetector(static zones) -> top candidate -> Homography -> detect_single_camera_bounces -> review video",
        "ground_truth": "not_used",
        "camera": args.camera,
        "frames_dir": str(frames_dir),
        "frames": len(image_paths),
        "fps": args.fps,
        "model": str(args.model),
        "conf": args.conf,
        "outputs": {
            "video": str(output),
            "summary": str(summary_path),
            "detections_csv": str(csv_path),
            "contact_sheet": str(contact_sheet_path) if contact_sheet_written else None,
        },
        "detections": {
            "frames_with_top_detection": frames_with_detection,
            "top_detection_rate": round(frames_with_detection / max(1, len(image_paths)), 4),
            "raw_yolo_detections": raw_total,
            "kept_yolo_detections": kept_total,
            "static_blocked": int(final_stats.get("static_blocked", 0)),
            "static_zones_created": int(final_stats.get("static_zones_created", 0)),
            "static_zones_expired": int(final_stats.get("static_zones_expired", 0)),
            "motion_released": int(final_stats.get("motion_released", 0)),
            "fail_open_kept": int(final_stats.get("fail_open_kept", 0)),
            "active_static_zones_end": int(final_stats.get("active_static_zones", 0)),
        },
        "bounce_result": bounce_result,
        "render": render_info,
        "detector_final_stats": final_stats,
        "runtime_seconds": {
            "yolo_pass": round(yolo_runtime, 2),
        },
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "video": str(output),
        "summary": str(summary_path),
        "detections": frames_with_detection,
        "raw": raw_total,
        "kept": kept_total,
        "static_blocked": int(final_stats.get("static_blocked", 0)),
        "bounces": len(bounces),
    }, ensure_ascii=False, indent=2), flush=True)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render YOLO static-filter fuzzy-bounce output for a single-camera JPG sequence.")
    parser.add_argument("--frames-dir", type=Path, default=PYTHON_CAM68_DEFAULT)
    parser.add_argument("--camera", default="cam68")
    parser.add_argument("--model", type=Path, default=Path("yolo_roadmap/best.pt"))
    parser.add_argument("--homography", type=Path, default=Path("src/homography_matrices.json"))
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--summary", type=Path, default=None)
    parser.add_argument("--csv", type=Path, default=None)
    parser.add_argument("--contact-sheet", type=Path, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--fps", type=float, default=25.0)
    parser.add_argument("--display-scale", type=float, default=0.5)
    parser.add_argument("--panel-width", type=int, default=380)
    parser.add_argument("--trail-len", type=int, default=28)
    parser.add_argument("--linger-frames", type=int, default=45)
    parser.add_argument("--max-gap", type=int, default=3)
    parser.add_argument("--smooth-window", type=int, default=3)
    parser.add_argument("--filter-window", type=int, default=3)
    parser.add_argument("--angle-thresh", type=float, default=10.0)
    parser.add_argument("--momentum-thresh", type=float, default=15.0)
    parser.add_argument("--tolerance", type=int, default=2)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--progress-every", type=int, default=250)
    parser.add_argument("--contact-window", type=int, default=180)
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
