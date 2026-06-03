"""Process the first 2 minutes of the 075325 dual-camera clip.

Outputs:
  1. LabelMe-style ball-only rectangle annotations for each camera.
  2. A dual-camera tracking preview video with comet trails, minimap bounces,
     and net-crossing speed overlays.

This is intentionally a one-off dataset utility. It reuses the same detector,
homography, Viterbi pairing, smoothing, bounce, and speed helpers used by the
offline dashboard-style renderer, but keeps the label export conservative:
only the final selected match-ball pixels are annotated.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from pathlib import Path

import cv2
import numpy as np

from app.court import DEFAULT_COURT
from app.config import load_config as load_app_config
from app.detection_pairing import pair_by_capture_time
from app.pipeline.homography import HomographyTransformer
from app.pipeline.multi_blob_matcher import MultiBlobMatcher
from app.triangulation import triangulate
from app.orchestrator import _MATCH_WINDOW, Orchestrator
from tools.render_tracking_video import (
    _smooth_2d_for_render,
    build_detector,
    load_config,
    run_detection_multi,
)


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

DISPLAY_SPEED_MIN_KMH = 20.0
DISPLAY_SPEED_MAX_KMH = 100.0


VIDEO66 = Path(r"D:\tennis-dataset\1001\075325\cam66_20260404_075325.mp4")
VIDEO68 = Path(r"D:\tennis-dataset\1001\075325\cam68_20260404_075325.mp4")
OUT_ROOT = Path(r"D:\tennis-dataset\1001\clip3")


def _clip(v: float, lo: float, hi: float) -> float:
    return float(max(lo, min(hi, v)))


def _labelme_shape(cx: float, cy: float, img_w: int, img_h: int, *, box_size: float = 18.0) -> dict:
    half = box_size / 2.0
    x0 = _clip(cx - half, 0, img_w - 1)
    y0 = _clip(cy - half, 0, img_h - 1)
    x1 = _clip(cx + half, 0, img_w - 1)
    y1 = _clip(cy + half, 0, img_h - 1)
    return {
        "attributes": {},
        "description": "visibility=visible; motion_state=moving; is_match_ball=true; source=dashboard_exact",
        "difficult": False,
        "flags": {},
        "group_id": 1,
        "kie_linking": [],
        "label": "ball",
        "points": [[x0, y0], [x1, y0], [x1, y1], [x0, y1]],
        "score": None,
        "shape_type": "rectangle",
    }


def _labelme_doc(frame_idx: int, shapes: list[dict], img_w: int, img_h: int) -> dict:
    return {
        "version": "2.5.4",
        "flags": {"serve": False},
        "shapes": shapes,
        "imagePath": f"{frame_idx:05d}.jpg",
        "imageData": None,
        "imageHeight": img_h,
        "imageWidth": img_w,
    }


def export_labelme_frames(
    video_path: Path,
    out_dir: Path,
    detections: dict[int, tuple[float, float, float]],
    n_frames: int,
) -> tuple[int, int]:
    """Write NNNNN.jpg + NNNNN.json for every frame."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {video_path}")

    img_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    img_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out_dir.mkdir(parents=True, exist_ok=True)

    written = 0
    labeled = 0
    for fi in range(n_frames):
        ret, frame = cap.read()
        if not ret:
            break

        stem = f"{fi:05d}"
        cv2.imwrite(str(out_dir / f"{stem}.jpg"), frame, [cv2.IMWRITE_JPEG_QUALITY, 88])

        shapes: list[dict] = []
        if fi in detections:
            px, py, _conf = detections[fi]
            shapes.append(_labelme_shape(px, py, img_w, img_h))
            labeled += 1

        with (out_dir / f"{stem}.json").open("w", encoding="utf-8") as f:
            json.dump(_labelme_doc(fi, shapes, img_w, img_h), f, ensure_ascii=False, indent=2)

        written += 1
        if written % 500 == 0:
            log.info("  %s: wrote %d/%d frames (%d labeled)", out_dir.name, written, n_frames, labeled)

    cap.release()
    return written, labeled


def _make_live_detection(
    cam_name: str,
    frame_idx: int,
    blobs: list[dict],
    homography: HomographyTransformer,
    fps: float,
) -> dict | None:
    """Convert TrackNet blobs into the same dict shape emitted by camera_pipeline."""
    if not blobs:
        return None

    candidates = []
    for b in blobs:
        px = float(b["pixel_x"])
        py = float(b["pixel_y"])
        wx, wy = homography.pixel_to_world(px, py)
        blob_sum = float(b.get("blob_sum", b.get("confidence", 0.0)) or 0.0)
        candidates.append({
            "x": wx,
            "y": wy,
            "world_x": wx,
            "world_y": wy,
            "pixel_x": px,
            "pixel_y": py,
            "blob_sum": blob_sum,
        })

    top = candidates[0]
    capture_ts = frame_idx / fps
    return {
        "camera_name": cam_name,
        "x": top["world_x"],
        "y": top["world_y"],
        "world_x": top["world_x"],
        "world_y": top["world_y"],
        "pixel_x": top["pixel_x"],
        "pixel_y": top["pixel_y"],
        "confidence": top["blob_sum"],
        "blob_sum": top["blob_sum"],
        "timestamp": capture_ts,
        "capture_ts": capture_ts,
        "frame_index": frame_idx,
        "candidates": candidates,
    }


def _feed_dashboard_pair(
    orch: Orchestrator,
    d1: dict,
    d2: dict,
    cam_names: tuple[str, str],
    cam_positions: dict[str, list[float]],
    smoothed_3d: dict[int, tuple[float, float, float]],
    det66: dict[int, tuple[float, float, float]],
    det68: dict[int, tuple[float, float, float]],
    points_3d: dict[int, tuple[float, float, float]],
) -> None:
    """Run one paired detection through the dashboard realtime path."""
    pair_id = (
        d1.get("capture_ts", d1["timestamp"]),
        d2.get("capture_ts", d2["timestamp"]),
    )
    if pair_id == orch._last_tri_pair:
        return

    blob_sum1 = d1.get("blob_sum", d1.get("confidence", 1.0))
    blob_sum2 = d2.get("blob_sum", d2.get("confidence", 1.0))
    avg_conf = (blob_sum1 + blob_sum2) / 2
    orch._conf_history.append(avg_conf)
    if len(orch._conf_history) > 500:
        orch._conf_history = orch._conf_history[-500:]
    if len(orch._conf_history) >= 50 and len(orch._conf_history) % 50 == 0:
        sorted_h = sorted(orch._conf_history)
        orch._conf_threshold = sorted_h[int(len(sorted_h) * orch._conf_percentile / 100)]
    if avg_conf < orch._conf_threshold:
        return

    x = y = z = None
    match = None
    if orch._live_matcher and "candidates" in d1 and "candidates" in d2:
        match = orch._live_matcher.match(d1, d2)
        if match is not None:
            x, y, z = match["x"], match["y"], match["z"]

    if x is None:
        x, y, z = triangulate(
            (d1["x"], d1["y"]),
            (d2["x"], d2["y"]),
            cam_positions[cam_names[0]],
            cam_positions[cam_names[1]],
        )

    orch._last_tri_pair = pair_id

    cap_ts1 = d1.get("capture_ts", d1["timestamp"])
    cap_ts2 = d2.get("capture_ts", d2["timestamp"])
    capture_ts = min(cap_ts1, cap_ts2)
    now = capture_ts
    fi = max(d1.get("frame_index", 0), d2.get("frame_index", 0))
    pt = {"x": x, "y": y, "z": z, "timestamp": now,
          "capture_ts": capture_ts, "frame_index": fi}
    points_3d[int(fi)] = (float(x), float(y), float(z))

    if orch._prev_3d is not None:
        gap_s = capture_ts - orch._prev_3d.get("capture_ts", capture_ts)
        frame_gap = fi - orch._prev_3d.get("frame_index", fi)
        if gap_s <= 0 or gap_s > orch._SPEED_MAX_GAP_S or frame_gap > orch._SPEED_MAX_FRAME_GAP:
            orch._speed_points.clear()
            orch._speed_buffer.clear()
    orch._speed_points.append(pt)

    if orch._prev_3d is not None:
        speed_kmh_fit = orch._estimate_speed_kmh_from_window()
        if speed_kmh_fit is not None:
            orch._speed_buffer.append(speed_kmh_fit)

        prev_y = orch._prev_3d["y"]
        curr_y = y
        if orch._net_crossing_enabled and (
            (prev_y < orch._NET_Y and curr_y >= orch._NET_Y)
            or (prev_y > orch._NET_Y and curr_y <= orch._NET_Y)
        ):
            if len(orch._speed_buffer) >= 1:
                speed_kmh = float(np.median(list(orch._speed_buffer)))
                if orch._SPEED_MIN <= speed_kmh <= orch._SPEED_MAX:
                    direction = "near_to_far" if curr_y > prev_y else "far_to_near"
                    crossing = {
                        "speed_kmh": int(round(speed_kmh)),
                        "direction": direction,
                        "timestamp": now,
                        "frame": int(fi),
                        "x": x,
                        "y": y,
                        "z": z,
                    }
                    orch._latest_net_crossing = crossing
                    orch._net_crossings.append(crossing)
                    if len(orch._net_crossings) > 100:
                        orch._net_crossings = orch._net_crossings[-100:]
    orch._prev_3d = pt

    cam_dets = {}
    if match is not None:
        c1w = match.get("cam1_world") or [d1.get("x"), d1.get("y")]
        c2w = match.get("cam2_world") or [d2.get("x"), d2.get("y")]
        c1p = match.get("cam1_pixel") or [d1.get("pixel_x"), d1.get("pixel_y")]
        c2p = match.get("cam2_pixel") or [d2.get("pixel_x"), d2.get("pixel_y")]
        cam_dets[cam_names[0]] = {
            "world_x": c1w[0], "world_y": c1w[1],
            "pixel_x": c1p[0], "pixel_y": c1p[1],
            "yolo_conf": d1.get("yolo_conf"),
            "blob_sum": match.get("cam1_blob_sum", d1.get("blob_sum", 0.0)),
        }
        cam_dets[cam_names[1]] = {
            "world_x": c2w[0], "world_y": c2w[1],
            "pixel_x": c2p[0], "pixel_y": c2p[1],
            "yolo_conf": d2.get("yolo_conf"),
            "blob_sum": match.get("cam2_blob_sum", d2.get("blob_sum", 0.0)),
        }
    else:
        for cname, det in [(cam_names[0], d1), (cam_names[1], d2)]:
            cam_dets[cname] = {
                "world_x": det.get("x"),
                "world_y": det.get("y"),
                "pixel_x": det.get("pixel_x"),
                "pixel_y": det.get("pixel_y"),
                "yolo_conf": det.get("yolo_conf"),
                "blob_sum": det.get("blob_sum", 0.0),
            }

    det66[int(fi)] = (
        float(cam_dets["cam66"]["pixel_x"]),
        float(cam_dets["cam66"]["pixel_y"]),
        float(cam_dets["cam66"].get("blob_sum", 1.0) or 1.0),
    )
    det68[int(fi)] = (
        float(cam_dets["cam68"]["pixel_x"]),
        float(cam_dets["cam68"]["pixel_y"]),
        float(cam_dets["cam68"].get("blob_sum", 1.0) or 1.0),
    )

    with orch._analytics_lock:
        tri_smoothed, hbounce = orch._run_live_bounce_detectors_locked(pt, cam_dets)
        if tri_smoothed is not None and tri_smoothed.get("frame_index") is not None:
            smoothed_3d[int(tri_smoothed["frame_index"])] = (
                float(tri_smoothed["x"]),
                float(tri_smoothed["y"]),
                float(tri_smoothed["z"]),
            )

        accepted_bounce = None
        accepted_bd = None
        if hbounce is not None:
            bd = hbounce.to_dict()
            event_capture_ts = bd.get("capture_ts")
            if event_capture_ts is None:
                event_capture_ts = getattr(hbounce, "capture_ts", None)
            if event_capture_ts is None:
                event_capture_ts = capture_ts
            event_capture_ts = float(event_capture_ts)
            bd["capture_ts"] = event_capture_ts
            bd["detect_delay"] = round(now - event_capture_ts, 2)
            accepted_bd = orch._gate_live_bounce_candidate_locked(
                bd,
                now=now,
                match_speed=True,
            )
            if accepted_bd is not None:
                accepted_bounce = hbounce
                orch._record_live_bounce_locked(accepted_bd, debug_source=accepted_bounce)

        orch._rally_tracker.update(pt, accepted_bounce)
        if accepted_bounce is not None and accepted_bd is not None:
            orch._last_bounce_ts = float(accepted_bd.get("timestamp", capture_ts))


def run_dashboard_exact_pipeline(video66: Path, video68: Path, max_frames: int, top_k: int):
    cfg = load_config()
    detector, postproc = build_detector(cfg)

    log.info("=== Detection: cam66 ===")
    multi66, raw66, n66 = run_detection_multi(str(video66), detector, postproc, max_frames, top_k=top_k)

    detector._bg_frame = None
    detector._video_median_computed = False

    log.info("=== Detection: cam68 ===")
    multi68, raw68, n68 = run_detection_multi(str(video68), detector, postproc, max_frames, top_k=top_k)
    n_frames = min(n66, n68, max_frames)

    log.info("=== Dashboard-exact realtime bounce path ===")
    app_cfg = load_app_config()
    orch = Orchestrator(app_cfg)
    orch._ws_enabled = False
    cam_names = ("cam66", "cam68")
    cam_positions = orch._get_camera_positions()
    orch._live_matcher = MultiBlobMatcher(
        cam_positions[cam_names[0]],
        cam_positions[cam_names[1]],
        valid_z_range=(0.0, 8.0),
        fps=25.0,
    )

    homo66 = HomographyTransformer(cfg["homography"]["path"], "cam66")
    homo68 = HomographyTransformer(cfg["homography"]["path"], "cam68")
    fps = 25.0
    q66: list[dict] = []
    q68: list[dict] = []
    det66: dict[int, tuple[float, float, float]] = {}
    det68: dict[int, tuple[float, float, float]] = {}
    points_3d: dict[int, tuple[float, float, float]] = {}
    smoothed_3d: dict[int, tuple[float, float, float]] = {}

    for fi in range(n_frames):
        if fi in multi66:
            det = _make_live_detection("cam66", fi, multi66[fi], homo66, fps)
            if det is not None:
                q66.append(orch._apply_live_candidate_continuity(
                    "cam66",
                    det,
                    max_candidates=orch._LIVE_MATCHER_CANDIDATES,
                ))
        if fi in multi68:
            det = _make_live_detection("cam68", fi, multi68[fi], homo68, fps)
            if det is not None:
                q68.append(orch._apply_live_candidate_continuity(
                    "cam68",
                    det,
                    max_candidates=orch._LIVE_MATCHER_CANDIDATES,
                ))

        if q66 and q68:
            pairing = pair_by_capture_time(q66, q68, match_window=_MATCH_WINDOW)
            q66 = pairing.remaining_first
            q68 = pairing.remaining_second
            for d1, d2 in pairing.pairs:
                _feed_dashboard_pair(
                    orch,
                    d1,
                    d2,
                    cam_names,
                    cam_positions,
                    smoothed_3d,
                    det66,
                    det68,
                    points_3d,
                )

        if fi and fi % 500 == 0:
            log.info(
                "  dashboard feed %d/%d frames: 3d=%d bounces=%d",
                fi,
                n_frames,
                len(points_3d),
                len(orch._live_bounces),
            )

    bounces = list(orch._live_bounces)
    raw_net_crossings = [dict(nc) for nc in orch._net_crossings]
    net_crossings = [
        dict(nc) for nc in raw_net_crossings
        if DISPLAY_SPEED_MIN_KMH <= float(nc.get("speed_kmh", 0.0)) <= DISPLAY_SPEED_MAX_KMH
    ]
    log.info(
        "Dashboard-exact result: 3d=%d smoothed=%d bounces=%d net_crossings=%d/%d displayed",
        len(points_3d),
        len(smoothed_3d),
        len(bounces),
        len(net_crossings),
        len(raw_net_crossings),
    )

    det66_render = _smooth_2d_for_render(det66, median_k=5, max_gap=3, interp_gap=4)
    det68_render = _smooth_2d_for_render(det68, median_k=5, max_gap=3, interp_gap=4)

    post_filter_stats = dict(orch._post_filter_stats)
    try:
        orch._manager.shutdown()
    except Exception:
        pass

    return {
        "cfg": cfg,
        "n_frames": n_frames,
        "raw66": raw66,
        "raw68": raw68,
        "det66": det66,
        "det68": det68,
        "det66_render": det66_render,
        "det68_render": det68_render,
        "points_3d": points_3d,
        "smoothed_3d": smoothed_3d,
        "bounces": bounces,
        "net_crossings": net_crossings,
        "raw_net_crossings": raw_net_crossings,
        "post_filter_stats": post_filter_stats,
    }


def _make_court(panel_w: int, panel_h: int):
    img = np.zeros((panel_h, panel_w, 3), dtype=np.uint8)
    img[:] = (22, 52, 34)
    margin = 20
    cw = DEFAULT_COURT

    def w2p(x: float, y: float) -> tuple[int, int]:
        px = int(margin + (x - cw.x_min) / (cw.x_max - cw.x_min) * (panel_w - 2 * margin))
        py = int(margin + (cw.y_max - y) / (cw.y_max - cw.y_min) * (panel_h - 2 * margin))
        return px, py

    tl = w2p(cw.x_min, cw.y_max)
    br = w2p(cw.x_max, cw.y_min)
    cv2.rectangle(img, tl, br, (38, 98, 54), -1)
    cv2.rectangle(img, tl, br, (240, 245, 238), 2, cv2.LINE_AA)
    cv2.line(img, w2p(cw.x_min, cw.net_y), w2p(cw.x_max, cw.net_y), (205, 218, 218), 2, cv2.LINE_AA)
    cv2.line(img, w2p(cw.x_min, cw.service_line_near), w2p(cw.x_max, cw.service_line_near), (190, 205, 198), 1, cv2.LINE_AA)
    cv2.line(img, w2p(cw.x_min, cw.service_line_far), w2p(cw.x_max, cw.service_line_far), (190, 205, 198), 1, cv2.LINE_AA)
    cv2.line(img, w2p(0, cw.service_line_near), w2p(0, cw.service_line_far), (190, 205, 198), 1, cv2.LINE_AA)
    cv2.putText(img, "MINIMAP", (18, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (210, 230, 218), 1, cv2.LINE_AA)
    return img, w2p


def _draw_comet(img: np.ndarray, pts: list[tuple[float, float]], scale: float, color=(52, 234, 255)) -> None:
    if len(pts) < 2:
        return
    pts = pts[-10:]
    n = len(pts)
    for i in range(1, n):
        a = i / max(n - 1, 1)
        p0 = (int(pts[i - 1][0] * scale), int(pts[i - 1][1] * scale))
        p1 = (int(pts[i][0] * scale), int(pts[i][1] * scale))
        c = tuple(int(v * (0.18 + 0.82 * a)) for v in color)
        cv2.line(img, p0, p1, c, max(1, int(1 + 3 * a)), cv2.LINE_AA)


def _draw_ball(img: np.ndarray, px: float, py: float, scale: float, color=(45, 235, 255)) -> None:
    x, y = int(px * scale), int(py * scale)
    h, w = img.shape[:2]
    if not (0 <= x < w and 0 <= y < h):
        return
    overlay = img.copy()
    cv2.circle(overlay, (x, y), 11, color, -1, cv2.LINE_AA)
    cv2.addWeighted(overlay, 0.18, img, 0.82, 0, img)
    cv2.circle(img, (x, y), 7, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.circle(img, (x, y), 3, color, -1, cv2.LINE_AA)


def _nearest_speed(frame: int, net_crossings: list[dict]) -> dict | None:
    active = [nc for nc in net_crossings if 0 <= frame - nc["frame"] <= 55]
    return active[-1] if active else None


def _bounce_frame(bounce: dict) -> int:
    frame = bounce.get("frame_index", bounce.get("frame"))
    if frame is None:
        return -1
    return int(frame)


def render_pretty_video(
    video66: Path,
    video68: Path,
    output: Path,
    n_frames: int,
    det66: dict[int, tuple[float, float, float]],
    det68: dict[int, tuple[float, float, float]],
    smoothed_3d: dict[int, tuple[float, float, float]],
    bounces: list[dict],
    net_crossings: list[dict],
) -> None:
    cap66 = cv2.VideoCapture(str(video66))
    cap68 = cv2.VideoCapture(str(video68))
    fps = cap66.get(cv2.CAP_PROP_FPS) or 25.0
    orig_w = int(cap66.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap66.get(cv2.CAP_PROP_FRAME_HEIGHT))

    canvas_w = 1920
    court_w = 300
    half_w = (canvas_w - court_w) // 2
    scale = half_w / orig_w
    half_h = int(orig_h * scale)
    canvas_h = half_h

    output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (canvas_w, canvas_h),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Cannot open video writer: {output}")

    court_base, w2p = _make_court(court_w, canvas_h)
    bounce_by_frame = {_bounce_frame(b): b for b in bounces if _bounce_frame(b) >= 0}
    shown_bounces: list[dict] = []
    trail66: list[tuple[float, float]] = []
    trail68: list[tuple[float, float]] = []

    for fi in range(n_frames):
        ok66, frame66 = cap66.read()
        ok68, frame68 = cap68.read()
        if not ok66 or not ok68:
            break

        small66 = cv2.resize(frame66, (half_w, half_h), interpolation=cv2.INTER_AREA)
        small68 = cv2.resize(frame68, (half_w, half_h), interpolation=cv2.INTER_AREA)

        if fi in det66:
            px, py, _ = det66[fi]
            if trail66 and math.hypot(px - trail66[-1][0], py - trail66[-1][1]) > 180:
                trail66.clear()
            trail66.append((px, py))
            trail66 = trail66[-10:]
            _draw_comet(small66, trail66, scale)
            _draw_ball(small66, px, py, scale)
        elif trail66 and fi - max([f for f in det66.keys() if f <= fi], default=-9999) > 8:
            trail66.clear()

        if fi in det68:
            px, py, _ = det68[fi]
            if trail68 and math.hypot(px - trail68[-1][0], py - trail68[-1][1]) > 180:
                trail68.clear()
            trail68.append((px, py))
            trail68 = trail68[-10:]
            _draw_comet(small68, trail68, scale, color=(97, 255, 142))
            _draw_ball(small68, px, py, scale, color=(97, 255, 142))
        elif trail68 and fi - max([f for f in det68.keys() if f <= fi], default=-9999) > 8:
            trail68.clear()

        for panel, label in ((small66, "cam66"), (small68, "cam68")):
            cv2.rectangle(panel, (0, 0), (150, 38), (0, 0, 0), -1)
            cv2.putText(panel, label, (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (245, 245, 245), 2, cv2.LINE_AA)

        court = court_base.copy()
        # Fading 3D trajectory on minimap.
        map_frames = [f for f in sorted(smoothed_3d) if fi - 90 <= f <= fi]
        for idx in range(1, len(map_frames)):
            f0, f1 = map_frames[idx - 1], map_frames[idx]
            x0, y0, _z0 = smoothed_3d[f0]
            x1, y1, _z1 = smoothed_3d[f1]
            alpha = idx / max(len(map_frames) - 1, 1)
            color = (int(45 * alpha), int(210 * alpha), int(255 * alpha))
            cv2.line(court, w2p(x0, y0), w2p(x1, y1), color, max(1, int(1 + 2 * alpha)), cv2.LINE_AA)

        if fi in smoothed_3d:
            x, y, z = smoothed_3d[fi]
            p = w2p(x, y)
            cv2.circle(court, p, 5, (0, 255, 255), -1, cv2.LINE_AA)
            cv2.putText(court, f"z {z:.1f}m", (18, canvas_h - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (215, 230, 230), 1, cv2.LINE_AA)

        if fi in bounce_by_frame:
            shown_bounces.append(bounce_by_frame[fi])
        for i, b in enumerate(shown_bounces[-8:]):
            bx = float(b.get("x_homo", b.get("x", 0.0)))
            by = float(b.get("y_homo", b.get("y", 0.0)))
            p = w2p(bx, by)
            age = max(0, fi - _bounce_frame(b))
            pulse = max(0, 18 - age)
            color = (50, 255, 110) if b.get("in_court", True) else (70, 70, 255)
            cv2.circle(court, p, 7, color, -1 if b.get("in_court", True) else 2, cv2.LINE_AA)
            if pulse > 0:
                cv2.circle(court, p, 9 + pulse, color, 1, cv2.LINE_AA)
            cv2.putText(court, str(max(1, len(shown_bounces) - len(shown_bounces[-8:]) + i + 1)), (p[0] + 9, p[1] + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (245, 245, 245), 1, cv2.LINE_AA)

        speed = _nearest_speed(fi, net_crossings)
        if speed:
            txt = f"{speed['speed_kmh']:.0f} km/h"
            cv2.rectangle(small66, (half_w - 190, 10), (half_w - 12, 54), (0, 0, 0), -1)
            cv2.putText(small66, txt, (half_w - 178, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (40, 240, 255), 2, cv2.LINE_AA)

        cv2.putText(small66, f"F{fi:04d}  {fi / fps:05.1f}s", (12, half_h - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1, cv2.LINE_AA)
        cv2.putText(small68, f"Bounces {len(shown_bounces)}", (12, half_h - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1, cv2.LINE_AA)

        canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
        # Display order follows the requested dashboard review layout:
        # cam68 on the left, cam66 in the middle, minimap on the right.
        canvas[:, :half_w] = small68
        canvas[:, half_w:half_w * 2] = small66
        canvas[:, half_w * 2:] = court
        writer.write(canvas)

        if fi and fi % 500 == 0:
            log.info("  rendered %d/%d frames", fi, n_frames)

    cap66.release()
    cap68.release()
    writer.release()
    log.info("Rendered video: %s", output)


def main() -> None:
    parser = argparse.ArgumentParser(description="Process clip3 first 2 minutes")
    parser.add_argument("--video66", type=Path, default=VIDEO66)
    parser.add_argument("--video68", type=Path, default=VIDEO68)
    parser.add_argument("--out-root", type=Path, default=OUT_ROOT)
    parser.add_argument("--seconds", type=float, default=120.0)
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--skip-labelme", action="store_true")
    parser.add_argument("--skip-render", action="store_true")
    args = parser.parse_args()

    cap = cv2.VideoCapture(str(args.video66))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    cap.release()
    max_frames = int(round(args.seconds * fps))

    log.info("Processing first %.1fs = %d frames", args.seconds, max_frames)
    result = run_dashboard_exact_pipeline(args.video66, args.video68, max_frames, args.top_k)
    n_frames = result["n_frames"]

    cam66_dir = args.out_root / "cam66_20260404_075325_2min"
    cam68_dir = args.out_root / "cam68_20260404_075325_2min"
    if not args.skip_labelme:
        log.info("=== Export LabelMe frames ===")
        w66, l66 = export_labelme_frames(args.video66, cam66_dir, result["det66"], n_frames)
        w68, l68 = export_labelme_frames(args.video68, cam68_dir, result["det68"], n_frames)
        log.info("LabelMe cam66: %d frames, %d ball boxes", w66, l66)
        log.info("LabelMe cam68: %d frames, %d ball boxes", w68, l68)

    video_out = args.out_root / "clip3_first2min_tracking_overlay.mp4"
    if not args.skip_render:
        log.info("=== Render tracking overlay video ===")
        render_pretty_video(
            args.video66,
            args.video68,
            video_out,
            n_frames,
            result["det66_render"],
            result["det68_render"],
            result["smoothed_3d"],
            result["bounces"],
            result["net_crossings"],
        )

    summary = {
        "video66": str(args.video66),
        "video68": str(args.video68),
        "frames": n_frames,
        "seconds": n_frames / fps,
        "labelme_cam66": str(cam66_dir),
        "labelme_cam68": str(cam68_dir),
        "overlay_video": str(video_out),
        "bounce_source": "dashboard_exact_orchestrator_live_path",
        "detections_cam66": len(result["det66"]),
        "detections_cam68": len(result["det68"]),
        "points_3d": len(result["points_3d"]),
        "smoothed_3d": len(result["smoothed_3d"]),
        "bounces": result["bounces"],
        "net_crossings": result["net_crossings"],
        "raw_net_crossings_count": len(result["raw_net_crossings"]),
        "display_speed_range_kmh": [DISPLAY_SPEED_MIN_KMH, DISPLAY_SPEED_MAX_KMH],
        "post_filter_stats": result.get("post_filter_stats", {}),
    }
    args.out_root.mkdir(parents=True, exist_ok=True)
    with (args.out_root / "clip3_first2min_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    log.info("Summary: %s", args.out_root / "clip3_first2min_summary.json")


if __name__ == "__main__":
    main()
