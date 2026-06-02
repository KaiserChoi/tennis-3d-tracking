"""High-recall offline YOLO bounce candidate finder.

Pipeline:

JPG sequence -> raw YOLO boxes -> multi-object pixel tracks ->
track-level static gating -> original fuzzy 2D bounce rule per segment ->
candidate review video/CSV/JSON/contact sheet.

This tool is intentionally offline and high recall. It does not change the
dashboard realtime path.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from ultralytics import YOLO


DEFAULT_FRAMES_DIR = Path(r"D:\tennis-dataset\1001\clip11\cam68_20260404_075325_2min")
DEFAULT_USER_MAIN = Path(
    r"C:\Users\PC\xwechat_files\wxid_79pis7amfn4422_0abe\msg\file\2026-05\main(1).py"
)


@dataclass
class Detection:
    frame_index: int
    det_index: int
    x: float
    y: float
    w: float
    h: float
    confidence: float
    class_id: int | None = None
    yolo_track_id: int | None = None
    track_id: int | None = None
    status: str = "unassigned"
    speed: float = 0.0

    @property
    def bbox(self) -> tuple[float, float, float, float]:
        return (
            self.x - self.w / 2.0,
            self.y - self.h / 2.0,
            self.x + self.w / 2.0,
            self.y + self.h / 2.0,
        )


@dataclass
class Track:
    track_id: int
    detections: list[Detection] = field(default_factory=list)
    finished: bool = False

    @property
    def last(self) -> Detection:
        return self.detections[-1]

    def predict(self, frame_index: int) -> tuple[float, float]:
        if len(self.detections) < 2:
            return self.last.x, self.last.y
        p1 = self.detections[-1]
        p0 = self.detections[-2]
        dt = max(1, p1.frame_index - p0.frame_index)
        gap = max(1, frame_index - p1.frame_index)
        vx = (p1.x - p0.x) / dt
        vy = (p1.y - p0.y) / dt
        return p1.x + vx * gap, p1.y + vy * gap

    def avg_confidence(self) -> float:
        if not self.detections:
            return 0.0
        return float(np.mean([d.confidence for d in self.detections]))

    def score(self) -> float:
        active = [d for d in self.detections if d.status != "static_blocked"]
        return len(active) * 0.05 + self.avg_confidence()


@dataclass
class BounceCandidate:
    sequence: int
    frame_index: int
    pixel_x: float
    pixel_y: float
    track_id: int
    segment_index: int
    confidence: float
    angle: float
    delta_v: float
    y_reversal: bool
    strength: str
    candidate_score: float
    track_score: float
    segment_len: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "sequence": self.sequence,
            "frame_index": self.frame_index,
            "pixel_x": round(self.pixel_x, 2),
            "pixel_y": round(self.pixel_y, 2),
            "track_id": self.track_id,
            "segment_index": self.segment_index,
            "confidence": round(self.confidence, 4),
            "angle": round(self.angle, 2),
            "delta_v": round(self.delta_v, 2),
            "y_reversal": self.y_reversal,
            "strength": self.strength,
            "candidate_score": round(self.candidate_score, 4),
            "track_score": round(self.track_score, 4),
            "segment_len": self.segment_len,
        }


def _safe_frame_index(path: Path) -> int:
    try:
        return int(path.stem)
    except ValueError:
        return -1


def _draw_text(
    image: np.ndarray,
    text: str,
    org: tuple[int, int],
    *,
    scale: float = 0.55,
    color: tuple[int, int, int] = (245, 245, 245),
    thickness: int = 1,
) -> None:
    cv2.putText(image, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thickness + 3, cv2.LINE_AA)
    cv2.putText(image, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def _video_info(video_path: Path) -> dict[str, Any]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    try:
        return {
            "path": str(video_path),
            "frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            "fps": float(cap.get(cv2.CAP_PROP_FPS) or 0.0),
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        }
    finally:
        cap.release()


def _extract_video_frames(
    *,
    video_path: Path,
    frames_dir: Path,
    max_frames: int | None,
    progress_every: int,
    jpeg_quality: int,
) -> tuple[Path, dict[str, Any]]:
    info = _video_info(video_path)
    target_frames = info["frames"] if max_frames is None else min(info["frames"], max_frames)
    manifest_path = frames_dir / "manifest.json"
    existing_frames = sorted(frames_dir.glob("*.jpg")) if frames_dir.exists() else []
    if existing_frames and manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            manifest = {}
        if (
            manifest.get("source_video") == str(video_path)
            and int(manifest.get("frames", -1)) == target_frames
            and len(existing_frames) == target_frames
        ):
            info["extracted_frames_dir"] = str(frames_dir)
            info["extracted_frames"] = target_frames
            info["extraction_reused"] = True
            return frames_dir, info
        raise RuntimeError(
            f"Frame cache already exists but does not match this run: {frames_dir}. "
            "Use a different --extracted-frames-dir."
        )

    frames_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    start = time.time()
    written = 0
    try:
        while written < target_frames:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            frame_path = frames_dir / f"{written:06d}.jpg"
            cv2.imwrite(str(frame_path), frame, [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)])
            written += 1
            if progress_every > 0 and written % progress_every == 0:
                elapsed = time.time() - start
                print(f"extracted {written}/{target_frames} frames, elapsed={elapsed:.1f}s", flush=True)
    finally:
        cap.release()

    manifest = {
        "source_video": str(video_path),
        "frames": written,
        "source_info": info,
        "jpeg_quality": jpeg_quality,
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    info["extracted_frames_dir"] = str(frames_dir)
    info["extracted_frames"] = written
    info["extraction_reused"] = False
    return frames_dir, info


def _track_color(track_id: int) -> tuple[int, int, int]:
    palette = [
        (0, 255, 255),
        (255, 0, 255),
        (0, 210, 120),
        (255, 180, 40),
        (80, 180, 255),
        (180, 120, 255),
        (80, 255, 80),
        (255, 120, 120),
    ]
    return palette[track_id % len(palette)]


def _load_user_bounce_func(path: Path):
    spec = importlib.util.spec_from_file_location("user_original_main_for_bounce", str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import user main file: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    func = getattr(module, "evaluate_bounces_fuzzy", None)
    if not callable(func):
        raise RuntimeError(f"No evaluate_bounces_fuzzy() found in {path}")
    return func


def evaluate_bounces_fuzzy(
    lookup_table: list[tuple[float, float] | None],
    window: int,
    angle_thresh: float,
    momentum_thresh: float,
    tolerance: int = 2,
) -> tuple[dict[int, tuple[float, float]], dict[int, dict[str, Any]]]:
    """Original fuzzy bounce rule: angle AND nearby Y-reversal or momentum."""
    candidate_bounces: set[int] = set()
    frame_stats: dict[int, dict[str, Any]] = {}
    valid_frames = [i for i, pt in enumerate(lookup_table) if pt is not None]

    for i in range(len(valid_frames)):
        curr_idx = valid_frames[i]
        if i < window or i >= len(valid_frames) - window:
            continue

        prev_idx = valid_frames[i - window]
        next_idx = valid_frames[i + window]
        if (curr_idx - prev_idx) > window * 3 or (next_idx - curr_idx) > window * 3:
            continue

        p_prev = lookup_table[prev_idx]
        p_curr = lookup_table[curr_idx]
        p_next = lookup_table[next_idx]
        if p_prev is None or p_curr is None or p_next is None:
            continue

        v_in = np.array([p_curr[0] - p_prev[0], p_curr[1] - p_prev[1]])
        v_out = np.array([p_next[0] - p_curr[0], p_next[1] - p_curr[1]])
        norm_in = np.linalg.norm(v_in)
        norm_out = np.linalg.norm(v_out)

        angle = 0.0
        y_reversal = False
        delta_v = 0.0
        if norm_in > 1e-5 and norm_out > 1e-5:
            cos_theta = np.clip(np.dot(v_in, v_out) / (norm_in * norm_out), -1.0, 1.0)
            angle = float(np.degrees(np.arccos(cos_theta)))
            y_reversal = bool(v_in[1] > 0 and v_out[1] < 0)
            speed_in = norm_in / max(1, curr_idx - prev_idx)
            speed_out = norm_out / max(1, next_idx - curr_idx)
            delta_v = float(abs(speed_in - speed_out))

        frame_stats[curr_idx] = {
            "angle": angle,
            "y_reversal": y_reversal,
            "delta_v": delta_v,
            "angle_ok": angle >= angle_thresh,
            "y_ok": y_reversal,
            "mom_ok": delta_v >= momentum_thresh,
        }

    for curr_idx, stats in frame_stats.items():
        if not stats["angle_ok"]:
            continue
        local_y_ok = False
        local_mom_ok = False
        for j in range(curr_idx - tolerance, curr_idx + tolerance + 1):
            if j in frame_stats:
                local_y_ok = local_y_ok or bool(frame_stats[j]["y_ok"])
                local_mom_ok = local_mom_ok or bool(frame_stats[j]["mom_ok"])
        if local_y_ok or local_mom_ok:
            candidate_bounces.add(curr_idx)

    bounces: dict[int, tuple[float, float]] = {}
    sorted_bounces = sorted(candidate_bounces)
    if not sorted_bounces:
        return bounces, frame_stats

    cluster = [sorted_bounces[0]]
    for idx in sorted_bounces[1:]:
        if idx - cluster[-1] <= window * 2 + tolerance:
            cluster.append(idx)
        else:
            best_idx = max(cluster, key=lambda k: lookup_table[k][1] if lookup_table[k] is not None else -1e9)
            pt = lookup_table[best_idx]
            if pt is not None:
                bounces[best_idx] = pt
            cluster = [idx]

    best_idx = max(cluster, key=lambda k: lookup_table[k][1] if lookup_table[k] is not None else -1e9)
    pt = lookup_table[best_idx]
    if pt is not None:
        bounces[best_idx] = pt
    return bounces, frame_stats


def collect_yolo_detections(
    *,
    image_paths: list[Path],
    model_path: Path,
    conf: float,
    device: str,
    progress_every: int,
) -> tuple[dict[int, list[Detection]], dict[str, Any]]:
    model = YOLO(str(model_path))
    detections_by_frame: dict[int, list[Detection]] = {}
    stats = {
        "frames": len(image_paths),
        "frames_with_detections": 0,
        "raw_detections": 0,
        "max_detections_in_frame": 0,
        "confidence_sum": 0.0,
    }
    start = time.time()

    for seq, image_path in enumerate(image_paths, start=1):
        frame_index = _safe_frame_index(image_path)
        frame = cv2.imread(str(image_path))
        frame_dets: list[Detection] = []
        if frame is not None:
            results = model.predict(frame, conf=conf, device=device, verbose=False)
            result = results[0] if results else None
            if result is not None and result.boxes is not None and len(result.boxes):
                xywh = result.boxes.xywh.cpu().numpy()
                confs = result.boxes.conf.cpu().numpy()
                classes = result.boxes.cls.cpu().numpy() if result.boxes.cls is not None else np.zeros(len(xywh))
                for det_index, (box, score, cls_id) in enumerate(zip(xywh, confs, classes)):
                    x, y, w, h = [float(v) for v in box]
                    frame_dets.append(
                        Detection(
                            frame_index=frame_index,
                            det_index=det_index,
                            x=x,
                            y=y,
                            w=w,
                            h=h,
                            confidence=float(score),
                            class_id=int(cls_id),
                        )
                    )
        detections_by_frame[frame_index] = frame_dets
        stats["raw_detections"] += len(frame_dets)
        stats["max_detections_in_frame"] = max(stats["max_detections_in_frame"], len(frame_dets))
        stats["confidence_sum"] += float(sum(d.confidence for d in frame_dets))
        if frame_dets:
            stats["frames_with_detections"] += 1

        if progress_every > 0 and seq % progress_every == 0:
            elapsed = time.time() - start
            print(
                f"YOLO {seq}/{len(image_paths)} frames, "
                f"frames_with_det={stats['frames_with_detections']}, boxes={stats['raw_detections']}, "
                f"elapsed={elapsed:.1f}s",
                flush=True,
            )

    stats["avg_confidence"] = round(stats["confidence_sum"] / max(1, stats["raw_detections"]), 4)
    stats["detection_frame_rate"] = round(stats["frames_with_detections"] / max(1, len(image_paths)), 4)
    return detections_by_frame, stats


def _link_score(track: Track, det: Detection, *, max_jump_px: float) -> float | None:
    gap = det.frame_index - track.last.frame_index
    if gap <= 0:
        return None
    pred_x, pred_y = track.predict(det.frame_index)
    dist = float(np.hypot(det.x - pred_x, det.y - pred_y))
    if dist > max_jump_px:
        return None

    score = dist
    if det.yolo_track_id is not None and det.yolo_track_id == track.last.yolo_track_id:
        score -= 35.0
    if len(track.detections) >= 2:
        p0 = track.detections[-2]
        p1 = track.detections[-1]
        v0 = np.array([p1.x - p0.x, p1.y - p0.y], dtype=float)
        v1 = np.array([det.x - p1.x, det.y - p1.y], dtype=float)
        n0 = np.linalg.norm(v0)
        n1 = np.linalg.norm(v1)
        if n0 > 1e-5 and n1 > 1e-5:
            cos_theta = float(np.clip(np.dot(v0, v1) / (n0 * n1), -1.0, 1.0))
            angle = math.degrees(math.acos(cos_theta))
            score += min(40.0, angle * 0.25)
    score -= det.confidence * 12.0
    score += max(0, gap - 1) * 8.0
    return score


def build_tracks(
    detections_by_frame: dict[int, list[Detection]],
    *,
    max_gap: int,
    max_jump_px: float,
    min_len: int,
) -> tuple[list[Track], dict[str, Any]]:
    active: list[Track] = []
    finished: list[Track] = []
    next_track_id = 1

    for frame_index in sorted(detections_by_frame):
        frame_dets = detections_by_frame[frame_index]
        used_dets: set[int] = set()
        proposals: list[tuple[float, int, int]] = []

        for ti, track in enumerate(active):
            gap = frame_index - track.last.frame_index
            if gap > max_gap:
                continue
            for di, det in enumerate(frame_dets):
                score = _link_score(track, det, max_jump_px=max_jump_px)
                if score is not None:
                    proposals.append((score, ti, di))

        used_tracks: set[int] = set()
        for _score, ti, di in sorted(proposals, key=lambda item: item[0]):
            if ti in used_tracks or di in used_dets:
                continue
            track = active[ti]
            det = frame_dets[di]
            prev = track.last
            dt = max(1, det.frame_index - prev.frame_index)
            det.speed = float(np.hypot(det.x - prev.x, det.y - prev.y) / dt)
            det.track_id = track.track_id
            track.detections.append(det)
            used_tracks.add(ti)
            used_dets.add(di)

        for di, det in enumerate(frame_dets):
            if di in used_dets:
                continue
            det.track_id = next_track_id
            active.append(Track(track_id=next_track_id, detections=[det]))
            next_track_id += 1

        still_active: list[Track] = []
        for track in active:
            if frame_index - track.last.frame_index > max_gap:
                finished.append(track)
            else:
                still_active.append(track)
        active = still_active

    finished.extend(active)
    tracks = [t for t in finished if len(t.detections) >= min_len]
    dropped = [t for t in finished if len(t.detections) < min_len]
    stats = {
        "tracks_total": len(finished),
        "tracks_kept_min_len": len(tracks),
        "tracks_dropped_short": len(dropped),
        "max_track_len": max((len(t.detections) for t in tracks), default=0),
        "track_points_kept": sum(len(t.detections) for t in tracks),
    }
    return tracks, stats


def apply_track_static_filter(
    tracks: list[Track],
    *,
    static_frames: int,
    static_std_px: float,
    static_zone_radius: float,
    release_distance_px: float,
    release_speed_px: float,
    release_frames: int,
    starvation_frames: int,
    fail_open_frames: int,
) -> dict[str, Any]:
    zones_created = 0
    for track in tracks:
        recent: deque[Detection] = deque(maxlen=static_frames)
        zone: tuple[float, float] | None = None
        release_streak = 0

        for idx, det in enumerate(track.detections):
            det.status = "active"
            recent.append(det)
            if len(track.detections) >= 2 and idx > 0:
                prev = track.detections[idx - 1]
                dt = max(1, det.frame_index - prev.frame_index)
                det.speed = float(np.hypot(det.x - prev.x, det.y - prev.y) / dt)

            if zone is None and len(recent) == static_frames:
                xs = np.array([p.x for p in recent], dtype=float)
                ys = np.array([p.y for p in recent], dtype=float)
                if float(np.std(xs)) <= static_std_px and float(np.std(ys)) <= static_std_px:
                    zone = (float(np.mean(xs)), float(np.mean(ys)))
                    zones_created += 1
                    for p in recent:
                        p.status = "static_blocked"

            if zone is not None:
                dist = float(np.hypot(det.x - zone[0], det.y - zone[1]))
                if dist > release_distance_px or det.speed >= release_speed_px:
                    release_streak += 1
                else:
                    release_streak = 0

                if release_streak >= release_frames:
                    zone = None
                    det.status = "motion_released"
                    recent.clear()
                    recent.append(det)
                    release_streak = 0
                elif dist <= static_zone_radius:
                    det.status = "static_blocked"

    points_by_frame: dict[int, list[Detection]] = defaultdict(list)
    for track in tracks:
        for det in track.detections:
            points_by_frame[det.frame_index].append(det)

    fail_opened = 0
    starvation = 0
    fail_open_until = -1
    for frame_index in sorted(points_by_frame):
        points = points_by_frame[frame_index]
        active = [d for d in points if d.status != "static_blocked"]
        if points and not active:
            starvation += 1
        else:
            starvation = 0
        if starvation >= starvation_frames:
            fail_open_until = frame_index + fail_open_frames
            starvation = 0
        if frame_index <= fail_open_until and points and not active:
            best = max(points, key=lambda d: d.confidence)
            best.status = "fail_open"
            fail_opened += 1

    static_points = sum(1 for t in tracks for d in t.detections if d.status == "static_blocked")
    eligible_points = sum(1 for t in tracks for d in t.detections if d.status != "static_blocked")
    return {
        "static_zones_created": zones_created,
        "static_blocked_points": static_points,
        "eligible_points": eligible_points,
        "fail_open_points": fail_opened,
    }


def _smooth_segment(points: list[Detection], *, max_gap: int, window: int) -> list[Detection]:
    if len(points) < 2:
        return points

    by_frame = {p.frame_index: p for p in points}
    filled: list[Detection] = []
    sorted_points = sorted(points, key=lambda p: p.frame_index)
    for a, b in zip(sorted_points, sorted_points[1:]):
        filled.append(a)
        gap = b.frame_index - a.frame_index
        if 1 < gap <= max_gap + 1:
            for step in range(1, gap):
                t = step / gap
                frame_index = a.frame_index + step
                interp = Detection(
                    frame_index=frame_index,
                    det_index=-1,
                    x=(1 - t) * a.x + t * b.x,
                    y=(1 - t) * a.y + t * b.y,
                    w=(1 - t) * a.w + t * b.w,
                    h=(1 - t) * a.h + t * b.h,
                    confidence=(1 - t) * a.confidence + t * b.confidence,
                    class_id=a.class_id,
                    yolo_track_id=a.yolo_track_id,
                    track_id=a.track_id,
                    status="interpolated",
                )
                by_frame[frame_index] = interp
                filled.append(interp)
    filled.append(sorted_points[-1])
    filled = sorted(filled, key=lambda p: p.frame_index)

    if window <= 1:
        return filled
    half = window // 2
    smoothed: list[Detection] = []
    for i, point in enumerate(filled):
        lo = max(0, i - half)
        hi = min(len(filled), i + half + 1)
        chunk = filled[lo:hi]
        copy = Detection(
            frame_index=point.frame_index,
            det_index=point.det_index,
            x=float(np.mean([p.x for p in chunk])),
            y=float(np.mean([p.y for p in chunk])),
            w=point.w,
            h=point.h,
            confidence=float(np.mean([p.confidence for p in chunk])),
            class_id=point.class_id,
            yolo_track_id=point.yolo_track_id,
            track_id=point.track_id,
            status=point.status,
            speed=point.speed,
        )
        smoothed.append(copy)
    return smoothed


def _split_eligible_segments(track: Track, *, max_gap: int, min_len: int) -> list[list[Detection]]:
    segments: list[list[Detection]] = []
    current: list[Detection] = []
    prev_frame: int | None = None
    for det in sorted(track.detections, key=lambda d: d.frame_index):
        if det.status == "static_blocked":
            if len(current) >= min_len:
                segments.append(current)
            current = []
            prev_frame = None
            continue
        if prev_frame is not None and det.frame_index - prev_frame > max_gap + 1:
            if len(current) >= min_len:
                segments.append(current)
            current = []
        current.append(det)
        prev_frame = det.frame_index
    if len(current) >= min_len:
        segments.append(current)
    return segments


def _candidate_score(
    *,
    angle: float,
    delta_v: float,
    y_reversal: bool,
    confidence: float,
    segment_len: int,
    track_score: float,
) -> float:
    score = 0.0
    score += min(1.0, angle / 45.0) * 0.35
    score += min(1.0, delta_v / 35.0) * 0.22
    score += confidence * 0.18
    score += min(1.0, segment_len / 24.0) * 0.15
    score += min(1.0, track_score / 4.0) * 0.05
    if y_reversal:
        score += 0.05
    return float(score)


def _strength(score: float, angle: float, delta_v: float, y_reversal: bool, segment_len: int) -> str:
    if score >= 0.65:
        return "STRONG"
    if angle >= 28.0 and (delta_v >= 15.0 or y_reversal) and segment_len >= 8:
        return "STRONG"
    return "WEAK"


def find_bounce_candidates(
    tracks: list[Track],
    *,
    bounce_eval_func,
    max_gap: int,
    min_len: int,
    smooth_window: int,
    filter_window: int,
    angle_thresh: float,
    momentum_thresh: float,
    tolerance: int,
    cluster_frames: int,
) -> tuple[list[BounceCandidate], dict[str, Any], dict[int, dict[str, Any]]]:
    raw_candidates: list[BounceCandidate] = []
    frame_stats_by_frame: dict[int, dict[str, Any]] = {}
    segment_count = 0

    for track in tracks:
        segments = _split_eligible_segments(track, max_gap=max_gap, min_len=min_len)
        for segment_index, segment in enumerate(segments, start=1):
            segment_count += 1
            smoothed = _smooth_segment(segment, max_gap=max_gap, window=smooth_window)
            if len(smoothed) < min_len:
                continue
            max_frame = max(p.frame_index for p in smoothed)
            lookup: list[tuple[float, float] | None] = [None] * (max_frame + 1)
            point_by_frame = {p.frame_index: p for p in smoothed}
            for p in smoothed:
                lookup[p.frame_index] = (p.x, p.y)

            bounces, frame_stats = bounce_eval_func(
                lookup,
                filter_window,
                angle_thresh,
                momentum_thresh,
                tolerance,
            )
            for frame_index, stats in frame_stats.items():
                frame_stats_by_frame.setdefault(frame_index, stats)

            track_score = track.score()
            for frame_index, (px, py) in sorted(bounces.items()):
                point = point_by_frame.get(frame_index)
                if point is None:
                    continue
                stats = frame_stats.get(frame_index, {})
                score = _candidate_score(
                    angle=float(stats.get("angle", 0.0)),
                    delta_v=float(stats.get("delta_v", 0.0)),
                    y_reversal=bool(stats.get("y_reversal", False)),
                    confidence=point.confidence,
                    segment_len=len(segment),
                    track_score=track_score,
                )
                raw_candidates.append(
                    BounceCandidate(
                        sequence=0,
                        frame_index=frame_index,
                        pixel_x=float(px),
                        pixel_y=float(py),
                        track_id=track.track_id,
                        segment_index=segment_index,
                        confidence=point.confidence,
                        angle=float(stats.get("angle", 0.0)),
                        delta_v=float(stats.get("delta_v", 0.0)),
                        y_reversal=bool(stats.get("y_reversal", False)),
                        strength=_strength(
                            score,
                            float(stats.get("angle", 0.0)),
                            float(stats.get("delta_v", 0.0)),
                            bool(stats.get("y_reversal", False)),
                            len(segment),
                        ),
                        candidate_score=score,
                        track_score=track_score,
                        segment_len=len(segment),
                    )
                )

    clusters: list[list[BounceCandidate]] = []
    for cand in sorted(raw_candidates, key=lambda c: c.frame_index):
        if not clusters or cand.frame_index - clusters[-1][-1].frame_index > cluster_frames:
            clusters.append([cand])
        else:
            clusters[-1].append(cand)

    deduped: list[BounceCandidate] = []
    for cluster in clusters:
        best = max(cluster, key=lambda c: (c.pixel_y, c.candidate_score, c.track_score, c.confidence))
        deduped.append(best)
    for seq, cand in enumerate(deduped, start=1):
        cand.sequence = seq

    stats = {
        "segments_evaluated": segment_count,
        "raw_candidates_before_cluster": len(raw_candidates),
        "clustered_candidates": len(deduped),
        "strong_candidates": sum(1 for c in deduped if c.strength == "STRONG"),
        "weak_candidates": sum(1 for c in deduped if c.strength == "WEAK"),
    }
    return deduped, stats, frame_stats_by_frame


def write_candidates_csv(path: Path, candidates: list[BounceCandidate]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "sequence",
        "frame_index",
        "pixel_x",
        "pixel_y",
        "track_id",
        "segment_index",
        "strength",
        "candidate_score",
        "track_score",
        "confidence",
        "angle",
        "delta_v",
        "y_reversal",
        "segment_len",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for cand in candidates:
            writer.writerow(cand.to_dict())


def _points_by_frame(tracks: list[Track]) -> dict[int, list[Detection]]:
    out: dict[int, list[Detection]] = defaultdict(list)
    for track in tracks:
        for det in track.detections:
            out[det.frame_index].append(det)
    return out


def render_video(
    *,
    image_paths: list[Path],
    output_path: Path,
    detections_by_frame: dict[int, list[Detection]],
    tracks: list[Track],
    candidates: list[BounceCandidate],
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
    out_w = frame_w + panel_width
    if out_w % 2:
        out_w += 1
    if frame_h % 2:
        frame_h += 1

    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (out_w, frame_h))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open VideoWriter: {output_path}")

    point_frames = _points_by_frame(tracks)
    candidate_tracks = {c.track_id for c in candidates}
    candidates_by_frame: dict[int, list[BounceCandidate]] = defaultdict(list)
    for cand in candidates:
        candidates_by_frame[cand.frame_index].append(cand)
    recent_candidates: deque[BounceCandidate] = deque(maxlen=24)
    total_candidates = len(candidates)
    strong_total = sum(1 for c in candidates if c.strength == "STRONG")
    weak_total = total_candidates - strong_total

    try:
        for seq, image_path in enumerate(image_paths, start=1):
            frame_index = _safe_frame_index(image_path)
            frame = cv2.imread(str(image_path))
            if frame is None:
                frame = np.zeros((h, w, 3), dtype=np.uint8)
            left = cv2.resize(frame, (frame_w, frame_h), interpolation=cv2.INTER_AREA)

            raw_dets = detections_by_frame.get(frame_index, [])
            for det in raw_dets:
                x1, y1, x2, y2 = [int(round(v * display_scale)) for v in det.bbox]
                cv2.rectangle(left, (x1, y1), (x2, y2), (95, 95, 95), 1, cv2.LINE_AA)

            for det in point_frames.get(frame_index, []):
                x1, y1, x2, y2 = [int(round(v * display_scale)) for v in det.bbox]
                px = int(round(det.x * display_scale))
                py = int(round(det.y * display_scale))
                if det.status == "static_blocked":
                    cv2.rectangle(left, (x1, y1), (x2, y2), (80, 80, 130), 1, cv2.LINE_AA)
                    cv2.circle(left, (px, py), 3, (90, 90, 160), -1, cv2.LINE_AA)
                else:
                    color = _track_color(det.track_id or 0)
                    cv2.rectangle(left, (x1, y1), (x2, y2), color, 1, cv2.LINE_AA)
                    cv2.circle(left, (px, py), 4, color, -1, cv2.LINE_AA)

            for track in tracks:
                if track.track_id not in candidate_tracks:
                    continue
                pts = [
                    d for d in track.detections
                    if d.status != "static_blocked" and 0 <= frame_index - d.frame_index <= 30
                ]
                if len(pts) < 2:
                    continue
                pts = sorted(pts, key=lambda p: p.frame_index)
                color = _track_color(track.track_id)
                for a, b in zip(pts, pts[1:]):
                    p0 = (int(round(a.x * display_scale)), int(round(a.y * display_scale)))
                    p1 = (int(round(b.x * display_scale)), int(round(b.y * display_scale)))
                    cv2.line(left, p0, p1, color, 2, cv2.LINE_AA)

            for cand in candidates_by_frame.get(frame_index, []):
                recent_candidates.append(cand)

            for cand in list(recent_candidates):
                age = frame_index - cand.frame_index
                if age < 0 or age > linger_frames:
                    continue
                xy = (int(round(cand.pixel_x * display_scale)), int(round(cand.pixel_y * display_scale)))
                color = (255, 0, 255) if cand.strength == "STRONG" else (0, 255, 255)
                cv2.circle(left, xy, 22 if cand.strength == "STRONG" else 16, color, 3, cv2.LINE_AA)
                cv2.drawMarker(left, xy, (255, 255, 255), markerType=cv2.MARKER_CROSS, markerSize=28, thickness=2)
                _draw_text(
                    left,
                    f"{cand.strength} #{cand.sequence}",
                    (xy[0] + 18, max(28, xy[1] - 18)),
                    scale=0.58,
                    color=color,
                    thickness=2,
                )

            seen_now = [c for c in candidates if c.frame_index <= frame_index]
            _draw_text(
                left,
                f"YOLO high-recall bounce | frame {frame_index:05d} | raw {len(raw_dets)} | candidates {len(seen_now)}/{total_candidates}",
                (18, 34),
                scale=0.62,
                color=(255, 255, 255),
                thickness=2,
            )

            canvas = np.zeros((frame_h, out_w, 3), dtype=np.uint8)
            canvas[:, :frame_w] = left
            panel = canvas[:, frame_w:]
            panel[:] = (18, 18, 18)
            cv2.rectangle(panel, (0, 0), (panel_width - 1, frame_h - 1), (60, 60, 60), 1)
            _draw_text(panel, "High Recall Bounce", (16, 34), scale=0.64, color=(255, 255, 255), thickness=2)
            _draw_text(panel, f"frame: {frame_index:05d}", (16, 66), color=(220, 220, 220))
            _draw_text(panel, f"raw boxes: {len(raw_dets)}", (16, 92), color=(220, 220, 220))
            _draw_text(panel, f"tracked pts: {len(point_frames.get(frame_index, []))}", (16, 118), color=(220, 255, 220))
            _draw_text(panel, f"strong/weak: {strong_total}/{weak_total}", (16, 144), color=(255, 210, 255))
            _draw_text(panel, f"shown: {len(seen_now)}/{total_candidates}", (16, 170), color=(255, 210, 255))
            _draw_text(panel, "Recent", (16, 214), scale=0.58, color=(245, 245, 245), thickness=2)
            for i, cand in enumerate(seen_now[-10:]):
                color = (255, 170, 255) if cand.strength == "STRONG" else (80, 240, 255)
                text = f"#{cand.sequence} f{cand.frame_index:05d} {cand.strength} s{cand.candidate_score:.2f}"
                _draw_text(panel, text, (16, 244 + i * 24), scale=0.46, color=color)

            writer.write(canvas)
            if seq % 500 == 0:
                print(f"rendered {seq}/{len(image_paths)} frames", flush=True)
    finally:
        writer.release()

    return {
        "frames": len(image_paths),
        "fps": fps,
        "width": out_w,
        "height": frame_h,
        "display_scale": display_scale,
    }


def make_contact_sheet(
    *,
    image_paths: list[Path],
    output_path: Path,
    candidates: list[BounceCandidate],
    crop_radius: int,
) -> bool:
    if not candidates:
        return False
    path_by_frame = {_safe_frame_index(path): path for path in image_paths}
    tiles: list[np.ndarray] = []
    for cand in candidates[:32]:
        frame_path = path_by_frame.get(cand.frame_index)
        if frame_path is None:
            continue
        frame = cv2.imread(str(frame_path))
        if frame is None:
            continue
        h, w = frame.shape[:2]
        x = int(round(cand.pixel_x))
        y = int(round(cand.pixel_y))
        x1 = max(0, x - crop_radius)
        y1 = max(0, y - crop_radius)
        x2 = min(w, x + crop_radius)
        y2 = min(h, y + crop_radius)
        crop = frame[y1:y2, x1:x2].copy()
        if crop.size == 0:
            continue
        crop = cv2.resize(crop, (320, 220), interpolation=cv2.INTER_AREA)
        cx = int(round((x - x1) * 320 / max(1, x2 - x1)))
        cy = int(round((y - y1) * 220 / max(1, y2 - y1)))
        color = (255, 0, 255) if cand.strength == "STRONG" else (0, 255, 255)
        cv2.circle(crop, (cx, cy), 20, color, 3, cv2.LINE_AA)
        cv2.drawMarker(crop, (cx, cy), (255, 255, 255), markerType=cv2.MARKER_CROSS, markerSize=26, thickness=2)
        _draw_text(crop, f"#{cand.sequence} f{cand.frame_index:05d} {cand.strength}", (10, 24), scale=0.52, color=color, thickness=2)
        tiles.append(crop)

    if not tiles:
        return False
    cols = min(4, len(tiles))
    rows = int(math.ceil(len(tiles) / cols))
    sheet = np.zeros((rows * 220, cols * 320, 3), dtype=np.uint8)
    for i, tile in enumerate(tiles):
        r = i // cols
        c = i % cols
        sheet[r * 220:(r + 1) * 220, c * 320:(c + 1) * 320] = tile
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return bool(cv2.imwrite(str(output_path), sheet))


def run(args: argparse.Namespace) -> dict[str, Any]:
    output_path = args.output
    source_info: dict[str, Any] = {"type": "frames", "frames_dir": str(args.frames_dir)}
    frames_dir = args.frames_dir
    if args.video is not None:
        if output_path is None:
            output_path = args.video.parent / f"{args.video.stem}_yolo_bounce_candidates_highrecall.mp4"
        extracted_frames_dir = args.extracted_frames_dir
        if extracted_frames_dir is None:
            extracted_frames_dir = output_path.parent / f"{output_path.stem}_frames"
        frames_dir, video_info = _extract_video_frames(
            video_path=args.video,
            frames_dir=extracted_frames_dir,
            max_frames=args.max_frames,
            progress_every=args.progress_every,
            jpeg_quality=args.jpeg_quality,
        )
        source_info = {"type": "video", **video_info}

    image_paths = sorted(frames_dir.glob("*.jpg"), key=_safe_frame_index)
    if args.max_frames is not None:
        image_paths = image_paths[:args.max_frames]
    if not image_paths:
        raise RuntimeError(f"No JPG frames found in {frames_dir}")

    if output_path is None:
        output_path = frames_dir.parent / "cam68_yolo_bounce_candidates_highrecall.mp4"
    summary_path = args.summary or output_path.with_suffix(".summary.json")
    candidates_csv = args.candidates_csv or output_path.with_suffix(".candidates.csv")
    contact_sheet = args.contact_sheet or output_path.with_suffix(".contact_sheet.jpg")
    render_fps = float(args.fps or source_info.get("fps") or 25.0)

    start = time.time()
    bounce_eval_func = _load_user_bounce_func(args.user_main)
    detections_by_frame, yolo_stats = collect_yolo_detections(
        image_paths=image_paths,
        model_path=args.model,
        conf=args.conf,
        device=args.device,
        progress_every=args.progress_every,
    )
    tracks, track_stats = build_tracks(
        detections_by_frame,
        max_gap=args.max_gap,
        max_jump_px=args.max_jump_px,
        min_len=args.min_len,
    )
    static_stats = apply_track_static_filter(
        tracks,
        static_frames=args.static_frames,
        static_std_px=args.static_std_px,
        static_zone_radius=args.static_zone_radius,
        release_distance_px=args.release_distance_px,
        release_speed_px=args.release_speed_px,
        release_frames=args.release_frames,
        starvation_frames=args.starvation_frames,
        fail_open_frames=args.fail_open_frames,
    )
    candidates, candidate_stats, _frame_stats = find_bounce_candidates(
        tracks,
        bounce_eval_func=bounce_eval_func,
        max_gap=args.max_gap,
        min_len=args.min_len,
        smooth_window=args.smooth_window,
        filter_window=args.filter_window,
        angle_thresh=args.angle_thresh,
        momentum_thresh=args.momentum_thresh,
        tolerance=args.tolerance,
        cluster_frames=args.cluster_frames,
    )
    write_candidates_csv(candidates_csv, candidates)
    render_info = render_video(
        image_paths=image_paths,
        output_path=output_path,
        detections_by_frame=detections_by_frame,
        tracks=tracks,
        candidates=candidates,
        fps=render_fps,
        display_scale=args.display_scale,
        panel_width=args.panel_width,
        linger_frames=args.linger_frames,
    )
    contact_written = make_contact_sheet(
        image_paths=image_paths,
        output_path=contact_sheet,
        candidates=candidates,
        crop_radius=args.contact_window,
    )

    summary = {
        "mode": "offline_yolo_track_static_fuzzy_high_recall",
        "source": source_info,
        "frames_dir": str(frames_dir),
        "model": str(args.model),
        "bounce_logic_source": str(args.user_main),
        "bounce_logic": "main(1).py:evaluate_bounces_fuzzy",
        "outputs": {
            "video": str(output_path),
            "summary": str(summary_path),
            "candidates_csv": str(candidates_csv),
            "contact_sheet": str(contact_sheet) if contact_written else None,
        },
        "params": {
            "conf": args.conf,
            "max_gap": args.max_gap,
            "max_jump_px": args.max_jump_px,
            "min_len": args.min_len,
            "static_frames": args.static_frames,
            "static_std_px": args.static_std_px,
            "release_distance_px": args.release_distance_px,
            "release_speed_px": args.release_speed_px,
            "release_frames": args.release_frames,
            "starvation_frames": args.starvation_frames,
            "smooth_window": args.smooth_window,
            "filter_window": args.filter_window,
            "angle_thresh": args.angle_thresh,
            "momentum_thresh": args.momentum_thresh,
            "tolerance": args.tolerance,
            "cluster_frames": args.cluster_frames,
            "fps": render_fps,
        },
        "yolo_stats": yolo_stats,
        "track_stats": track_stats,
        "static_filter_stats": static_stats,
        "candidate_stats": candidate_stats,
        "candidates": [c.to_dict() for c in candidates],
        "render": render_info,
        "runtime_seconds": round(time.time() - start, 2),
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "video": str(output_path),
        "summary": str(summary_path),
        "candidates_csv": str(candidates_csv),
        "contact_sheet": str(contact_sheet) if contact_written else None,
        "raw_frames": yolo_stats["frames_with_detections"],
        "raw_detections": yolo_stats["raw_detections"],
        "track_points": track_stats["track_points_kept"],
        "eligible_points": static_stats["eligible_points"],
        "candidates": candidate_stats["clustered_candidates"],
        "strong": candidate_stats["strong_candidates"],
        "weak": candidate_stats["weak_candidates"],
    }, ensure_ascii=False, indent=2), flush=True)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Find high-recall YOLO bounce candidates offline.")
    parser.add_argument("--video", type=Path, default=None)
    parser.add_argument("--frames-dir", type=Path, default=DEFAULT_FRAMES_DIR)
    parser.add_argument("--extracted-frames-dir", type=Path, default=None)
    parser.add_argument("--model", type=Path, default=Path("yolo_roadmap/best.pt"))
    parser.add_argument("--user-main", type=Path, default=DEFAULT_USER_MAIN)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--summary", type=Path, default=None)
    parser.add_argument("--candidates-csv", type=Path, default=None)
    parser.add_argument("--contact-sheet", type=Path, default=None)
    parser.add_argument("--device", default="0")
    parser.add_argument("--conf", type=float, default=0.20)
    parser.add_argument("--fps", type=float, default=None)
    parser.add_argument("--display-scale", type=float, default=0.5)
    parser.add_argument("--panel-width", type=int, default=420)
    parser.add_argument("--max-gap", type=int, default=5)
    parser.add_argument("--max-jump-px", type=float, default=90.0)
    parser.add_argument("--min-len", type=int, default=6)
    parser.add_argument("--static-frames", type=int, default=20)
    parser.add_argument("--static-std-px", type=float, default=3.0)
    parser.add_argument("--static-zone-radius", type=float, default=24.0)
    parser.add_argument("--release-distance-px", type=float, default=20.0)
    parser.add_argument("--release-speed-px", type=float, default=8.0)
    parser.add_argument("--release-frames", type=int, default=2)
    parser.add_argument("--starvation-frames", type=int, default=60)
    parser.add_argument("--fail-open-frames", type=int, default=10)
    parser.add_argument("--smooth-window", type=int, default=3)
    parser.add_argument("--filter-window", type=int, default=3)
    parser.add_argument("--angle-thresh", type=float, default=10.0)
    parser.add_argument("--momentum-thresh", type=float, default=15.0)
    parser.add_argument("--tolerance", type=int, default=2)
    parser.add_argument("--cluster-frames", type=int, default=10)
    parser.add_argument("--linger-frames", type=int, default=45)
    parser.add_argument("--contact-window", type=int, default=180)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--progress-every", type=int, default=250)
    parser.add_argument("--jpeg-quality", type=int, default=95)
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
