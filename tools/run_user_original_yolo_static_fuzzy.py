"""Run the user's original YOLO static filter + fuzzy bounce workflow.

The source files shared by the user are GUI/video scripts with hard-coded
paths. This runner keeps their core logic while adapting only IO:

- YOLO.track(conf=0.2, persist=True)
- track_id based static filter with MOVE_THRESHOLD=5, STATIC_FRAME_LIMIT=3
- original main(1).py smooth_trajectory() and evaluate_bounces_fuzzy()
- non-interactive MP4 render for an image sequence
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO


DEFAULT_FRAMES_DIR = Path(r"D:\tennis-dataset\1001\clip11\cam68_20260404_075325_2min")
DEFAULT_USER_MAIN = Path(
    r"C:\Users\PC\xwechat_files\wxid_79pis7amfn4422_0abe\msg\file\2026-05\main(1).py"
)


def _safe_frame_index(path: Path) -> int:
    try:
        return int(path.stem)
    except ValueError:
        return -1


def _load_user_main(path: Path):
    spec = importlib.util.spec_from_file_location("user_original_main", str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import user main file: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _draw_text(
    image: np.ndarray,
    text: str,
    org: tuple[int, int],
    *,
    scale: float = 0.5,
    color: tuple[int, int, int] = (255, 255, 255),
    thickness: int = 1,
) -> None:
    cv2.putText(image, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thickness + 3, cv2.LINE_AA)
    cv2.putText(image, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def _run_original_static_filter(
    *,
    frames_dir: Path,
    model_path: Path,
    conf: float,
    device: str,
    move_threshold: float,
    static_frame_limit: int,
    max_frames: int | None,
    progress_every: int,
) -> tuple[list[Path], list[dict[str, Any]], dict[int, list[dict[str, Any]]], dict[str, Any]]:
    image_paths = sorted(frames_dir.glob("*.jpg"), key=_safe_frame_index)
    if max_frames is not None:
        image_paths = image_paths[:max_frames]
    if not image_paths:
        raise RuntimeError(f"No JPG frames found in {frames_dir}")

    model = YOLO(str(model_path))
    track_history: dict[int, list[float]] = {}
    trajectory_rows: list[dict[str, Any]] = []
    per_frame_boxes: dict[int, list[dict[str, Any]]] = {}
    stats = {
        "frames": len(image_paths),
        "raw_detections": 0,
        "active_detections": 0,
        "masked_detections": 0,
        "frames_with_active": 0,
        "frames_without_active": 0,
        "unique_track_ids": 0,
    }

    start = time.time()
    for seq, image_path in enumerate(image_paths, start=1):
        frame_index = _safe_frame_index(image_path)
        frame = cv2.imread(str(image_path))
        frame_boxes: list[dict[str, Any]] = []
        if frame is None:
            per_frame_boxes[frame_index] = frame_boxes
            stats["frames_without_active"] += 1
            continue

        results = model.track(frame, persist=True, conf=conf, device=device, verbose=False)
        result = results[0] if results else None
        active_candidates: list[dict[str, Any]] = []
        if result is not None and result.boxes is not None and result.boxes.id is not None:
            boxes = result.boxes.xywh.cpu().numpy()
            track_ids = result.boxes.id.int().cpu().numpy()
            confs = result.boxes.conf.cpu().numpy() if result.boxes.conf is not None else np.ones(len(boxes))

            for box, track_id_raw, score in zip(boxes, track_ids, confs):
                track_id = int(track_id_raw)
                x, y, w, h = [float(v) for v in box]
                stats["raw_detections"] += 1

                if track_id not in track_history:
                    track_history[track_id] = [x, y, 0.0]

                last_x, last_y, static_count = track_history[track_id]
                distance = float(np.sqrt((x - last_x) ** 2 + (y - last_y) ** 2))
                if distance < move_threshold:
                    static_count += 1
                else:
                    static_count = 0.0

                track_history[track_id] = [x, y, static_count]
                is_active = static_count < static_frame_limit
                item = {
                    "frame_index": frame_index,
                    "track_id": track_id,
                    "x": x,
                    "y": y,
                    "w": w,
                    "h": h,
                    "x1": x - w / 2.0,
                    "y1": y - h / 2.0,
                    "x2": x + w / 2.0,
                    "y2": y + h / 2.0,
                    "confidence": float(score),
                    "distance": distance,
                    "static_count": int(static_count),
                    "active": bool(is_active),
                }
                frame_boxes.append(item)
                if is_active:
                    stats["active_detections"] += 1
                    active_candidates.append(item)
                else:
                    stats["masked_detections"] += 1

        per_frame_boxes[frame_index] = frame_boxes
        if active_candidates:
            stats["frames_with_active"] += 1
            top = max(active_candidates, key=lambda b: float(b["confidence"]))
            trajectory_rows.append({
                "frame_index": frame_index,
                "x": float(top["x"]),
                "y": float(top["y"]),
                "track_id": int(top["track_id"]),
                "confidence": float(top["confidence"]),
                "static_count": int(top["static_count"]),
            })
        else:
            stats["frames_without_active"] += 1

        if progress_every > 0 and seq % progress_every == 0:
            elapsed = time.time() - start
            print(
                f"processed {seq}/{len(image_paths)} frames, active_frames={stats['frames_with_active']}, "
                f"masked={stats['masked_detections']}, elapsed={elapsed:.1f}s",
                flush=True,
            )

    stats["unique_track_ids"] = len(track_history)
    stats["runtime_seconds"] = round(time.time() - start, 2)
    return image_paths, trajectory_rows, per_frame_boxes, stats


def _write_csvs(
    *,
    trajectory_csv: Path,
    boxes_csv: Path,
    trajectory_rows: list[dict[str, Any]],
    per_frame_boxes: dict[int, list[dict[str, Any]]],
) -> None:
    trajectory_csv.parent.mkdir(parents=True, exist_ok=True)
    with trajectory_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["frame_index", "x", "y", "track_id", "confidence", "static_count"],
        )
        writer.writeheader()
        for row in trajectory_rows:
            writer.writerow({
                "frame_index": row["frame_index"],
                "x": round(float(row["x"]), 3),
                "y": round(float(row["y"]), 3),
                "track_id": row["track_id"],
                "confidence": round(float(row["confidence"]), 5),
                "static_count": row["static_count"],
            })

    with boxes_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "frame_index",
                "track_id",
                "x",
                "y",
                "w",
                "h",
                "confidence",
                "distance",
                "static_count",
                "active",
            ],
        )
        writer.writeheader()
        for frame_index in sorted(per_frame_boxes):
            for box in per_frame_boxes[frame_index]:
                writer.writerow({
                    "frame_index": frame_index,
                    "track_id": box["track_id"],
                    "x": round(float(box["x"]), 3),
                    "y": round(float(box["y"]), 3),
                    "w": round(float(box["w"]), 3),
                    "h": round(float(box["h"]), 3),
                    "confidence": round(float(box["confidence"]), 5),
                    "distance": round(float(box["distance"]), 3),
                    "static_count": box["static_count"],
                    "active": box["active"],
                })


def _build_original_lookup(
    *,
    user_main,
    trajectory_rows: list[dict[str, Any]],
    total_frames: int,
    max_gap: int,
    smooth_window: int,
) -> tuple[list[tuple[int, int] | None], pd.DataFrame]:
    if not trajectory_rows:
        return [None] * total_frames, pd.DataFrame(columns=["frame_index", "x", "y"])
    df = pd.DataFrame(trajectory_rows)[["frame_index", "x", "y"]].copy()
    smoothed = user_main.smooth_trajectory(df, max_gap=max_gap, window=smooth_window)
    lookup_table: list[tuple[int, int] | None] = [None] * total_frames
    for _, row in smoothed.dropna(subset=["x", "y"]).iterrows():
        frame_index = int(row["frame_index"])
        if 0 <= frame_index < total_frames:
            lookup_table[frame_index] = (int(float(row["x"])), int(float(row["y"])))
    return lookup_table, smoothed


def _frame_status(
    frame_index: int,
    frame_stats: dict[int, dict[str, Any]],
    bounces_dict: dict[int, tuple[int, int]],
    *,
    angle_thresh: float,
    momentum_thresh: float,
    tolerance: int,
) -> tuple[str, tuple[int, int, int], str, str, str, bool, bool, bool]:
    status_text = ">> NO BALL <<"
    status_color = (0, 0, 255)
    ang_str = "---"
    yrev_str = "---"
    mom_str = "---"
    angle_ok = False
    local_y_ok = False
    local_mom_ok = False

    if frame_index in frame_stats:
        stats = frame_stats[frame_index]
        local_mom_peak = 0.0
        for j in range(frame_index - tolerance, frame_index + tolerance + 1):
            if j in frame_stats:
                local_y_ok = local_y_ok or bool(frame_stats[j]["y_ok"])
                local_mom_peak = max(local_mom_peak, float(frame_stats[j]["delta_v"]))
        local_mom_ok = local_mom_peak >= momentum_thresh
        angle_ok = bool(stats["angle_ok"])
        is_bounce_candidate = angle_ok and (local_y_ok or local_mom_ok)
        is_true_peak = frame_index in bounces_dict

        ang_str = f"{float(stats['angle']):.2f} deg"
        yrev_str = f"{local_y_ok} (Curr: {bool(stats['y_reversal'])})"
        mom_str = f"{local_mom_peak:.1f} px/f"
        if is_true_peak:
            status_text = ">> BOUNCE (NMS PEAK) <<"
            status_color = (0, 255, 255)
        elif is_bounce_candidate:
            status_text = ">> CANDIDATE (WAIT NMS) <<"
            status_color = (0, 165, 255)
        else:
            status_text = ">> NO BOUNCE <<"
            status_color = (0, 0, 255)

    return status_text, status_color, ang_str, yrev_str, mom_str, angle_ok, local_y_ok, local_mom_ok


def _render_original_style_video(
    *,
    image_paths: list[Path],
    output_path: Path,
    per_frame_boxes: dict[int, list[dict[str, Any]]],
    lookup_table: list[tuple[int, int] | None],
    bounces_dict: dict[int, tuple[int, int]],
    frame_stats: dict[int, dict[str, Any]],
    fps: float,
    angle_thresh: float,
    momentum_thresh: float,
    tolerance: int,
) -> dict[str, Any]:
    first = cv2.imread(str(image_paths[0]))
    if first is None:
        raise RuntimeError(f"Cannot read first frame: {image_paths[0]}")
    orig_h, orig_w = first.shape[:2]

    panel_w = 420
    max_screen_w = 1500
    max_screen_h = 700
    max_video_w = max_screen_w - panel_w
    max_video_h = max_screen_h
    scale_factor = min(max_video_w / orig_w, max_video_h / orig_h)
    scale_factor = min(1.0, scale_factor)
    video_disp_w = int(orig_w * scale_factor)
    video_disp_h = int(orig_h * scale_factor)
    total_w = video_disp_w + panel_w
    total_h = max(video_disp_h, 400)
    if total_w % 2:
        total_w += 1
    if total_h % 2:
        total_h += 1

    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (total_w, total_h),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not open VideoWriter: {output_path}")

    color_green = (0, 255, 0)
    color_purple = (255, 0, 255)
    color_yellow = (0, 255, 255)
    total_frames = len(lookup_table)
    bounces_so_far = 0

    def map_pt(pt: tuple[int, int]) -> tuple[int, int]:
        return int(pt[0] * scale_factor), int(pt[1] * scale_factor)

    try:
        for seq, image_path in enumerate(image_paths, start=1):
            frame_index = _safe_frame_index(image_path)
            frame = cv2.imread(str(image_path))
            if frame is None:
                frame = np.zeros((orig_h, orig_w, 3), dtype=np.uint8)
            overlay = frame.copy()

            for box in per_frame_boxes.get(frame_index, []):
                x1, y1, x2, y2 = [int(round(float(box[k]))) for k in ("x1", "y1", "x2", "y2")]
                if box["active"]:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color_green, 2)
                    cv2.putText(
                        frame,
                        f"Ball {box['track_id']}",
                        (x1, max(16, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        color_green,
                        2,
                    )
                else:
                    cv2.rectangle(overlay, (x1, y1), (x2, y2), (100, 100, 100), -1)
                    cv2.putText(
                        frame,
                        "MASKED",
                        (x1, min(orig_h - 8, y2 + 15)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.4,
                        (150, 150, 150),
                        1,
                    )

            alpha = 0.8
            cv2.addWeighted(overlay, 1 - alpha, frame, alpha, 0, frame)
            resized_video = cv2.resize(frame, (video_disp_w, video_disp_h), interpolation=cv2.INTER_AREA)

            display_frame = np.zeros((total_h, total_w, 3), dtype=np.uint8)
            display_frame[0:video_disp_h, 0:video_disp_w] = resized_video

            history_pts = []
            for i in range(max(0, frame_index - 30), frame_index + 1):
                if i < total_frames and lookup_table[i] is not None:
                    history_pts.append(map_pt(lookup_table[i]))
            if len(history_pts) > 1:
                for i in range(1, len(history_pts)):
                    cv2.line(display_frame, history_pts[i - 1], history_pts[i], color_purple, thickness=2)

            for b_idx, b_pos in bounces_dict.items():
                if 0 <= frame_index - b_idx < 15:
                    bp = map_pt(b_pos)
                    cv2.circle(display_frame, bp, radius=12, color=color_yellow, thickness=2)
                    cv2.circle(display_frame, bp, radius=4, color=color_yellow, thickness=-1)
                    _draw_text(display_frame, "BOUNCE!", (bp[0] + 15, bp[1] - 15), scale=0.8, color=color_yellow, thickness=2)

            current_pt = lookup_table[frame_index] if frame_index < total_frames else None
            if current_pt is not None:
                cp = map_pt(current_pt)
                cv2.circle(display_frame, cp, radius=6, color=color_green, thickness=-1)

            panel_x = video_disp_w
            cv2.rectangle(display_frame, (panel_x, 0), (total_w, total_h), (15, 15, 15), -1)
            bounces_so_far = sum(1 for b_idx in bounces_dict if b_idx <= frame_index)

            _draw_text(display_frame, f"[ FUZZY X-RAY : Frame {frame_index} ]", (panel_x + 10, 40), scale=0.6, color=(255, 255, 255), thickness=2)
            _draw_text(display_frame, f"Rule: Angle(0f) & [Y-Rev(+/-{tolerance}f) | Mom(+/-{tolerance}f)]", (panel_x + 10, 70), scale=0.45, color=color_green)
            _draw_text(display_frame, f"Original static: move<5px, static>=3 masked", (panel_x + 10, 94), scale=0.42, color=(180, 230, 180))

            status_text, status_color, ang_str, yrev_str, mom_str, angle_ok, local_y_ok, local_mom_ok = _frame_status(
                frame_index,
                frame_stats,
                bounces_dict,
                angle_thresh=angle_thresh,
                momentum_thresh=momentum_thresh,
                tolerance=tolerance,
            )

            y_base = 130
            line_spacing = 25
            a_col = (255, 255, 255) if angle_ok else color_green
            y_col = (255, 255, 255) if local_y_ok else color_green
            m_col = (255, 255, 255) if local_mom_ok else color_green
            _draw_text(display_frame, f"Current Angle : {ang_str}", (panel_x + 10, y_base), scale=0.5, color=a_col)
            _draw_text(display_frame, f"Local Y-Rev   : {yrev_str}", (panel_x + 10, y_base + line_spacing), scale=0.5, color=y_col)
            _draw_text(display_frame, f"Local Mom Peak: {mom_str}", (panel_x + 10, y_base + line_spacing * 2), scale=0.5, color=m_col)
            _draw_text(display_frame, "-------------------------------------------", (panel_x + 10, y_base + line_spacing * 3), scale=0.5, color=color_green)
            _draw_text(display_frame, f"Angle Thr     : >= {angle_thresh:.1f} deg", (panel_x + 10, y_base + line_spacing * 4), scale=0.5, color=color_green)
            _draw_text(display_frame, f"Momentum Thr  : >= {momentum_thresh:.1f} px/f", (panel_x + 10, y_base + line_spacing * 5), scale=0.5, color=color_green)
            _draw_text(display_frame, f"Sync Tol      : +/- {tolerance} Frames", (panel_x + 10, y_base + line_spacing * 6), scale=0.5, color=color_green)
            _draw_text(display_frame, status_text, (panel_x + 10, y_base + line_spacing * 8), scale=0.6, color=status_color, thickness=2)
            _draw_text(display_frame, f"Bounces       : {bounces_so_far}/{len(bounces_dict)}", (panel_x + 10, y_base + line_spacing * 10), scale=0.55, color=color_yellow, thickness=2)

            writer.write(display_frame)
            if seq % 500 == 0:
                print(f"rendered {seq}/{len(image_paths)} frames", flush=True)
    finally:
        writer.release()

    return {
        "width": total_w,
        "height": total_h,
        "fps": fps,
        "frames": len(image_paths),
        "scale_factor": scale_factor,
    }


def _make_contact_sheet(
    *,
    image_paths: list[Path],
    lookup_table: list[tuple[int, int] | None],
    bounces_dict: dict[int, tuple[int, int]],
    output_path: Path,
    crop_radius: int,
) -> bool:
    if not bounces_dict:
        return False
    path_by_frame = {_safe_frame_index(path): path for path in image_paths}
    tiles: list[np.ndarray] = []
    for seq, frame_index in enumerate(sorted(bounces_dict)[:24], start=1):
        frame_path = path_by_frame.get(frame_index)
        if frame_path is None:
            continue
        img = cv2.imread(str(frame_path))
        if img is None:
            continue
        pt = lookup_table[frame_index] if frame_index < len(lookup_table) else None
        if pt is None:
            pt = bounces_dict[frame_index]
        x, y = int(pt[0]), int(pt[1])
        h, w = img.shape[:2]
        x1 = max(0, x - crop_radius)
        y1 = max(0, y - crop_radius)
        x2 = min(w, x + crop_radius)
        y2 = min(h, y + crop_radius)
        crop = img[y1:y2, x1:x2].copy()
        if crop.size == 0:
            continue
        crop = cv2.resize(crop, (320, 220), interpolation=cv2.INTER_AREA)
        cx = int(round((x - x1) * 320 / max(1, x2 - x1)))
        cy = int(round((y - y1) * 220 / max(1, y2 - y1)))
        cv2.circle(crop, (cx, cy), 20, (0, 255, 255), 3, cv2.LINE_AA)
        cv2.circle(crop, (cx, cy), 4, (0, 255, 255), -1, cv2.LINE_AA)
        _draw_text(crop, f"#{seq} f{frame_index:05d}", (10, 24), scale=0.58, color=(0, 255, 255), thickness=2)
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
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return bool(cv2.imwrite(str(output_path), sheet))


def run(args: argparse.Namespace) -> dict[str, Any]:
    frames_dir = args.frames_dir
    output_path = args.output or frames_dir.parent / "cam68_user_original_static_fuzzy.mp4"
    summary_path = args.summary or output_path.with_suffix(".summary.json")
    trajectory_csv = args.trajectory_csv or output_path.with_suffix(".trajectory.csv")
    boxes_csv = args.boxes_csv or output_path.with_suffix(".boxes.csv")
    smoothed_csv = args.smoothed_csv or output_path.with_suffix(".smoothed.csv")
    contact_sheet = args.contact_sheet or output_path.with_suffix(".contact_sheet.jpg")

    user_main = _load_user_main(args.user_main)
    image_paths, trajectory_rows, per_frame_boxes, static_stats = _run_original_static_filter(
        frames_dir=frames_dir,
        model_path=args.model,
        conf=args.conf,
        device=args.device,
        move_threshold=args.move_threshold,
        static_frame_limit=args.static_frame_limit,
        max_frames=args.max_frames,
        progress_every=args.progress_every,
    )
    _write_csvs(
        trajectory_csv=trajectory_csv,
        boxes_csv=boxes_csv,
        trajectory_rows=trajectory_rows,
        per_frame_boxes=per_frame_boxes,
    )

    total_frames = max(_safe_frame_index(image_paths[-1]) + 1, len(image_paths))
    lookup_table, smoothed = _build_original_lookup(
        user_main=user_main,
        trajectory_rows=trajectory_rows,
        total_frames=total_frames,
        max_gap=args.max_gap,
        smooth_window=args.smooth_window,
    )
    smoothed.to_csv(smoothed_csv, index=False)
    bounces_dict, frame_stats = user_main.evaluate_bounces_fuzzy(
        lookup_table,
        args.filter_window,
        args.angle_thresh,
        args.momentum_thresh,
        args.tolerance,
    )

    render_info = _render_original_style_video(
        image_paths=image_paths,
        output_path=output_path,
        per_frame_boxes=per_frame_boxes,
        lookup_table=lookup_table,
        bounces_dict=bounces_dict,
        frame_stats=frame_stats,
        fps=args.fps,
        angle_thresh=args.angle_thresh,
        momentum_thresh=args.momentum_thresh,
        tolerance=args.tolerance,
    )
    contact_written = _make_contact_sheet(
        image_paths=image_paths,
        lookup_table=lookup_table,
        bounces_dict=bounces_dict,
        output_path=contact_sheet,
        crop_radius=args.contact_window,
    )

    bounces = [
        {
            "sequence": seq,
            "frame_index": int(frame_index),
            "pixel_x": int(pos[0]),
            "pixel_y": int(pos[1]),
            "angle": round(float(frame_stats.get(frame_index, {}).get("angle", 0.0)), 2),
            "delta_v": round(float(frame_stats.get(frame_index, {}).get("delta_v", 0.0)), 2),
            "y_reversal": bool(frame_stats.get(frame_index, {}).get("y_reversal", False)),
        }
        for seq, (frame_index, pos) in enumerate(sorted(bounces_dict.items()), start=1)
    ]
    summary = {
        "mode": "user_original_adapted_io",
        "source_files": {
            "main": str(args.user_main),
            "static_filter_logic": "verify_with_static_filter(1).py core loop: YOLO.track + track_id static_count",
        },
        "adaptations": [
            "hard-coded old video/model paths replaced by command-line paths",
            "OpenCV interactive window replaced by MP4 writer",
            "image sequence used directly instead of first converting to an input video",
            "when multiple active tracks exist in a frame, highest-confidence active track is written to the single trajectory CSV expected by main(1).py",
        ],
        "frames_dir": str(frames_dir),
        "model": str(args.model),
        "outputs": {
            "video": str(output_path),
            "summary": str(summary_path),
            "trajectory_csv": str(trajectory_csv),
            "boxes_csv": str(boxes_csv),
            "smoothed_csv": str(smoothed_csv),
            "contact_sheet": str(contact_sheet) if contact_written else None,
        },
        "params": {
            "conf": args.conf,
            "move_threshold": args.move_threshold,
            "static_frame_limit": args.static_frame_limit,
            "max_gap": args.max_gap,
            "smooth_window": args.smooth_window,
            "filter_window": args.filter_window,
            "angle_thresh": args.angle_thresh,
            "momentum_thresh": args.momentum_thresh,
            "tolerance": args.tolerance,
            "fps": args.fps,
        },
        "static_filter_stats": static_stats,
        "trajectory_points": len(trajectory_rows),
        "smoothed_points": int(smoothed.dropna(subset=["x", "y"]).shape[0]) if not smoothed.empty else 0,
        "bounce_count": len(bounces),
        "bounces": bounces,
        "render": render_info,
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "video": str(output_path),
        "summary": str(summary_path),
        "trajectory_points": len(trajectory_rows),
        "masked": static_stats["masked_detections"],
        "bounces": len(bounces),
    }, ensure_ascii=False, indent=2), flush=True)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run user original YOLO static filter and fuzzy bounce logic.")
    parser.add_argument("--frames-dir", type=Path, default=DEFAULT_FRAMES_DIR)
    parser.add_argument("--model", type=Path, default=Path("yolo_roadmap/best.pt"))
    parser.add_argument("--user-main", type=Path, default=DEFAULT_USER_MAIN)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--summary", type=Path, default=None)
    parser.add_argument("--trajectory-csv", type=Path, default=None)
    parser.add_argument("--boxes-csv", type=Path, default=None)
    parser.add_argument("--smoothed-csv", type=Path, default=None)
    parser.add_argument("--contact-sheet", type=Path, default=None)
    parser.add_argument("--device", default="0")
    parser.add_argument("--conf", type=float, default=0.2)
    parser.add_argument("--fps", type=float, default=25.0)
    parser.add_argument("--move-threshold", type=float, default=5.0)
    parser.add_argument("--static-frame-limit", type=int, default=3)
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
