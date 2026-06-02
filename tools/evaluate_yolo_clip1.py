"""Evaluate the dashboard YOLO + static filter + single-cam bounce path on clip1.

The input dataset is an image sequence with LabelMe JSON files rather than MP4s.
This script keeps the runtime path intentionally close to the live dashboard:
JPG frame -> OSD mask -> YoloRoadmapDetector -> top kept candidate ->
HomographyTransformer -> detect_single_camera_bounces().
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from app.pipeline.homography import HomographyTransformer
from app.pipeline.inference import YoloRoadmapDetector
from app.pipeline.yolo_bounce_filter import detect_single_camera_bounces


DEFAULT_ROOT = Path(r"D:\tennis-dataset\1001\clip1")
CAM_FOLDERS = {
    "cam66": "cam66_20260307_173403_2min",
    "cam68": "cam68_20260307_173403_2min",
}


def _shape_center(shape: dict[str, Any]) -> tuple[float, float] | None:
    pts = shape.get("points") or []
    if len(pts) < 2:
        return None
    return (float(pts[0][0] + pts[1][0]) / 2.0, float(pts[0][1] + pts[1][1]) / 2.0)


def load_label_detections(folder: Path, cam: str) -> dict[str, Any]:
    homography = HomographyTransformer("src/homography_matrices.json", cam)
    by_kind: dict[str, list[dict[str, Any]]] = {
        "match_ball": [],
        "moving_ball": [],
        "static_ball": [],
        "all_ball": [],
    }
    per_frame: dict[int, dict[str, list[dict[str, Any]]]] = {}
    label_counts: Counter[str] = Counter()
    multi_ball_frames = 0

    for json_path in sorted(folder.glob("*.json")):
        frame_index = int(json_path.stem)
        try:
            data = json.loads(json_path.read_text(encoding="utf-8"))
        except Exception:
            continue

        ball_count = 0
        frame_bucket = per_frame.setdefault(
            frame_index,
            {"match_ball": [], "moving_ball": [], "static_ball": [], "all_ball": []},
        )
        for shape in data.get("shapes", []):
            label = shape.get("label", "")
            label_counts[label] += 1
            if label != "ball":
                continue
            center = _shape_center(shape)
            if center is None:
                continue
            px, py = center
            wx, wy = homography.pixel_to_world(px, py)
            desc = shape.get("description", "") or ""
            item = {
                "frame_index": frame_index,
                "pixel_x": px,
                "pixel_y": py,
                "x": wx,
                "y": wy,
                "confidence": 1.0,
                "description": desc,
            }
            ball_count += 1
            by_kind["all_ball"].append(item)
            frame_bucket["all_ball"].append(item)
            if "is_match_ball=true" in desc:
                by_kind["match_ball"].append(item)
                frame_bucket["match_ball"].append(item)
            if "motion_state=moving" in desc:
                by_kind["moving_ball"].append(item)
                frame_bucket["moving_ball"].append(item)
            if "motion_state=static" in desc:
                by_kind["static_ball"].append(item)
                frame_bucket["static_ball"].append(item)

        if ball_count > 1:
            multi_ball_frames += 1

    annotation_bounces = {
        kind: detect_single_camera_bounces(points, camera_name=cam)
        for kind, points in by_kind.items()
    }
    return {
        "by_kind": by_kind,
        "per_frame": per_frame,
        "annotation_bounces": annotation_bounces,
        "summary": {
            "json_files": len(list(folder.glob("*.json"))),
            "ball_shapes": label_counts.get("ball", 0),
            "ball_frames": len([k for k, v in per_frame.items() if v["all_ball"]]),
            "multi_ball_frames": multi_ball_frames,
            "match_ball_shapes": len(by_kind["match_ball"]),
            "moving_ball_shapes": len(by_kind["moving_ball"]),
            "static_ball_shapes": len(by_kind["static_ball"]),
            "top_labels": label_counts.most_common(10),
            "annotation_bounce_counts": {
                kind: result.get("count", 0)
                for kind, result in annotation_bounces.items()
            },
        },
    }


def nearest_label_distance(
    det: dict[str, Any],
    labels: dict[int, dict[str, list[dict[str, Any]]]],
    kind: str,
) -> float | None:
    points = labels.get(int(det["frame_index"]), {}).get(kind, [])
    if not points:
        return None
    px = float(det["pixel_x"])
    py = float(det["pixel_y"])
    return min(float(np.hypot(px - p["pixel_x"], py - p["pixel_y"])) for p in points)


def distance_stats(distances: list[float | None]) -> dict[str, Any]:
    valid = [float(d) for d in distances if d is not None]
    if not valid:
        return {"label_frames": 0, "within_30px": 0, "within_50px": 0, "median_px": None}
    arr = np.asarray(valid, dtype=float)
    return {
        "label_frames": len(valid),
        "within_30px": int(np.sum(arr <= 30.0)),
        "within_50px": int(np.sum(arr <= 50.0)),
        "within_30px_rate": round(float(np.mean(arr <= 30.0)), 4),
        "within_50px_rate": round(float(np.mean(arr <= 50.0)), 4),
        "median_px": round(float(np.median(arr)), 2),
        "p90_px": round(float(np.percentile(arr, 90)), 2),
    }


def match_bounces(predicted: list[dict[str, Any]], reference: list[dict[str, Any]], tol: int) -> dict[str, Any]:
    ref_frames = [int(b["frame_index"]) for b in reference]
    matched = []
    for b in predicted:
        frame = int(b["frame_index"])
        if not ref_frames:
            continue
        diff = min(abs(frame - r) for r in ref_frames)
        if diff <= tol:
            matched.append(diff)
    return {
        "predicted": len(predicted),
        "reference": len(reference),
        f"matched_{tol}f": len(matched),
        "median_abs_frame_diff": round(float(np.median(matched)), 2) if matched else None,
    }


def run_yolo_on_folder(
    folder: Path,
    cam: str,
    labels: dict[str, Any],
    out_dir: Path,
    max_frames: int | None,
) -> dict[str, Any]:
    homography = HomographyTransformer("src/homography_matrices.json", cam)
    detector = YoloRoadmapDetector(
        model_path="yolo_roadmap/best.pt",
        frames_in=1,
        frames_out=1,
        device="cuda",
        conf=0.25,
    )

    image_paths = sorted(folder.glob("*.jpg"))
    if max_frames is not None:
        image_paths = image_paths[:max_frames]

    detections: list[dict[str, Any]] = []
    candidate_count = 0
    frames_with_kept_candidates = 0
    frames_without_kept_candidates = 0
    start = time.time()

    csv_path = out_dir / f"{cam}_yolo_top_detections.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "frame_index",
                "pixel_x",
                "pixel_y",
                "world_x",
                "world_y",
                "confidence",
                "track_id",
                "pseudo_track_id",
                "static_count",
                "static_status",
                "static_zone_id",
                "dist_match_ball",
                "dist_moving_ball",
            ],
        )
        writer.writeheader()

        for idx, image_path in enumerate(image_paths, start=1):
            frame_index = int(image_path.stem)
            frame = cv2.imread(str(image_path))
            if frame is None:
                frames_without_kept_candidates += 1
                continue
            # Match the live camera pipeline's OSD mask before inference.
            frame = frame.copy()
            frame[0:41, 0:603] = 0
            blobs = detector.infer([frame])[0]
            candidate_count += len(blobs)
            if not blobs:
                frames_without_kept_candidates += 1
            else:
                frames_with_kept_candidates += 1
                top = blobs[0]
                wx, wy = homography.pixel_to_world(top["pixel_x"], top["pixel_y"])
                det = {
                    "camera_name": cam,
                    "frame_index": frame_index,
                    "pixel_x": float(top["pixel_x"]),
                    "pixel_y": float(top["pixel_y"]),
                    "x": wx,
                    "y": wy,
                    "confidence": float(top.get("yolo_conf", top.get("blob_sum", 0.0))),
                    "yolo_conf": float(top.get("yolo_conf", top.get("blob_sum", 0.0))),
                    "track_id": top.get("track_id"),
                    "pseudo_track_id": top.get("pseudo_track_id"),
                    "static_count": top.get("static_count", 0),
                    "static_status": top.get("static_status"),
                    "static_zone_id": top.get("static_zone_id"),
                    "source": top.get("source", "yolo_roadmap"),
                }
                det["dist_match_ball"] = nearest_label_distance(det, labels["per_frame"], "match_ball")
                det["dist_moving_ball"] = nearest_label_distance(det, labels["per_frame"], "moving_ball")
                detections.append(det)
                writer.writerow({
                    "frame_index": det["frame_index"],
                    "pixel_x": round(det["pixel_x"], 3),
                    "pixel_y": round(det["pixel_y"], 3),
                    "world_x": round(det["x"], 5),
                    "world_y": round(det["y"], 5),
                    "confidence": round(det["confidence"], 5),
                    "track_id": det["track_id"],
                    "pseudo_track_id": det["pseudo_track_id"],
                    "static_count": det["static_count"],
                    "static_status": det["static_status"],
                    "static_zone_id": det["static_zone_id"],
                    "dist_match_ball": None if det["dist_match_ball"] is None else round(det["dist_match_ball"], 3),
                    "dist_moving_ball": None if det["dist_moving_ball"] is None else round(det["dist_moving_ball"], 3),
                })

            if idx % 250 == 0:
                elapsed = time.time() - start
                print(
                    f"[{cam}] {idx}/{len(image_paths)} frames, kept={len(detections)}, "
                    f"fps={idx / max(elapsed, 1e-6):.1f}",
                    file=sys.stderr,
                    flush=True,
                )

    yolo_bounces = detect_single_camera_bounces(detections, camera_name=cam)
    match_ref = labels["annotation_bounces"]["match_ball"].get("bounces", [])
    moving_ref = labels["annotation_bounces"]["moving_ball"].get("bounces", [])
    return {
        "folder": str(folder),
        "frames_requested": len(image_paths),
        "frames_with_kept_candidates": frames_with_kept_candidates,
        "frames_without_kept_candidates": frames_without_kept_candidates,
        "candidate_count": candidate_count,
        "top_detection_count": len(detections),
        "top_detection_rate": round(len(detections) / max(1, len(image_paths)), 4),
        "csv": str(csv_path),
        "runtime_seconds": round(time.time() - start, 2),
        "detector_stats": detector.get_runtime_stats(),
        "label_distance_match_ball": distance_stats([d.get("dist_match_ball") for d in detections]),
        "label_distance_moving_ball": distance_stats([d.get("dist_moving_ball") for d in detections]),
        "yolo_bounce_count": yolo_bounces.get("count", 0),
        "yolo_bounces": yolo_bounces.get("bounces", []),
        "match_to_annotation_match_ball_bounces_10f": match_bounces(yolo_bounces.get("bounces", []), match_ref, 10),
        "match_to_annotation_moving_ball_bounces_10f": match_bounces(yolo_bounces.get("bounces", []), moving_ref, 10),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--max-frames", type=int, default=None)
    args = parser.parse_args()

    ts = time.strftime("%Y%m%d_%H%M%S")
    out_dir = args.out_dir or Path("debug_output") / "yolo_clip1_eval" / ts
    out_dir.mkdir(parents=True, exist_ok=True)

    result: dict[str, Any] = {
        "root": str(args.root),
        "out_dir": str(out_dir),
        "max_frames": args.max_frames,
        "pipeline": "JPG -> OSD mask -> YoloRoadmapDetector(static zones) -> top candidate -> Homography -> detect_single_camera_bounces",
        "cameras": {},
    }

    for cam, rel in CAM_FOLDERS.items():
        folder = args.root / rel
        labels = load_label_detections(folder, cam)
        yolo = run_yolo_on_folder(folder, cam, labels, out_dir, args.max_frames)
        result["cameras"][cam] = {
            "annotation_summary": labels["summary"],
            "annotation_match_ball_bounces": labels["annotation_bounces"]["match_ball"].get("bounces", []),
            "annotation_moving_ball_bounces_count": labels["annotation_bounces"]["moving_ball"].get("count", 0),
            "yolo": yolo,
        }

    summary_path = out_dir / "yolo_clip1_summary.json"
    summary_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"summary": str(summary_path), "out_dir": str(out_dir)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
