"""Evaluate bounce landing accuracy in world coordinates (cm).

Compares detected bounce positions against GT annotations transformed
via homography to world coordinates.  Reports mean/median/p90 error in cm,
recall, and precision.

Usage:
    python -m tools.eval_bounce_accuracy [--max-frames 1800]
"""

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────
GT_DIR = "uploads/cam66_20260307_173403_2min"
VIDEO_66 = "uploads/cam66_20260307_173403_2min.mp4"
VIDEO_68 = "uploads/cam68_20260307_173403_2min.mp4"
MAX_FRAMES = 1800
TOLERANCE_FRAMES = 5  # match window for GT↔detected bounce

# Court
SINGLES_X_MIN, SINGLES_X_MAX = 1.37, 6.86
COURT_Y_MIN, COURT_Y_MAX = 0.0, 23.77


def load_config():
    with open("config.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ── Load GT bounces ───────────────────────────────────────────────────────────

def load_gt_bounces(gt_dir: str, max_frames: int) -> list[tuple[int, float, float]]:
    """Load GT bounce annotations: [(frame, pixel_x, pixel_y), ...]."""
    bounces = []
    for fi in range(max_frames):
        fp = os.path.join(gt_dir, f"{fi:05d}.json")
        if not os.path.exists(fp):
            continue
        with open(fp) as f:
            data = json.load(f)
        for shape in data.get("shapes", []):
            desc = (shape.get("description") or "").lower()
            pts = shape.get("points", [])
            if not pts or "match_ball" not in desc or "bounce" not in desc:
                continue
            if shape.get("shape_type") == "rectangle" and len(pts) >= 2:
                px = (pts[0][0] + pts[1][0]) / 2
                py = (pts[0][1] + pts[1][1]) / 2
            else:
                px, py = pts[0][0], pts[0][1]
            bounces.append((fi, px, py))
            break
    return bounces


# ── Transform GT pixel → world ────────────────────────────────────────────────

def pixel_to_world_batch(bounces: list[tuple], homography_path: str, cam_key: str):
    """Transform pixel bounces to world coordinates using homography.

    Returns: [(frame, world_x, world_y), ...]
    """
    with open(homography_path) as f:
        matrices = json.load(f)
    H = np.array(matrices[cam_key]["H_image_to_world"])

    results = []
    for fi, px, py in bounces:
        pt = np.array([px, py, 1.0])
        w = H @ pt
        wx, wy = float(w[0] / w[2]), float(w[1] / w[2])
        results.append((fi, wx, wy))
    return results


# ── Run detection pipeline ────────────────────────────────────────────────────

def run_pipeline(max_frames: int):
    """Run the full pipeline and return detected bounces with world coords.

    Returns: list[dict] with keys: frame, x, y, z, in_court
    """
    from tools.render_tracking_video import (
        build_detector,
        detect_bounces,
        run_detection_multi,
        smooth_trajectory_sg,
        triangulate_multi_blob,
    )

    cfg = load_config()
    detector, postproc = build_detector(cfg)

    # Run per-camera detection
    logger.info("Running cam66 detection...")
    multi66, det66, n66 = run_detection_multi(VIDEO_66, detector, postproc, max_frames, top_k=2)
    logger.info("Running cam68 detection...")
    multi68, det68, n68 = run_detection_multi(VIDEO_68, detector, postproc, max_frames, top_k=2)

    # Triangulate
    logger.info("Triangulating...")
    points_3d, _, stats = triangulate_multi_blob(multi66, multi68, cfg)
    logger.info("Triangulation: %d 3D points from %d frames", len(points_3d), max(n66, n68))

    # Smooth
    flight_3d = {fi: points_3d[fi][:3] for fi in points_3d}
    smoothed_3d = smooth_trajectory_sg(flight_3d, window_length=11, polyorder=3)

    # Detect bounces
    bounces = detect_bounces(smoothed_3d)
    logger.info("Detected %d bounces", len(bounces))

    return bounces, points_3d, smoothed_3d


# ── Match & evaluate ──────────────────────────────────────────────────────────

@dataclass
class MatchResult:
    gt_frame: int
    det_frame: int
    gt_world: tuple[float, float]
    det_world: tuple[float, float]
    error_m: float
    frame_delta: int


def match_bounces(
    gt_world: list[tuple[int, float, float]],
    detected: list[dict],
    tolerance: int = TOLERANCE_FRAMES,
) -> tuple[list[MatchResult], list[int], list[int]]:
    """Match GT bounces to detected bounces.

    Returns: (matches, unmatched_gt_indices, unmatched_det_indices)
    """
    gt_matched = set()
    det_matched = set()
    matches = []

    for di, db in enumerate(detected):
        df = db["frame"]
        dx, dy = db["x"], db["y"]

        best_dist_frames = float("inf")
        best_gi = -1

        for gi, (gf, gwx, gwy) in enumerate(gt_world):
            if gi in gt_matched:
                continue
            fdist = abs(df - gf)
            if fdist < best_dist_frames and fdist <= tolerance:
                best_dist_frames = fdist
                best_gi = gi

        if best_gi >= 0:
            gf, gwx, gwy = gt_world[best_gi]
            error = ((dx - gwx) ** 2 + (dy - gwy) ** 2) ** 0.5
            matches.append(MatchResult(
                gt_frame=gf, det_frame=df,
                gt_world=(gwx, gwy), det_world=(dx, dy),
                error_m=error, frame_delta=abs(df - gf),
            ))
            gt_matched.add(best_gi)
            det_matched.add(di)

    unmatched_gt = [i for i in range(len(gt_world)) if i not in gt_matched]
    unmatched_det = [i for i in range(len(detected)) if i not in det_matched]
    return matches, unmatched_gt, unmatched_det


def print_report(
    matches: list[MatchResult],
    n_gt: int,
    n_det: int,
    unmatched_gt: list[int],
    gt_world: list[tuple[int, float, float]],
):
    """Print accuracy report."""
    errors_cm = [m.error_m * 100 for m in matches]

    print("\n" + "=" * 60)
    print("  BOUNCE LANDING ACCURACY REPORT")
    print("=" * 60)

    print(f"\n  GT bounces:       {n_gt}")
    print(f"  Detected bounces: {n_det}")
    print(f"  Matched:          {len(matches)}")
    print(f"  Recall:           {len(matches)/n_gt*100:.1f}%" if n_gt else "  Recall: N/A")
    print(f"  Precision:        {len(matches)/n_det*100:.1f}%" if n_det else "  Precision: N/A")

    if errors_cm:
        errors_sorted = sorted(errors_cm)
        n = len(errors_sorted)
        print(f"\n  Position Error (cm):")
        print(f"    Mean:     {np.mean(errors_cm):6.1f} cm")
        print(f"    Median:   {np.median(errors_cm):6.1f} cm")
        print(f"    P90:      {errors_sorted[min(int(n*0.9), n-1)]:6.1f} cm")
        print(f"    Max:      {max(errors_cm):6.1f} cm")
        print(f"    < 20cm:   {sum(1 for e in errors_cm if e < 20)/n*100:5.1f}%")
        print(f"    < 30cm:   {sum(1 for e in errors_cm if e < 30)/n*100:5.1f}%")
        print(f"    < 50cm:   {sum(1 for e in errors_cm if e < 50)/n*100:5.1f}%")

    # Pass/fail
    print(f"\n  --- PASS/FAIL ---")
    mean_ok = np.mean(errors_cm) < 30 if errors_cm else False
    recall_ok = len(matches) / n_gt >= 0.80 if n_gt else False
    print(f"    Mean < 30cm:  {'PASS OK' if mean_ok else 'FAIL NG'} ({np.mean(errors_cm):.1f}cm)" if errors_cm else "    Mean: N/A")
    print(f"    Recall > 80%: {'PASS OK' if recall_ok else 'FAIL NG'} ({len(matches)/n_gt*100:.1f}%)" if n_gt else "    Recall: N/A")

    if unmatched_gt:
        print(f"\n  Missed GT bounces (frames): {[gt_world[i][0] for i in unmatched_gt]}")

    print("=" * 60 + "\n")

    return {"mean_cm": np.mean(errors_cm) if errors_cm else None,
            "recall": len(matches) / n_gt if n_gt else 0,
            "precision": len(matches) / n_det if n_det else 0}


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-frames", type=int, default=MAX_FRAMES)
    args = parser.parse_args()

    cfg = load_config()
    homography_path = cfg["homography"]["path"]

    # Load & transform GT bounces
    gt_bounces_px = load_gt_bounces(GT_DIR, args.max_frames)
    logger.info("GT bounces: %d", len(gt_bounces_px))
    gt_bounces_world = pixel_to_world_batch(gt_bounces_px, homography_path, "cam66")

    for fi, wx, wy in gt_bounces_world:
        logger.info("  GT bounce frame %d: (%.2f, %.2f) m", fi, wx, wy)

    # Run pipeline
    detected_bounces, points_3d, smoothed_3d = run_pipeline(args.max_frames)

    # Match & report
    matches, unmatched_gt, unmatched_det = match_bounces(gt_bounces_world, detected_bounces)
    results = print_report(matches, len(gt_bounces_world), len(detected_bounces),
                          unmatched_gt, gt_bounces_world)

    # Detail per match
    if matches:
        print("  Per-bounce details:")
        print(f"  {'GT frame':>10} {'Det frame':>10} {'Error cm':>10} {'GT pos':>20} {'Det pos':>20}")
        for m in sorted(matches, key=lambda m: m.gt_frame):
            print(f"  {m.gt_frame:10d} {m.det_frame:10d} {m.error_m*100:10.1f}"
                  f"  ({m.gt_world[0]:.2f}, {m.gt_world[1]:.2f})"
                  f"  ({m.det_world[0]:.2f}, {m.det_world[1]:.2f})")

    return results


if __name__ == "__main__":
    main()
