"""Evaluate net-crossing speed accuracy.

Compares the live two-point speed estimate against trajectory-fitted
reference speed (parabolic fit using many points).

Usage:
    python -m tools.eval_speed_accuracy [--max-frames 1800]
"""

import logging
from pathlib import Path

import numpy as np
import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

VIDEO_66 = "uploads/cam66_20260307_173403_2min.mp4"
VIDEO_68 = "uploads/cam68_20260307_173403_2min.mp4"
MAX_FRAMES = 1800
NET_Y = 11.885
FPS = 25.0


def load_config():
    with open("config.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def find_net_crossings_twopoint(points_3d: dict, fps: float = FPS) -> list[dict]:
    """Simulate live two-point net crossing speed (same as orchestrator)."""
    frames = sorted(points_3d.keys())
    crossings = []
    for i in range(1, len(frames)):
        f_prev, f_curr = frames[i - 1], frames[i]
        p_prev = points_3d[f_prev]
        p_curr = points_3d[f_curr]
        y_prev, y_curr = p_prev[1], p_curr[1]

        crossed = ((y_prev < NET_Y <= y_curr) or (y_prev > NET_Y >= y_curr))
        if not crossed:
            continue

        dt = (f_curr - f_prev) / fps
        if dt < 0.001:
            continue

        dx = p_curr[0] - p_prev[0]
        dy = p_curr[1] - p_prev[1]
        dz = p_curr[2] - p_prev[2]
        dist = (dx**2 + dy**2 + dz**2) ** 0.5
        speed_kmh = (dist / dt) * 3.6

        if 20 <= speed_kmh <= 250:
            direction = "near_to_far" if y_curr > y_prev else "far_to_near"
            crossings.append({
                "frame": f_curr,
                "speed_kmh": round(speed_kmh, 1),
                "direction": direction,
                "y_prev": y_prev,
                "y_curr": y_curr,
            })
    return crossings


def find_net_crossings_trajectory(points_3d: dict, fps: float = FPS) -> list[dict]:
    """Compute reference speed via parabolic fit around each net crossing.

    Uses a window of points before/after the crossing to fit a physics-based
    parabola, extracting the speed from the fit coefficients.
    """
    from app.trajectory import fit_spatial_parabola

    frames = sorted(points_3d.keys())
    crossings = []

    for i in range(1, len(frames)):
        f_prev, f_curr = frames[i - 1], frames[i]
        y_prev = points_3d[f_prev][1]
        y_curr = points_3d[f_curr][1]

        if not ((y_prev < NET_Y <= y_curr) or (y_prev > NET_Y >= y_curr)):
            continue

        # Gather window of points around crossing
        window_pts = []
        for j in range(max(0, i - 8), min(len(frames), i + 8)):
            fj = frames[j]
            p = points_3d[fj]
            window_pts.append({"x": p[0], "y": p[1], "z": p[2], "frame": fj})

        if len(window_pts) < 5:
            continue

        fit = fit_spatial_parabola(window_pts)
        if fit is None or fit.get("speed_kmh", 0) < 20 or fit.get("speed_kmh", 0) > 350:
            continue

        direction = "near_to_far" if y_curr > y_prev else "far_to_near"
        crossings.append({
            "frame": f_curr,
            "speed_kmh": round(fit["speed_kmh"], 1),
            "direction": direction,
        })

    return crossings


def compare_speeds(twopoint: list[dict], trajectory: list[dict]):
    """Match and compare two-point vs trajectory speeds."""
    matches = []
    traj_used = set()

    for tp in twopoint:
        best_gi = -1
        best_fdist = float("inf")
        for gi, tr in enumerate(trajectory):
            if gi in traj_used:
                continue
            fdist = abs(tp["frame"] - tr["frame"])
            if fdist < best_fdist and fdist <= 5:
                best_fdist = fdist
                best_gi = gi

        if best_gi >= 0:
            tr = trajectory[best_gi]
            traj_used.add(best_gi)
            abs_err = abs(tp["speed_kmh"] - tr["speed_kmh"])
            rel_err = abs_err / tr["speed_kmh"] * 100 if tr["speed_kmh"] > 0 else 0
            matches.append({
                "frame": tp["frame"],
                "twopoint_kmh": tp["speed_kmh"],
                "reference_kmh": tr["speed_kmh"],
                "abs_error_kmh": round(abs_err, 1),
                "rel_error_pct": round(rel_err, 1),
                "direction": tp["direction"],
            })

    return matches


def print_report(matches: list[dict]):
    """Print speed accuracy report."""
    print("\n" + "=" * 60)
    print("  NET CROSSING SPEED ACCURACY REPORT")
    print("=" * 60)

    if not matches:
        print("  No matched net crossings found.")
        print("=" * 60)
        return {}

    abs_errors = [m["abs_error_kmh"] for m in matches]
    rel_errors = [m["rel_error_pct"] for m in matches]

    print(f"\n  Matched crossings: {len(matches)}")
    print(f"\n  Absolute Error (km/h):")
    print(f"    Mean:   {np.mean(abs_errors):6.1f} km/h")
    print(f"    Median: {np.median(abs_errors):6.1f} km/h")
    print(f"    Max:    {max(abs_errors):6.1f} km/h")
    print(f"\n  Relative Error (%):")
    print(f"    Mean:   {np.mean(rel_errors):6.1f}%")
    print(f"    Median: {np.median(rel_errors):6.1f}%")
    print(f"    Max:    {max(rel_errors):6.1f}%")

    # Pass/fail: < 15% relative OR < 10 km/h absolute
    pass_count = sum(1 for m in matches if m["rel_error_pct"] < 15 or m["abs_error_kmh"] < 10)
    pass_rate = pass_count / len(matches) * 100
    print(f"\n  --- PASS/FAIL ---")
    print(f"    Speed within threshold (< 15% or < 10km/h): {pass_rate:.1f}%")
    print(f"    {'PASS OK' if pass_rate >= 80 else 'FAIL NG'} (need >= 80%)")

    print(f"\n  Per-crossing details:")
    print(f"  {'Frame':>8} {'2-Point':>10} {'Reference':>10} {'AbsErr':>8} {'RelErr':>8} {'Dir'}")
    for m in sorted(matches, key=lambda m: m["frame"]):
        print(f"  {m['frame']:8d} {m['twopoint_kmh']:8.1f}kh {m['reference_kmh']:8.1f}kh"
              f" {m['abs_error_kmh']:6.1f}kh {m['rel_error_pct']:6.1f}%"
              f"  {m['direction']}")

    print("=" * 60 + "\n")

    return {
        "mean_abs_kmh": round(np.mean(abs_errors), 1),
        "mean_rel_pct": round(np.mean(rel_errors), 1),
        "pass_rate": round(pass_rate, 1),
    }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-frames", type=int, default=MAX_FRAMES)
    args = parser.parse_args()

    from tools.render_tracking_video import (
        build_detector,
        run_detection_multi,
        smooth_trajectory_sg,
        triangulate_multi_blob,
    )

    cfg = load_config()
    detector, postproc = build_detector(cfg)

    logger.info("Running cam66 detection...")
    multi66, det66, n66 = run_detection_multi(VIDEO_66, detector, postproc, args.max_frames, top_k=2)
    logger.info("Running cam68 detection...")
    multi68, det68, n68 = run_detection_multi(VIDEO_68, detector, postproc, args.max_frames, top_k=2)

    logger.info("Triangulating...")
    points_3d, _, stats = triangulate_multi_blob(multi66, multi68, cfg)

    # Smooth for trajectory fitting
    flight_3d = {fi: points_3d[fi][:3] for fi in points_3d}
    smoothed_3d = smooth_trajectory_sg(flight_3d, window_length=11, polyorder=3)

    # Two-point speeds (simulates live)
    twopoint = find_net_crossings_twopoint(smoothed_3d)
    logger.info("Two-point net crossings: %d", len(twopoint))

    # Reference speeds (trajectory fit)
    trajectory = find_net_crossings_trajectory(smoothed_3d)
    logger.info("Trajectory-fit net crossings: %d", len(trajectory))

    # Compare
    matches = compare_speeds(twopoint, trajectory)
    results = print_report(matches)
    return results


if __name__ == "__main__":
    main()
