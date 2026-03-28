"""Ablation: compare live pipeline vs offline pipeline stage by stage.

Runs both pipelines on the same video and compares outputs at each stage:
  Stage 1: Per-camera detection (pixel coords)
  Stage 2: Triangulation (3D points)
  Stage 3: Smoothing (SG filter)
  Stage 4: Bounce detection

Usage:
    python -m tools.ablation_live_vs_offline [--max-frames 1800]
"""

import logging
import math
import time
from collections import defaultdict

import numpy as np
import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

VIDEO_66 = "uploads/cam66_20260307_173403_2min.mp4"
VIDEO_68 = "uploads/cam68_20260307_173403_2min.mp4"
MAX_FRAMES = 1800

SINGLES_X_MIN, SINGLES_X_MAX = 1.37, 6.86
COURT_L = 23.77


def load_config():
    with open("config.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg


# ══════════════════════════════════════════════════════════════════════════════
# OFFLINE PIPELINE (verified correct)
# ══════════════════════════════════════════════════════════════════════════════

def run_offline(cfg, max_frames):
    """Run the offline pipeline exactly as render_tracking_video.py does."""
    from tools.render_tracking_video import (
        build_detector,
        detect_bounces,
        run_detection_multi,
        smooth_trajectory_sg,
        triangulate_multi_blob,
    )

    detector, postproc = build_detector(cfg)

    # Stage 1: Detection
    logger.info("=== OFFLINE Stage 1: Detection ===")
    multi66, det66, n66 = run_detection_multi(VIDEO_66, detector, postproc, max_frames, top_k=2)
    multi68, det68, n68 = run_detection_multi(VIDEO_68, detector, postproc, max_frames, top_k=2)

    # Stage 2: Triangulation
    logger.info("=== OFFLINE Stage 2: Triangulation ===")
    points_3d, _, stats = triangulate_multi_blob(multi66, multi68, cfg)

    # Stage 3: Smoothing
    logger.info("=== OFFLINE Stage 3: SG Smoothing ===")
    flight_3d = {fi: points_3d[fi][:3] for fi in points_3d}
    smoothed = smooth_trajectory_sg(flight_3d, window_length=11, polyorder=3)

    # Stage 4: Bounce detection
    logger.info("=== OFFLINE Stage 4: Bounce Detection ===")
    bounces = detect_bounces(smoothed)

    return {
        "det66": det66,     # {frame: (px, py, conf)}
        "det68": det68,
        "multi66": multi66, # {frame: [blobs]}
        "multi68": multi68,
        "points_3d": points_3d,  # {frame: (x, y, z, ray_dist)}
        "smoothed": smoothed,     # {frame: (x, y, z)}
        "bounces": bounces,       # [{frame, x, y, z, in_court}]
    }


# ══════════════════════════════════════════════════════════════════════════════
# LIVE PIPELINE (simulated — same logic as orchestrator)
# ══════════════════════════════════════════════════════════════════════════════

def run_live(cfg, max_frames, offline_data=None):
    """Simulate the live pipeline with INDEPENDENT detection (no court X filter)."""
    import cv2
    from app.analytics import HybridBounceDetector
    from app.pipeline.homography import HomographyTransformer
    from app.pipeline.inference import create_detector
    from app.pipeline.multi_blob_matcher import MultiBlobMatcher
    from app.pipeline.postprocess import BallTracker
    from app.triangulation import triangulate

    mcfg = cfg["model"]
    homo_path = cfg["homography"]["path"]
    homo66 = HomographyTransformer(homo_path, "cam66")
    homo68 = HomographyTransformer(homo_path, "cam68")

    logger.info("=== LIVE Stage 1: Independent detection (no court X filter) ===")

    def detect_single_cam(video_path, homo):
        """Detect with top-K, NO court X filtering (matches offline)."""
        detector = create_detector(
            mcfg["path"], tuple(mcfg["input_size"]),
            mcfg["frames_in"], mcfg.get("frames_out", mcfg["frames_in"]),
            mcfg.get("device", "cuda"),
        )
        tracker = BallTracker(original_size=(1920, 1080), threshold=0.5)
        cap = cv2.VideoCapture(video_path)
        n_frames = min(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), max_frames)
        detector.compute_video_median(cap, 0, n_frames)
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        frames_buf = []
        detections = {}
        seq_len = mcfg["frames_in"]

        for fi in range(n_frames):
            ret, frame = cap.read()
            if not ret:
                break
            frame[0:41, 0:603] = 0
            frames_buf.append(frame)
            if len(frames_buf) < seq_len:
                continue

            heatmaps = detector.infer(frames_buf)
            for i in range(min(mcfg.get("frames_out", seq_len), len(heatmaps))):
                blobs = tracker.process_heatmap_multi(heatmaps[i], max_blobs=2)
                if not blobs:
                    continue
                top = blobs[0]
                px, py, conf = top["pixel_x"], top["pixel_y"], top["blob_sum"]
                wx, wy = homo.pixel_to_world(px, py)
                out_fi = fi - len(frames_buf) + 1 + i
                # NO court X filtering — matches offline
                candidates = []
                for b in blobs:
                    bwx, bwy = homo.pixel_to_world(b["pixel_x"], b["pixel_y"])
                    candidates.append({
                        **b, "x": bwx, "y": bwy,
                        "world_x": bwx, "world_y": bwy,
                    })
                detections[out_fi] = {
                    "x": wx, "y": wy,
                    "pixel_x": px, "pixel_y": py,
                    "confidence": conf, "blob_sum": conf,
                    "timestamp": out_fi / 25.0,
                    "frame_index": out_fi,
                    "candidates": candidates,
                }
            frames_buf.clear()

        cap.release()
        return detections

    live_det66 = detect_single_cam(VIDEO_66, homo66)
    live_det68 = detect_single_cam(VIDEO_68, homo68)
    logger.info("Live det66: %d frames, det68: %d frames", len(live_det66), len(live_det68))

    # ── Stage 2: Triangulation (live matcher, like orchestrator) ──
    logger.info("=== LIVE Stage 2: Triangulation ===")
    cam_positions = {
        n: cfg["cameras"][n]["position_3d"]
        for n in cfg["cameras"] if "position_3d" in cfg["cameras"][n]
    }
    cam_names = sorted(cam_positions.keys())
    pos1 = cam_positions[cam_names[0]]
    pos2 = cam_positions[cam_names[1]]

    # Use MultiBlobMatcher (same params as offline)
    matcher = MultiBlobMatcher(pos1, pos2, valid_z_range=(0.0, 8.0))
    points_3d = {}
    common_frames = sorted(set(live_det66.keys()) & set(live_det68.keys()))
    for fi in common_frames:
        d1, d2 = live_det66[fi], live_det68[fi]
        match = matcher.match(d1, d2)
        if match is not None:
            points_3d[fi] = (match["x"], match["y"], match["z"], match.get("ray_distance", 0))

    logger.info("Live triangulation: %d 3D points from %d common frames", len(points_3d), len(common_frames))

    # ── Stage 3: SG Smoothing (like orchestrator._smooth_latest) ──
    logger.info("=== LIVE Stage 3: SG Smoothing ===")
    # Use the SAME offline SG function for fair comparison
    from tools.render_tracking_video import smooth_trajectory_sg
    flight_3d = {fi: points_3d[fi][:3] for fi in points_3d}
    smoothed = smooth_trajectory_sg(flight_3d, window_length=11, polyorder=3)

    # ── Stage 4: Bounce detection (HybridBounceDetector, streaming) ──
    logger.info("=== LIVE Stage 4: Bounce Detection (HybridBounceDetector) ===")
    hbd = HybridBounceDetector()
    bounces = []
    for fi in sorted(smoothed.keys()):
        sx, sy, sz = smoothed[fi]
        pt = {"x": sx, "y": sy, "z": sz, "timestamp": fi / 25.0}
        cam_dets = {}
        if fi in live_det66 and isinstance(live_det66[fi], dict):
            cam_dets[cam_names[0]] = live_det66[fi]
        if fi in live_det68 and isinstance(live_det68[fi], dict):
            cam_dets[cam_names[1]] = live_det68[fi]
        b = hbd.update(pt, cam_dets)
        if b is not None:
            bounces.append({
                "frame": fi, "x": b.x, "y": b.y, "z": b.z,
                "in_court": b.in_court,
            })

    return {
        "det66": {fi: (d["pixel_x"], d["pixel_y"], d["confidence"]) for fi, d in live_det66.items()},
        "det68": {fi: (d["pixel_x"], d["pixel_y"], d["confidence"]) for fi, d in live_det68.items()},
        "points_3d": points_3d,
        "smoothed": smoothed,
        "bounces": bounces,
    }


# ══════════════════════════════════════════════════════════════════════════════
# COMPARISON
# ══════════════════════════════════════════════════════════════════════════════

def compare_detections(offline_det, live_det, label):
    """Compare per-camera detections."""
    off_frames = set(offline_det.keys())
    live_frames = set(live_det.keys())
    common = off_frames & live_frames
    only_off = off_frames - live_frames
    only_live = live_frames - off_frames

    pixel_diffs = []
    for fi in sorted(common):
        opx, opy = offline_det[fi][0], offline_det[fi][1]
        lpx, lpy = live_det[fi][0], live_det[fi][1]
        d = math.hypot(opx - lpx, opy - lpy)
        pixel_diffs.append(d)

    print(f"\n  {label}:")
    print(f"    Offline: {len(off_frames)} frames")
    print(f"    Live:    {len(live_frames)} frames")
    print(f"    Common:  {len(common)} | Only-offline: {len(only_off)} | Only-live: {len(only_live)}")
    if pixel_diffs:
        arr = np.array(pixel_diffs)
        print(f"    Pixel diff (common frames): mean={arr.mean():.1f} median={np.median(arr):.1f} "
              f"<5px={np.mean(arr<5)*100:.0f}% <10px={np.mean(arr<10)*100:.0f}%")
        if arr.mean() < 1.0:
            print(f"    --> IDENTICAL")
        elif arr.mean() < 5.0:
            print(f"    --> CLOSE (minor differences)")
        else:
            print(f"    --> DIFFERENT")


def compare_3d(offline_3d, live_3d, label):
    """Compare 3D points."""
    off_frames = set(offline_3d.keys())
    live_frames = set(live_3d.keys())
    common = off_frames & live_frames

    dists = []
    for fi in sorted(common):
        o = offline_3d[fi][:3]
        l = live_3d[fi][:3]
        d = math.sqrt(sum((a - b) ** 2 for a, b in zip(o, l)))
        dists.append(d)

    print(f"\n  {label}:")
    print(f"    Offline: {len(off_frames)} | Live: {len(live_frames)} | Common: {len(common)}")
    print(f"    Only-offline: {len(off_frames - live_frames)} | Only-live: {len(live_frames - off_frames)}")
    if dists:
        arr = np.array(dists)
        print(f"    3D dist (m): mean={arr.mean():.3f} median={np.median(arr):.3f} "
              f"<0.1m={np.mean(arr<0.1)*100:.0f}% <0.5m={np.mean(arr<0.5)*100:.0f}%")
        if arr.mean() < 0.01:
            print(f"    --> IDENTICAL")
        elif arr.mean() < 0.1:
            print(f"    --> CLOSE")
        else:
            print(f"    --> DIFFERENT (mean={arr.mean():.2f}m)")


def compare_bounces(offline_b, live_b):
    """Compare bounce detections."""
    print(f"\n  Bounces:")
    print(f"    Offline: {len(offline_b)} bounces")
    print(f"    Live:    {len(live_b)} bounces")

    # Match by frame proximity
    matched = []
    live_used = set()
    for ob in offline_b:
        of = ob["frame"]
        best_li = -1
        best_dist = float("inf")
        for li, lb in enumerate(live_b):
            if li in live_used:
                continue
            fd = abs(of - lb["frame"])
            if fd < best_dist:
                best_dist = fd
                best_li = li
        if best_li >= 0 and best_dist <= 10:
            live_used.add(best_li)
            lb = live_b[best_li]
            pos_err = math.sqrt((ob["x"] - lb["x"])**2 + (ob["y"] - lb["y"])**2)
            matched.append({
                "off_frame": of, "live_frame": lb["frame"],
                "frame_diff": abs(of - lb["frame"]),
                "pos_err_m": pos_err,
            })

    unmatched_off = len(offline_b) - len(matched)
    unmatched_live = len(live_b) - len(matched)

    print(f"    Matched:         {len(matched)}")
    print(f"    Missed (offline only): {unmatched_off}")
    print(f"    Extra (live only):     {unmatched_live}")

    if matched:
        frame_diffs = [m["frame_diff"] for m in matched]
        pos_errs = [m["pos_err_m"] for m in matched]
        print(f"    Frame diff: mean={np.mean(frame_diffs):.1f} max={max(frame_diffs)}")
        print(f"    Position err: mean={np.mean(pos_errs)*100:.1f}cm max={max(pos_errs)*100:.1f}cm")

    print(f"\n    Offline bounces:")
    for b in offline_b:
        print(f"      frame={b['frame']:5d} ({b['x']:.2f}, {b['y']:.2f}, z={b['z']:.2f}) {'IN' if b['in_court'] else 'OUT'}")
    print(f"    Live bounces:")
    for b in live_b:
        print(f"      frame={b['frame']:5d} ({b['x']:.2f}, {b['y']:.2f}, z={b['z']:.2f}) {'IN' if b['in_court'] else 'OUT'}")

    if len(matched) == len(offline_b) == len(live_b) and all(m["frame_diff"] <= 2 for m in matched):
        print(f"    --> IDENTICAL")
    elif len(matched) >= len(offline_b) * 0.8:
        print(f"    --> CLOSE")
    else:
        print(f"    --> DIFFERENT")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-frames", type=int, default=MAX_FRAMES)
    args = parser.parse_args()

    cfg = load_config()

    print("\n" + "=" * 70)
    print("  ABLATION: LIVE vs OFFLINE PIPELINE")
    print("=" * 70)

    # Run both pipelines (live reuses offline Stage 1 to isolate Stage 2-4)
    offline = run_offline(cfg, args.max_frames)
    live = run_live(cfg, args.max_frames, offline_data=offline)

    # Stage 1: Detection comparison
    print("\n" + "=" * 70)
    print("  STAGE 1: Per-Camera Detection")
    print("=" * 70)
    compare_detections(offline["det66"], live["det66"], "cam66")
    compare_detections(offline["det68"], live["det68"], "cam68")

    # Stage 2: Triangulation comparison
    print("\n" + "=" * 70)
    print("  STAGE 2: Triangulation (3D points)")
    print("=" * 70)
    compare_3d(offline["points_3d"], live["points_3d"], "Raw 3D")

    # Stage 3: Smoothing comparison
    print("\n" + "=" * 70)
    print("  STAGE 3: SG Smoothed Trajectory")
    print("=" * 70)
    compare_3d(offline["smoothed"], live["smoothed"], "Smoothed 3D")

    # Stage 4: Bounce detection
    print("\n" + "=" * 70)
    print("  STAGE 4: Bounce Detection")
    print("=" * 70)
    compare_bounces(offline["bounces"], live["bounces"])

    print("\n" + "=" * 70)
    print("  DONE")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
