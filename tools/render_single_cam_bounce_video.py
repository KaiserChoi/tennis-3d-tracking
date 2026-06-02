from __future__ import annotations

import argparse
import json
from collections import deque
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from app.analytics import SingleCamBounceDetector
from app.pipeline.homography import HomographyTransformer


def _load_predictions(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data.get("frames"), list):
        raise ValueError(f"Invalid predictions JSON: {path}")
    return data


def _make_det(
    frame: dict[str, Any],
    *,
    homography: HomographyTransformer,
    fps: float,
) -> dict | None:
    if not frame.get("visible"):
        return None
    px = frame.get("x")
    py = frame.get("y")
    if px is None or py is None:
        return None
    wx, wy = homography.pixel_to_world(float(px), float(py))
    fi = int(frame["frame_index"])
    prob = float(frame.get("prob") or frame.get("raw_prob") or 0.0)
    return {
        "camera_name": "cam68",
        "frame_index": fi,
        "timestamp": fi / fps,
        "capture_ts": fi / fps,
        "pixel_x": float(px),
        "pixel_y": float(py),
        "x": float(wx),
        "y": float(wy),
        "world_x": float(wx),
        "world_y": float(wy),
        "blob_sum": prob,
        "confidence": prob,
    }


def _run_detector(
    frames: list[dict[str, Any]],
    *,
    homography: HomographyTransformer,
    fps: float,
    detector: SingleCamBounceDetector,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    bounces: list[dict[str, Any]] = []
    for frame in frames:
        det = _make_det(frame, homography=homography, fps=fps)
        if det is None:
            continue
        bounce = detector.update("cam68", det)
        if bounce is not None:
            bd = bounce.to_dict()
            px = bd.get("cam_pixels", {}).get("cam68")
            bd["pixel_x"] = px[0] if px else None
            bd["pixel_y"] = px[1] if px else None
            bd["bounce_mode"] = "mono_cam68"
            bounces.append(bd)
    return bounces, detector.get_stats()


def _draw_text(
    image: np.ndarray,
    text: str,
    org: tuple[int, int],
    *,
    scale: float = 0.72,
    color: tuple[int, int, int] = (255, 255, 255),
) -> None:
    cv2.putText(image, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 5, cv2.LINE_AA)
    cv2.putText(image, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2, cv2.LINE_AA)


def _draw_trail(
    image: np.ndarray,
    trail: deque[tuple[int, int]],
) -> None:
    pts = list(trail)
    if len(pts) < 2:
        return
    for i in range(1, len(pts)):
        alpha = i / max(1, len(pts) - 1)
        color = (0, int(90 + 120 * alpha), 255)
        cv2.line(image, pts[i - 1], pts[i], color, 2, cv2.LINE_AA)


def _draw_bounce_marker(
    image: np.ndarray,
    xy: tuple[int, int],
    *,
    label: str,
) -> None:
    cv2.circle(image, xy, 22, (255, 0, 255), 4, cv2.LINE_AA)
    cv2.drawMarker(image, xy, (255, 255, 255), markerType=cv2.MARKER_CROSS, markerSize=28, thickness=3)
    _draw_text(image, label, (xy[0] + 18, max(30, xy[1] - 18)), scale=0.68, color=(255, 180, 255))


def render_video(
    *,
    predictions_path: Path,
    output_path: Path,
    summary_path: Path,
    homography_path: Path,
    frames_dir: Path | None,
    fps: float,
    display_scale: float,
    trail_len: int,
    linger_frames: int,
    max_gap_frames: int,
    max_inactive_frames: int,
    max_jump_px_per_frame: float,
    min_prominence_px: float,
    confirm_frames: int,
) -> dict[str, Any]:
    payload = _load_predictions(predictions_path)
    frames = payload["frames"]
    if frames_dir is None:
        frames_dir = Path(payload.get("frames_dir") or predictions_path.parent)
    homo = HomographyTransformer(str(homography_path), "cam68")
    detector = SingleCamBounceDetector(
        net_line_y=260.0,
        confirm_frames=confirm_frames,
        max_gap_frames=max_gap_frames,
        max_inactive_frames=max_inactive_frames,
        max_jump_px_per_frame=max_jump_px_per_frame,
        min_prominence_px=min_prominence_px,
    )
    bounces, stats = _run_detector(frames, homography=homo, fps=fps, detector=detector)

    bounce_by_frame: dict[int, list[dict[str, Any]]] = {}
    for bd in bounces:
        fi = int(bd.get("frame_index") or -1)
        bounce_by_frame.setdefault(fi, []).append(bd)

    first_img = cv2.imread(str(frames_dir / frames[0]["image"]))
    if first_img is None:
        raise RuntimeError(f"Cannot read first frame from {frames_dir}")
    h, w = first_img.shape[:2]
    out_w = int(round(w * display_scale))
    out_h = int(round(h * display_scale))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (out_w, out_h),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not open VideoWriter: {output_path}")

    trail: deque[tuple[int, int]] = deque(maxlen=trail_len)
    recent_bounces: deque[tuple[int, tuple[int, int], str]] = deque(maxlen=8)
    try:
        for frame in frames:
            frame_path = frames_dir / str(frame.get("image", f"{int(frame['frame_index']):05d}.jpg"))
            img = cv2.imread(str(frame_path))
            if img is None:
                img = np.zeros((h, w, 3), dtype=np.uint8)
            if display_scale != 1.0:
                canvas = cv2.resize(img, (out_w, out_h), interpolation=cv2.INTER_AREA)
            else:
                canvas = img.copy()

            fi = int(frame["frame_index"])

            if frame.get("visible") and frame.get("x") is not None and frame.get("y") is not None:
                px = int(round(float(frame["x"]) * display_scale))
                py = int(round(float(frame["y"]) * display_scale))
                trail.append((px, py))
                _draw_trail(canvas, trail)
                cv2.circle(canvas, (px, py), 5, (0, 220, 255), -1, cv2.LINE_AA)
                cv2.circle(canvas, (px, py), 9, (0, 80, 255), 2, cv2.LINE_AA)
            else:
                _draw_trail(canvas, trail)

            for bd in bounce_by_frame.get(fi, []):
                bx = bd.get("pixel_x")
                by = bd.get("pixel_y")
                if bx is None or by is None:
                    continue
                xy = (int(round(float(bx) * display_scale)), int(round(float(by) * display_scale)))
                tag = "IN" if bd.get("in_court") else "OUT"
                label = f"BOUNCE {tag} #{len(recent_bounces) + 1}"
                recent_bounces.append((fi, xy, label))

            for bfi, xy, label in list(recent_bounces):
                age = fi - bfi
                if age < 0 or age > linger_frames:
                    continue
                _draw_bounce_marker(canvas, xy, label=label)

            _draw_text(
                canvas,
                f"cam68 mono bounce | frame {fi:05d} | bounces {len(bounces)}",
                (18, 34),
                color=(255, 255, 255),
            )
            _draw_text(
                canvas,
                f"accepted={stats.get('accepted', 0)} windows={stats.get('windows_processed', 0)} last_reject={stats.get('last_reject_reason', '')}",
                (18, 66),
                scale=0.62,
                color=(220, 255, 220),
            )
            writer.write(canvas)
    finally:
        writer.release()

    summary = {
        "predictions_path": str(predictions_path),
        "frames_dir": str(frames_dir),
        "output_path": str(output_path),
        "frames": len(frames),
        "fps": fps,
        "display_scale": display_scale,
        "params": {
            "confirm_frames": confirm_frames,
            "max_gap_frames": max_gap_frames,
            "max_inactive_frames": max_inactive_frames,
            "max_jump_px_per_frame": max_jump_px_per_frame,
            "min_prominence_px": min_prominence_px,
        },
        "bounces": bounces,
        "bounce_count": len(bounces),
        "stats": stats,
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Render cam68 single-camera net-line bounce detector output.")
    parser.add_argument(
        "--predictions",
        type=Path,
        default=Path(r"D:\tennis-dataset\1001\clip3\cam68_tracknet_prediction_vis.predictions.json"),
    )
    parser.add_argument("--frames-dir", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=Path(r"D:\tennis-dataset\1001\clip3\cam68_single_cam_bounce_netline.mp4"))
    parser.add_argument("--summary", type=Path, default=Path(r"D:\tennis-dataset\1001\clip3\cam68_single_cam_bounce_netline.summary.json"))
    parser.add_argument("--homography", type=Path, default=Path("src/homography_matrices.json"))
    parser.add_argument("--fps", type=float, default=25.0)
    parser.add_argument("--display-scale", type=float, default=0.75)
    parser.add_argument("--trail-len", type=int, default=18)
    parser.add_argument("--linger-frames", type=int, default=20)
    parser.add_argument("--confirm-frames", type=int, default=4)
    parser.add_argument("--max-gap-frames", type=int, default=50)
    parser.add_argument("--max-inactive-frames", type=int, default=50)
    parser.add_argument("--max-jump-px-per-frame", type=float, default=900.0)
    parser.add_argument("--min-prominence-px", type=float, default=4.0)
    args = parser.parse_args()

    summary = render_video(
        predictions_path=args.predictions,
        output_path=args.output,
        summary_path=args.summary,
        homography_path=args.homography,
        frames_dir=args.frames_dir,
        fps=args.fps,
        display_scale=args.display_scale,
        trail_len=args.trail_len,
        linger_frames=args.linger_frames,
        confirm_frames=args.confirm_frames,
        max_gap_frames=args.max_gap_frames,
        max_inactive_frames=args.max_inactive_frames,
        max_jump_px_per_frame=args.max_jump_px_per_frame,
        min_prominence_px=args.min_prominence_px,
    )
    print(json.dumps({
        "output_path": summary["output_path"],
        "summary_path": str(args.summary),
        "frames": summary["frames"],
        "bounce_count": summary["bounce_count"],
        "stats": summary["stats"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
