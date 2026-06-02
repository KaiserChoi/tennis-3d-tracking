"""Render raw YOLO detections for a JPG image sequence.

This intentionally avoids static filtering, trajectory filtering, bounce
detection, minimap rendering, and homography projection. It is just the model's
own boxes over the source frames.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from ultralytics import YOLO


DEFAULT_FRAMES_DIR = Path(r"D:\tennis-dataset\1001\clip11\cam68_20260404_075325_2min")


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
    scale: float = 0.7,
    color: tuple[int, int, int] = (255, 255, 255),
) -> None:
    cv2.putText(image, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 5, cv2.LINE_AA)
    cv2.putText(image, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2, cv2.LINE_AA)


def run(args: argparse.Namespace) -> dict[str, Any]:
    image_paths = sorted(args.frames_dir.glob("*.jpg"), key=_safe_frame_index)
    if args.max_frames is not None:
        image_paths = image_paths[:args.max_frames]
    if not image_paths:
        raise RuntimeError(f"No JPG frames found in {args.frames_dir}")

    first = cv2.imread(str(image_paths[0]))
    if first is None:
        raise RuntimeError(f"Cannot read first frame: {image_paths[0]}")
    h, w = first.shape[:2]

    output_path = args.output or args.frames_dir.parent / "cam68_yolo_only_raw.mp4"
    summary_path = args.summary or output_path.with_suffix(".summary.json")
    preview_path = args.preview or output_path.with_suffix(".preview.jpg")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    out_w = int(round(w * args.display_scale))
    out_h = int(round(h * args.display_scale))
    if out_w % 2:
        out_w += 1
    if out_h % 2:
        out_h += 1
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        args.fps,
        (out_w, out_h),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not open VideoWriter: {output_path}")

    model = YOLO(str(args.model))
    stats = {
        "frames": len(image_paths),
        "frames_with_detections": 0,
        "detections": 0,
        "max_detections_in_frame": 0,
        "conf_sum": 0.0,
        "sample_frames_with_detections": [],
    }
    start = time.time()

    try:
        for seq, image_path in enumerate(image_paths, start=1):
            frame_index = _safe_frame_index(image_path)
            frame = cv2.imread(str(image_path))
            if frame is None:
                frame = np.zeros((h, w, 3), dtype=np.uint8)

            results = model.predict(frame, conf=args.conf, device=args.device, verbose=False)
            result = results[0] if results else None
            count = 0
            if result is not None and result.boxes is not None:
                count = len(result.boxes)
                if result.boxes.conf is not None and count:
                    stats["conf_sum"] += float(np.sum(result.boxes.conf.cpu().numpy()))

            stats["detections"] += count
            stats["max_detections_in_frame"] = max(stats["max_detections_in_frame"], count)
            if count:
                stats["frames_with_detections"] += 1
                if len(stats["sample_frames_with_detections"]) < 30:
                    stats["sample_frames_with_detections"].append(frame_index)

            annotated = result.plot() if result is not None else frame
            _draw_text(
                annotated,
                f"YOLO only | frame {frame_index:05d} | boxes {count} | conf >= {args.conf:.2f}",
                (18, 34),
            )
            if args.display_scale != 1.0:
                annotated = cv2.resize(annotated, (out_w, out_h), interpolation=cv2.INTER_AREA)
            writer.write(annotated)

            if args.preview_frame is not None and frame_index == args.preview_frame:
                cv2.imwrite(str(preview_path), annotated)

            if args.progress_every > 0 and seq % args.progress_every == 0:
                elapsed = time.time() - start
                print(
                    f"processed {seq}/{len(image_paths)} frames, "
                    f"frames_with_det={stats['frames_with_detections']}, boxes={stats['detections']}, "
                    f"elapsed={elapsed:.1f}s",
                    flush=True,
                )
    finally:
        writer.release()

    if args.preview_frame is None:
        cap = cv2.VideoCapture(str(output_path))
        mid = max(0, len(image_paths) // 2)
        cap.set(cv2.CAP_PROP_POS_FRAMES, mid)
        ok, preview = cap.read()
        cap.release()
        if ok and preview is not None:
            cv2.imwrite(str(preview_path), preview)

    runtime = time.time() - start
    avg_conf = stats["conf_sum"] / stats["detections"] if stats["detections"] else 0.0
    summary = {
        "mode": "yolo_only_predict",
        "frames_dir": str(args.frames_dir),
        "model": str(args.model),
        "conf": args.conf,
        "device": args.device,
        "outputs": {
            "video": str(output_path),
            "summary": str(summary_path),
            "preview": str(preview_path) if preview_path.exists() else None,
        },
        "stats": {
            **stats,
            "detection_frame_rate": round(stats["frames_with_detections"] / max(1, len(image_paths)), 4),
            "avg_boxes_per_frame": round(stats["detections"] / max(1, len(image_paths)), 4),
            "avg_confidence": round(avg_conf, 4),
        },
        "render": {
            "fps": args.fps,
            "width": out_w,
            "height": out_h,
            "display_scale": args.display_scale,
        },
        "runtime_seconds": round(runtime, 2),
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "video": str(output_path),
        "summary": str(summary_path),
        "preview": str(preview_path) if preview_path.exists() else None,
        "frames_with_detections": stats["frames_with_detections"],
        "detections": stats["detections"],
    }, ensure_ascii=False, indent=2), flush=True)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render YOLO-only raw detections for an image sequence.")
    parser.add_argument("--frames-dir", type=Path, default=DEFAULT_FRAMES_DIR)
    parser.add_argument("--model", type=Path, default=Path("yolo_roadmap/best.pt"))
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--summary", type=Path, default=None)
    parser.add_argument("--preview", type=Path, default=None)
    parser.add_argument("--preview-frame", type=int, default=None)
    parser.add_argument("--conf", type=float, default=0.2)
    parser.add_argument("--device", default="0")
    parser.add_argument("--fps", type=float, default=25.0)
    parser.add_argument("--display-scale", type=float, default=0.75)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--progress-every", type=int, default=250)
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
