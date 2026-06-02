from __future__ import annotations

import argparse
import json
from collections import deque
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml

from app.pipeline.inference import TrackNetDetector


def draw_prediction(
    frame_bgr: np.ndarray,
    *,
    frame_index: int,
    pred_xy_model: tuple[int, int] | None,
    pred_prob: float | None,
    raw_xy_model: tuple[int, int] | None,
    raw_prob: float | None,
    reason: str,
    image_scale: float,
    display_scale: float,
    trail: deque[tuple[int, int]],
) -> np.ndarray:
    if display_scale != 1.0:
        h, w = frame_bgr.shape[:2]
        canvas = cv2.resize(
            frame_bgr,
            (int(round(w * display_scale)), int(round(h * display_scale))),
            interpolation=cv2.INTER_AREA,
        )
    else:
        canvas = frame_bgr.copy()

    selected_xy: tuple[int, int] | None = None
    if pred_xy_model is not None:
        x_model, y_model = pred_xy_model
        x = int(round(x_model / image_scale * display_scale))
        y = int(round(y_model / image_scale * display_scale))
        selected_xy = (x, y)
        trail.append(selected_xy)

    if raw_xy_model is not None and raw_prob is not None:
        raw_x = int(round(raw_xy_model[0] / image_scale * display_scale))
        raw_y = int(round(raw_xy_model[1] / image_scale * display_scale))
        if selected_xy is not None and (raw_x, raw_y) != selected_xy:
            cv2.drawMarker(canvas, (raw_x, raw_y), (255, 120, 0), markerType=cv2.MARKER_TILTED_CROSS, markerSize=16, thickness=2)
            cv2.circle(canvas, (raw_x, raw_y), 7, (255, 120, 0), 1, cv2.LINE_AA)

    if len(trail) >= 2:
        pts = list(trail)
        for i in range(1, len(pts)):
            alpha = i / max(1, len(pts) - 1)
            color = (0, int(80 + 120 * alpha), 255)
            cv2.line(canvas, pts[i - 1], pts[i], color, 2, cv2.LINE_AA)

    if selected_xy is not None:
        cv2.drawMarker(canvas, selected_xy, (0, 0, 255), markerType=cv2.MARKER_CROSS, markerSize=20, thickness=2)
        cv2.circle(canvas, selected_xy, 9, (0, 0, 255), 2, cv2.LINE_AA)

    label = f"F{frame_index:05d}  raw={raw_prob or 0.0:.3f}  pred={pred_prob or 0.0:.3f}  {reason}"
    cv2.putText(canvas, label, (18, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.78, (0, 0, 0), 5, cv2.LINE_AA)
    cv2.putText(canvas, label, (18, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.78, (255, 255, 255), 2, cv2.LINE_AA)
    return canvas


def _load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a dict: {path}")
    return data


def _sorted_images(frames_dir: Path) -> list[Path]:
    images = sorted(
        [p for p in frames_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}],
        key=lambda p: (int(p.stem) if p.stem.isdigit() else p.stem),
    )
    if not images:
        raise FileNotFoundError(f"No image frames found in {frames_dir}")
    return images


def _compute_bg_frame(image_paths: list[Path], input_size: tuple[int, int], max_samples: int = 200) -> np.ndarray:
    input_h, input_w = input_size
    if len(image_paths) <= max_samples:
        sample_paths = image_paths
    else:
        idxs = np.linspace(0, len(image_paths) - 1, max_samples).astype(int)
        sample_paths = [image_paths[int(i)] for i in idxs]

    frames: list[np.ndarray] = []
    for path in sample_paths:
        frame = cv2.imread(str(path))
        if frame is None:
            continue
        resized = cv2.resize(frame, (input_w, input_h), interpolation=cv2.INTER_AREA)
        frames.append(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB))
    if not frames:
        raise RuntimeError("Could not read frames for TrackNet median background")

    median_rgb = np.median(np.stack(frames, axis=0), axis=0).astype(np.uint8)
    return median_rgb.transpose(2, 0, 1).astype(np.float32) / 255.0


def _apply_heatmap_mask(
    heatmap: np.ndarray,
    mask_regions: list[list[int]],
    *,
    original_size: tuple[int, int],
    input_size: tuple[int, int],
) -> np.ndarray:
    if not mask_regions:
        return heatmap
    original_w, original_h = original_size
    input_h, input_w = input_size
    sx = input_w / max(1, original_w)
    sy = input_h / max(1, original_h)
    masked = heatmap.copy()
    for region in mask_regions:
        if len(region) != 4:
            continue
        x0, y0, x1, y1 = region
        mx0 = max(0, min(input_w, int(round(x0 * sx))))
        mx1 = max(0, min(input_w, int(round(x1 * sx))))
        my0 = max(0, min(input_h, int(round(y0 * sy))))
        my1 = max(0, min(input_h, int(round(y1 * sy))))
        if mx1 > mx0 and my1 > my0:
            masked[my0:my1, mx0:mx1] = 0.0
    return masked


def _select_blob_model(
    heatmap: np.ndarray,
    threshold: float,
) -> tuple[tuple[int, int] | None, float | None, str]:
    binary = (heatmap >= threshold).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    best_label = -1
    best_sum = -1.0
    for label in range(1, num_labels):
        x, y, w, h, area = stats[label]
        if area <= 0:
            continue
        component = heatmap[y : y + h, x : x + w]
        component_mask = labels[y : y + h, x : x + w] == label
        blob_sum = float(component[component_mask].sum())
        if blob_sum > best_sum:
            best_sum = blob_sum
            best_label = label

    if best_label < 0:
        return None, None, f"no_blob<{threshold:.2f}"

    ys, xs = np.nonzero(labels == best_label)
    weights = heatmap[ys, xs].astype(np.float64)
    total = float(weights.sum())
    if total <= 0:
        return None, None, "empty_blob"

    cx = int(round(float((xs * weights).sum() / total)))
    cy = int(round(float((ys * weights).sum() / total)))
    pred_prob = float(weights.max())
    return (cx, cy), pred_prob, f"blob_sum={best_sum:.2f}"


def _raw_peak_model(heatmap: np.ndarray) -> tuple[tuple[int, int] | None, float | None]:
    if heatmap.size == 0:
        return None, None
    flat_idx = int(np.argmax(heatmap))
    raw_y, raw_x = np.unravel_index(flat_idx, heatmap.shape)
    return (int(raw_x), int(raw_y)), float(heatmap[raw_y, raw_x])


def _model_xy_to_pixel(
    xy_model: tuple[int, int] | None,
    image_scale: float,
    *,
    width: int,
    height: int,
) -> tuple[int, int] | None:
    if xy_model is None:
        return None
    x = int(round(xy_model[0] / image_scale))
    y = int(round(xy_model[1] / image_scale))
    return max(0, min(width - 1, x)), max(0, min(height - 1, y))


def render_tracknet_video(
    *,
    frames_dir: Path,
    output_path: Path,
    config_path: Path,
    fps: float,
    display_scale: float,
    trail_len: int,
    max_frames: int | None,
) -> dict[str, Any]:
    cfg = _load_config(config_path)
    model_cfg = cfg.get("model", {}) if isinstance(cfg.get("model"), dict) else {}
    model_path = str(model_cfg.get("path", "model_weight/TrackNet_finetuned.pt"))
    input_size_list = model_cfg.get("input_size", [288, 512])
    input_size = (int(input_size_list[0]), int(input_size_list[1]))
    threshold = float(model_cfg.get("threshold", 0.3))
    device = str(model_cfg.get("device", "cuda"))
    mask_regions = model_cfg.get("heatmap_mask", []) or []
    frames_in = int(model_cfg.get("frames_in", 8))
    frames_out = int(model_cfg.get("frames_out", frames_in))

    image_paths = _sorted_images(frames_dir)
    if max_frames is not None:
        image_paths = image_paths[: max(0, max_frames)]
    if not image_paths:
        raise RuntimeError("No frames selected")

    first_frame = cv2.imread(str(image_paths[0]))
    if first_frame is None:
        raise RuntimeError(f"Cannot read first frame: {image_paths[0]}")
    original_h, original_w = first_frame.shape[:2]
    display_w = int(round(original_w * display_scale))
    display_h = int(round(original_h * display_scale))
    input_h, input_w = input_size
    sx = input_w / max(1, original_w)
    sy = input_h / max(1, original_h)
    image_scale = sx
    if abs(sx - sy) > 1e-3:
        raise ValueError(f"Input/original aspect ratio mismatch: sx={sx:.6f}, sy={sy:.6f}")

    detector = TrackNetDetector(
        model_path=model_path,
        input_size=input_size,
        frames_in=frames_in,
        frames_out=frames_out,
        device=device,
    )
    detector._bg_frame = _compute_bg_frame(image_paths, input_size)
    detector._video_median_computed = True

    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (display_w, display_h),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not open VideoWriter: {output_path}")

    trail: deque[tuple[int, int]] = deque(maxlen=trail_len)
    predictions: list[dict[str, Any]] = []
    detected = 0
    raw_over_threshold = 0
    written = 0
    try:
        for batch_start in range(0, len(image_paths), frames_out):
            batch_paths = image_paths[batch_start : batch_start + frames_in]
            if not batch_paths:
                break
            batch_frames: list[np.ndarray] = []
            for path in batch_paths:
                frame = cv2.imread(str(path))
                if frame is None:
                    frame = batch_frames[-1].copy() if batch_frames else first_frame.copy()
                batch_frames.append(frame)
            actual_count = min(frames_out, len(image_paths) - batch_start, len(batch_frames))
            while len(batch_frames) < frames_in:
                batch_frames.append(batch_frames[-1].copy())

            heatmaps = detector.infer(batch_frames)
            for offset in range(actual_count):
                frame_index = batch_start + offset
                frame = batch_frames[offset]
                heatmap = np.asarray(heatmaps[offset], dtype=np.float32)
                heatmap = _apply_heatmap_mask(
                    heatmap,
                    mask_regions,
                    original_size=(original_w, original_h),
                    input_size=input_size,
                )
                raw_xy, raw_prob = _raw_peak_model(heatmap)
                if raw_prob is not None and raw_prob >= threshold:
                    raw_over_threshold += 1
                pred_xy, pred_prob, reason = _select_blob_model(heatmap, threshold)
                if pred_xy is not None:
                    detected += 1
                pred_pixel = _model_xy_to_pixel(
                    pred_xy,
                    image_scale,
                    width=original_w,
                    height=original_h,
                )
                raw_pixel = _model_xy_to_pixel(
                    raw_xy,
                    image_scale,
                    width=original_w,
                    height=original_h,
                )
                predictions.append(
                    {
                        "frame_index": frame_index,
                        "image": image_paths[frame_index].name,
                        "visible": pred_pixel is not None,
                        "x": pred_pixel[0] if pred_pixel is not None else None,
                        "y": pred_pixel[1] if pred_pixel is not None else None,
                        "prob": round(float(pred_prob), 6) if pred_prob is not None else None,
                        "raw_x": raw_pixel[0] if raw_pixel is not None else None,
                        "raw_y": raw_pixel[1] if raw_pixel is not None else None,
                        "raw_prob": round(float(raw_prob), 6) if raw_prob is not None else None,
                        "x_model": pred_xy[0] if pred_xy is not None else None,
                        "y_model": pred_xy[1] if pred_xy is not None else None,
                        "raw_x_model": raw_xy[0] if raw_xy is not None else None,
                        "raw_y_model": raw_xy[1] if raw_xy is not None else None,
                        "reason": reason,
                    }
                )
                canvas = draw_prediction(
                    frame,
                    frame_index=frame_index,
                    pred_xy_model=pred_xy,
                    pred_prob=pred_prob,
                    raw_xy_model=raw_xy,
                    raw_prob=raw_prob,
                    reason=reason,
                    image_scale=image_scale,
                    display_scale=display_scale,
                    trail=trail,
                )
                writer.write(canvas)
                written += 1
            if written and written % 500 == 0:
                print(f"wrote {written}/{len(image_paths)} frames")
    finally:
        writer.release()

    summary = {
        "frames_dir": str(frames_dir),
        "output_path": str(output_path),
        "config_path": str(config_path),
        "frames": written,
        "fps": fps,
        "display_scale": display_scale,
        "threshold": threshold,
        "detected_frames": detected,
        "raw_over_threshold_frames": raw_over_threshold,
        "image_scale": image_scale,
        "model_input_size": [input_h, input_w],
        "original_size": [original_h, original_w],
    }
    summary_path = output_path.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    predictions_path = output_path.with_suffix(".predictions.json")
    predictions_payload = {
        "frames_dir": str(frames_dir),
        "output_path": str(output_path),
        "summary_path": str(summary_path),
        "coordinate_space": "original_image_pixels",
        "image_size": {"width": original_w, "height": original_h},
        "model_input_size": {"width": input_w, "height": input_h},
        "threshold": threshold,
        "fps": fps,
        "frames": predictions,
    }
    predictions_path.write_text(
        json.dumps(predictions_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    summary["predictions_path"] = str(predictions_path)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Render TrackNet predictions on a single-camera frame folder.")
    parser.add_argument(
        "--frames-dir",
        type=Path,
        default=Path(r"D:\tennis-dataset\1001\clip3\cam68_20260404_075325_2min"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(r"D:\tennis-dataset\1001\clip3\cam68_tracknet_prediction_vis.mp4"),
    )
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument("--fps", type=float, default=25.0)
    parser.add_argument("--display-scale", type=float, default=0.75)
    parser.add_argument("--trail-len", type=int, default=30)
    parser.add_argument("--max-frames", type=int, default=None)
    args = parser.parse_args()

    summary = render_tracknet_video(
        frames_dir=args.frames_dir,
        output_path=args.output,
        config_path=args.config,
        fps=args.fps,
        display_scale=args.display_scale,
        trail_len=args.trail_len,
        max_frames=args.max_frames,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
