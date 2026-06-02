from __future__ import annotations

import argparse
import csv
import json
from collections import deque
from pathlib import Path
from typing import Any

import cv2
import numpy as np


class KalmanFilter2D:
    def __init__(self, dt: float = 1.0):
        self.dt = dt
        self.x = np.zeros((4, 1))
        self.P = np.eye(4) * 100
        self.F = np.array(
            [
                [1, 0, dt, 0],
                [0, 1, 0, dt],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ],
            dtype=np.float64,
        )
        self.H = np.array(
            [
                [1, 0, 0, 0],
                [0, 1, 0, 0],
            ],
            dtype=np.float64,
        )
        self.Q = np.array(
            [
                [0.5, 0, 0, 0],
                [0, 0.5, 0, 0],
                [0, 0, 0.1, 0],
                [0, 0, 0, 0.1],
            ],
            dtype=np.float64,
        )
        self.R = np.array(
            [
                [8, 0],
                [0, 8],
            ],
            dtype=np.float64,
        )

    def predict(self) -> np.ndarray:
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x[0:2].flatten()

    def update(self, z: np.ndarray) -> np.ndarray:
        z = z.reshape(-1, 1)
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(4) - K @ self.H) @ self.P
        return self.x[0:2].flatten()


def _load_predictions(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    frames = payload.get("frames")
    if not isinstance(frames, list):
        raise ValueError(f"Invalid predictions JSON, missing frames list: {path}")
    return payload


def _clip_xy(x: float, y: float, width: int, height: int) -> tuple[float, float]:
    return (
        float(np.clip(x, 0, max(0, width - 1))),
        float(np.clip(y, 0, max(0, height - 1))),
    )


def process_with_kalman(
    frames: list[dict[str, Any]],
    *,
    width: int,
    height: int,
    conf_threshold: float,
    max_consecutive_pred: int,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    kf: KalmanFilter2D | None = None
    initialized = False
    consecutive_pred = 0

    for point in frames:
        frame_idx = int(point["frame_index"])
        prob_raw = point.get("prob")
        has_measurement = (
            bool(point.get("visible"))
            and point.get("x") is not None
            and point.get("y") is not None
            and prob_raw is not None
            and float(prob_raw) >= conf_threshold
        )

        measured_x = float(point["x"]) if has_measurement else None
        measured_y = float(point["y"]) if has_measurement else None
        prob = float(prob_raw) if has_measurement else 0.0

        if not has_measurement:
            if initialized and kf is not None:
                pred_x, pred_y = kf.predict()
                pred_x, pred_y = _clip_xy(float(pred_x), float(pred_y), width, height)
                consecutive_pred += 1
                if consecutive_pred > max_consecutive_pred:
                    kf.x[0, 0] = pred_x
                    kf.x[1, 0] = pred_y
                    kf.x[2, 0] = 0
                    kf.x[3, 0] = 0
                    consecutive_pred = 0
                results.append(
                    {
                        **point,
                        "visible": True,
                        "x": round(pred_x, 2),
                        "y": round(pred_y, 2),
                        "prob": 0.0,
                        "source": "kalman",
                        "is_predicted": True,
                        "measured_x": None,
                        "measured_y": None,
                        "measured_prob": None,
                    }
                )
            else:
                results.append(
                    {
                        **point,
                        "visible": False,
                        "x": None,
                        "y": None,
                        "prob": 0.0,
                        "source": "none",
                        "is_predicted": False,
                        "measured_x": None,
                        "measured_y": None,
                        "measured_prob": None,
                    }
                )
            continue

        consecutive_pred = 0
        if not initialized:
            kf = KalmanFilter2D()
            kf.x[0, 0] = measured_x
            kf.x[1, 0] = measured_y
            kf.x[2, 0] = 0
            kf.x[3, 0] = 0
            initialized = True
            filt_x, filt_y = measured_x, measured_y
        else:
            assert kf is not None
            kf.predict()
            kf.update(np.array([measured_x, measured_y], dtype=np.float64))
            filt_x, filt_y = _clip_xy(float(kf.x[0, 0]), float(kf.x[1, 0]), width, height)
            kf.x[0, 0] = filt_x
            kf.x[1, 0] = filt_y

        results.append(
            {
                **point,
                "visible": True,
                "x": round(float(filt_x), 2),
                "y": round(float(filt_y), 2),
                "prob": round(prob, 6),
                "source": "tracknet_kalman_update",
                "is_predicted": False,
                "measured_x": round(float(measured_x), 2),
                "measured_y": round(float(measured_y), 2),
                "measured_prob": round(prob, 6),
            }
        )

    return results


def _draw_marker(
    canvas: np.ndarray,
    xy: tuple[int, int],
    *,
    color: tuple[int, int, int],
    predicted: bool,
) -> None:
    if predicted:
        cv2.circle(canvas, xy, 10, color, 2, cv2.LINE_AA)
        cv2.circle(canvas, xy, 3, color, -1, cv2.LINE_AA)
    else:
        cv2.drawMarker(canvas, xy, color, markerType=cv2.MARKER_CROSS, markerSize=20, thickness=2)
        cv2.circle(canvas, xy, 9, color, 2, cv2.LINE_AA)


def draw_kalman_prediction(
    frame_bgr: np.ndarray,
    point: dict[str, Any],
    *,
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

    measured_xy: tuple[int, int] | None = None
    if point.get("measured_x") is not None and point.get("measured_y") is not None:
        measured_xy = (
            int(round(float(point["measured_x"]) * display_scale)),
            int(round(float(point["measured_y"]) * display_scale)),
        )

    selected_xy: tuple[int, int] | None = None
    if point.get("visible") and point.get("x") is not None and point.get("y") is not None:
        selected_xy = (
            int(round(float(point["x"]) * display_scale)),
            int(round(float(point["y"]) * display_scale)),
        )
        trail.append(selected_xy)

    if measured_xy is not None and selected_xy is not None and measured_xy != selected_xy:
        cv2.drawMarker(
            canvas,
            measured_xy,
            (255, 120, 0),
            markerType=cv2.MARKER_TILTED_CROSS,
            markerSize=16,
            thickness=2,
        )
        cv2.circle(canvas, measured_xy, 7, (255, 120, 0), 1, cv2.LINE_AA)

    if len(trail) >= 2:
        pts = list(trail)
        for i in range(1, len(pts)):
            alpha = i / max(1, len(pts) - 1)
            color = (0, int(80 + 120 * alpha), 255)
            cv2.line(canvas, pts[i - 1], pts[i], color, 2, cv2.LINE_AA)

    if selected_xy is not None:
        predicted = bool(point.get("is_predicted"))
        color = (255, 220, 0) if predicted else (0, 0, 255)
        _draw_marker(canvas, selected_xy, color=color, predicted=predicted)

    source = str(point.get("source", "none"))
    label = (
        f"F{int(point['frame_index']):05d}  "
        f"{source}  prob={float(point.get('prob') or 0.0):.3f}"
    )
    cv2.putText(canvas, label, (18, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.78, (0, 0, 0), 5, cv2.LINE_AA)
    cv2.putText(canvas, label, (18, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.78, (255, 255, 255), 2, cv2.LINE_AA)
    return canvas


def _write_csv(path: Path, frames: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "frame_index",
                "x",
                "y",
                "prob",
                "is_predicted",
                "source",
                "measured_x",
                "measured_y",
                "measured_prob",
            ]
        )
        for point in frames:
            writer.writerow(
                [
                    point["frame_index"],
                    "" if point.get("x") is None else point["x"],
                    "" if point.get("y") is None else point["y"],
                    point.get("prob", 0.0),
                    point.get("is_predicted", False),
                    point.get("source", ""),
                    "" if point.get("measured_x") is None else point["measured_x"],
                    "" if point.get("measured_y") is None else point["measured_y"],
                    "" if point.get("measured_prob") is None else point["measured_prob"],
                ]
            )


def render_kalman_video(
    *,
    predictions_path: Path,
    output_path: Path,
    json_output_path: Path,
    csv_output_path: Path,
    fps: float,
    display_scale: float,
    trail_len: int,
    conf_threshold: float,
    max_consecutive_pred: int,
) -> dict[str, Any]:
    payload = _load_predictions(predictions_path)
    image_size = payload.get("image_size") or {}
    width = int(image_size.get("width") or payload.get("original_size", [1080, 1920])[1])
    height = int(image_size.get("height") or payload.get("original_size", [1080, 1920])[0])
    frames_dir = Path(payload.get("frames_dir") or predictions_path.parent)
    source_frames = payload["frames"]
    filtered_frames = process_with_kalman(
        source_frames,
        width=width,
        height=height,
        conf_threshold=conf_threshold,
        max_consecutive_pred=max_consecutive_pred,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    json_output_path.parent.mkdir(parents=True, exist_ok=True)
    csv_output_path.parent.mkdir(parents=True, exist_ok=True)

    video_w = int(round(width * display_scale))
    video_h = int(round(height * display_scale))
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (video_w, video_h),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not open VideoWriter: {output_path}")

    trail: deque[tuple[int, int]] = deque(maxlen=trail_len)
    written = 0
    try:
        for point in filtered_frames:
            frame_path = frames_dir / str(point.get("image", f"{int(point['frame_index']):05d}.jpg"))
            frame = cv2.imread(str(frame_path))
            if frame is None:
                frame = np.zeros((height, width, 3), dtype=np.uint8)
            canvas = draw_kalman_prediction(frame, point, display_scale=display_scale, trail=trail)
            writer.write(canvas)
            written += 1
            if written and written % 500 == 0:
                print(f"wrote {written}/{len(filtered_frames)} frames")
    finally:
        writer.release()

    predicted_count = sum(1 for p in filtered_frames if p.get("is_predicted"))
    measured_count = sum(1 for p in filtered_frames if p.get("source") == "tracknet_kalman_update")
    none_count = sum(1 for p in filtered_frames if p.get("source") == "none")
    filtered_payload = {
        "source_predictions_path": str(predictions_path),
        "frames_dir": str(frames_dir),
        "output_video_path": str(output_path),
        "coordinate_space": "original_image_pixels",
        "image_size": {"width": width, "height": height},
        "conf_threshold": conf_threshold,
        "max_consecutive_pred": max_consecutive_pred,
        "frames": filtered_frames,
    }
    json_output_path.write_text(
        json.dumps(filtered_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    _write_csv(csv_output_path, filtered_frames)

    summary = {
        "source_predictions_path": str(predictions_path),
        "output_video_path": str(output_path),
        "json_output_path": str(json_output_path),
        "csv_output_path": str(csv_output_path),
        "frames": written,
        "measured_frames": measured_count,
        "kalman_predicted_frames": predicted_count,
        "none_frames": none_count,
        "fps": fps,
        "display_scale": display_scale,
        "conf_threshold": conf_threshold,
    }
    summary_path = output_path.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Fill TrackNet missing frames with Kalman filter and render video.")
    parser.add_argument(
        "--predictions",
        type=Path,
        default=Path(r"D:\tennis-dataset\1001\clip3\cam68_tracknet_prediction_vis.predictions.json"),
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--json-output", type=Path, default=None)
    parser.add_argument("--csv-output", type=Path, default=None)
    parser.add_argument("--fps", type=float, default=25.0)
    parser.add_argument("--display-scale", type=float, default=0.75)
    parser.add_argument("--trail-len", type=int, default=30)
    parser.add_argument("--conf-threshold", type=float, default=0.5)
    parser.add_argument("--max-consecutive-pred", type=int, default=10)
    args = parser.parse_args()

    base = args.predictions
    output = args.output or base.with_name(base.name.replace(".predictions.json", "_kalman.mp4"))
    json_output = args.json_output or base.with_name(base.name.replace(".predictions.json", ".kalman.json"))
    csv_output = args.csv_output or base.with_name(base.name.replace(".predictions.json", ".kalman.csv"))
    summary = render_kalman_video(
        predictions_path=args.predictions,
        output_path=output,
        json_output_path=json_output,
        csv_output_path=csv_output,
        fps=args.fps,
        display_scale=args.display_scale,
        trail_len=args.trail_len,
        conf_threshold=args.conf_threshold,
        max_consecutive_pred=args.max_consecutive_pred,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
