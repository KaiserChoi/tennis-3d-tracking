"""Overlay annotations on images to visually verify ball positions."""

import json
import os

import cv2
import numpy as np


def main():
    cam66_dir = r"D:\tennis\cam66_20260306_170528"
    cam68_dir = r"D:\tennis\cam68_20260306_170528"

    # Key frame pairs with offset=14: (cam66_frame, cam68_frame)
    pairs = [
        (122, 108, "start - ball high"),
        (124, 110, "descending"),
        (127, 113, "mid-flight"),
        (131, 117, "near bounce"),
        (133, 119, "after bounce"),
        (138, 124, "rising"),
        (142, 128, "end"),
    ]

    rows = []
    for c66, c68, label in pairs:
        # Load images
        img66_path = os.path.join(cam66_dir, f"{c66:05d}.jpg")
        img68_path = os.path.join(cam68_dir, f"{c68:05d}.jpg")

        if not os.path.exists(img66_path) or not os.path.exists(img68_path):
            print(f"Skipping {c66}/{c68}: file not found")
            continue

        img66 = cv2.imread(img66_path)
        img68 = cv2.imread(img68_path)

        # Load annotations
        ann66_path = os.path.join(cam66_dir, f"{c66:05d}.json")
        ann68_path = os.path.join(cam68_dir, f"{c68:05d}.json")

        if os.path.exists(ann66_path):
            with open(ann66_path) as f:
                d = json.load(f)
            px, py = d["shapes"][0]["points"][0]
            # Draw crosshair
            ix, iy = int(px), int(py)
            cv2.circle(img66, (ix, iy), 20, (0, 255, 0), 2)
            cv2.circle(img66, (ix, iy), 4, (0, 255, 0), -1)
            cv2.line(img66, (ix - 30, iy), (ix + 30, iy), (0, 255, 0), 1)
            cv2.line(img66, (ix, iy - 30), (ix, iy + 30), (0, 255, 0), 1)

        if os.path.exists(ann68_path):
            with open(ann68_path) as f:
                d = json.load(f)
            px, py = d["shapes"][0]["points"][0]
            ix, iy = int(px), int(py)
            cv2.circle(img68, (ix, iy), 20, (0, 255, 0), 2)
            cv2.circle(img68, (ix, iy), 4, (0, 255, 0), -1)
            cv2.line(img68, (ix - 30, iy), (ix + 30, iy), (0, 255, 0), 1)
            cv2.line(img68, (ix, iy - 30), (ix, iy + 30), (0, 255, 0), 1)

        # Resize to 640 wide for display
        scale = 640.0 / img66.shape[1]
        img66_sm = cv2.resize(img66, (640, int(img66.shape[0] * scale)))
        img68_sm = cv2.resize(img68, (640, int(img68.shape[0] * scale)))

        # Add labels
        cv2.putText(img66_sm, f"cam66 #{c66} {label}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(img68_sm, f"cam68 #{c68} (offset=14)", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # Combine side by side
        combined = np.hstack([img66_sm, img68_sm])
        rows.append(combined)

    if rows:
        # Stack all rows
        full = np.vstack(rows)
        out_path = r"D:\tennis\annotation_overlay.jpg"
        cv2.imwrite(out_path, full, [cv2.IMWRITE_JPEG_QUALITY, 90])
        print(f"Saved overlay to {out_path}")
        print(f"Image size: {full.shape[1]}x{full.shape[0]}")
    else:
        print("No valid pairs found")


if __name__ == "__main__":
    main()
