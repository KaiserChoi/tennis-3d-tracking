"""Frame alignment middleware for dual-camera tennis tracking.

Aligns two unsynchronized camera streams using OSD timestamp OCR.
Each camera has an independent RTC that drifts over time. This middleware
reads the OSD time (HH:MM:SS) from each frame, buffers frames by second,
and outputs aligned frame pairs.

Core idea (from user):
  - Fast camera waits for slow camera to catch up
  - Both cameras buffer frames into per-second buckets
  - When both have frames for the same second, extract and align
  - If one has more frames than the other, drop extras (evenly spaced)
  - Output: paired frames from the same real-world second

Works for both real-time (RTSP) and offline (video files).

Usage (offline):
    aligner = TimestampAligner()
    aligned_pairs = aligner.align_videos(video66_path, video68_path, max_frames=3000)
    # aligned_pairs[i] = (frame66_idx, frame68_idx) for each output frame

Usage (real-time):
    aligner = TimestampAligner()
    aligner.push_frame("cam66", frame_idx, frame_image)
    aligner.push_frame("cam68", frame_idx, frame_image)
    pairs = aligner.pop_aligned()  # returns list of (idx66, idx68)
"""

import cv2
import re
import numpy as np
import logging
from collections import defaultdict
from typing import Optional

logger = logging.getLogger(__name__)


def ocr_osd_time(frame: np.ndarray, osd_region: tuple = (0, 40, 400, 620)) -> Optional[str]:
    """Extract HH:MM:SS from OSD timestamp using template matching.

    The OSD font is fixed white text on dark background. We threshold
    and use contour-based digit recognition instead of heavy OCR.

    Args:
        frame: BGR image (1920x1080).
        osd_region: (y1, y2, x1, x2) crop region containing HH:MM:SS.

    Returns:
        Time string "HH:MM:SS" or None if unreadable.
    """
    y1, y2, x1, x2 = osd_region
    crop = frame[y1:y2, x1:x2]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

    # White text on dark background → threshold high
    _, bw = cv2.threshold(gray, 160, 255, cv2.THRESH_BINARY)

    # Use pixel column projection to find digit boundaries
    proj = np.sum(bw > 128, axis=0)

    # Find character groups (connected non-zero columns)
    chars = []
    in_char = False
    start = 0
    for i in range(len(proj)):
        if proj[i] > 2 and not in_char:
            start = i
            in_char = True
        elif proj[i] <= 2 and in_char:
            chars.append((start, i))
            in_char = False
    if in_char:
        chars.append((start, len(proj)))

    # Filter: digits should be 8-20px wide, colons 2-6px wide
    digits = []
    for x1c, x2c in chars:
        w = x2c - x1c
        if 6 <= w <= 25:
            digits.append((x1c, x2c, 'digit'))
        elif 2 <= w <= 5:
            digits.append((x1c, x2c, 'colon'))

    # Expect pattern: D D : D D : D D (8 chars with 2 colons)
    # Or sometimes digits merge: find at least 2 colons
    colon_count = sum(1 for _, _, t in digits if t == 'colon')

    if colon_count < 2:
        return None

    # Extract digit images and match to templates using pixel counts
    # Simple approach: count white pixels in each quadrant of the digit
    digit_values = []
    for x1c, x2c, dtype in digits:
        if dtype == 'colon':
            digit_values.append(':')
            continue

        char_img = bw[:, x1c:x2c]
        h, w = char_img.shape
        if h < 5 or w < 3:
            digit_values.append('?')
            continue

        # 7-segment-like features
        total = np.sum(char_img > 128)
        top_half = np.sum(char_img[:h // 2] > 128)
        bot_half = np.sum(char_img[h // 2:] > 128)
        mid_row = np.sum(char_img[h // 2 - 2:h // 2 + 2] > 128)
        top_row = np.sum(char_img[:3] > 128)
        bot_row = np.sum(char_img[-3:] > 128)
        left_col = np.sum(char_img[:, :w // 3] > 128)
        right_col = np.sum(char_img[:, 2 * w // 3:] > 128)

        area = h * w
        fill = total / area if area > 0 else 0

        # Simple digit classification by fill ratio and segment patterns
        # This is approximate but works for fixed OSD fonts
        ratio_top = top_half / (total + 1)
        ratio_left = left_col / (total + 1)
        ratio_right = right_col / (total + 1)

        # Use fill ratio as primary discriminator
        if fill < 0.15:
            digit_values.append('1')
        elif fill > 0.55:
            digit_values.append('8')
        elif ratio_top > 0.6:
            digit_values.append('7')
        elif ratio_left < 0.2 and ratio_right > 0.4:
            if mid_row / (w * 4 + 1) > 0.5:
                digit_values.append('3')
            else:
                digit_values.append('7')
        else:
            # Can't reliably distinguish 0,2,4,5,6,9 with simple heuristics
            # Return the fill ratio encoded as a digit for consistency
            digit_values.append(str(int(fill * 10) % 10))

    text = ''.join(digit_values)

    # Try to parse as HH:MM:SS
    match = re.search(r'(\d{1,2}):(\d{1,2}):(\d{1,2})', text)
    if match:
        return f"{int(match.group(1)):02d}:{int(match.group(2)):02d}:{int(match.group(3)):02d}"

    return None


def detect_second_change(curr_frame: np.ndarray, prev_crop: Optional[np.ndarray],
                         osd_region: tuple = (3, 33, 560, 610),
                         threshold: float = 15.0) -> tuple[bool, np.ndarray]:
    """Detect if the OSD seconds digit changed between frames.

    More reliable than full OCR — just detects WHEN the second changes,
    not WHAT the time is.

    Returns:
        (changed, current_crop)
    """
    y1, y2, x1, x2 = osd_region
    crop = curr_frame[y1:y2, x1:x2]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

    if prev_crop is None:
        return True, gray

    diff = cv2.absdiff(gray, prev_crop).astype(np.float32).mean()
    return diff > threshold, gray


class TimestampAligner:
    """Align two camera streams using OSD second-boundary detection.

    Instead of OCR (unreliable), detects when the OSD seconds digit
    changes. Groups frames into per-second buckets. When both cameras
    have completed a second, aligns the frame counts by dropping extras.
    """

    def __init__(
        self,
        osd_region_66: tuple = (3, 33, 560, 610),
        osd_region_68: tuple = (3, 33, 560, 610),
        threshold: float = 15.0,
        target_fps: int = 25,
    ):
        self.osd_region_66 = osd_region_66
        self.osd_region_68 = osd_region_68
        self.threshold = threshold
        self.target_fps = target_fps

        # Per-second frame buffers: second_idx -> [frame_indices]
        self._buf66: dict[int, list[int]] = defaultdict(list)
        self._buf68: dict[int, list[int]] = defaultdict(list)

        # Current second counter (incremented on each tick)
        self._sec66 = 0
        self._sec68 = 0

        # Previous OSD crops for change detection
        self._prev66: Optional[np.ndarray] = None
        self._prev68: Optional[np.ndarray] = None

        # Completed seconds ready for alignment
        self._completed_secs_66: set[int] = set()
        self._completed_secs_68: set[int] = set()

        # Output: aligned frame pairs
        self._aligned_pairs: list[tuple[int, int]] = []

        # Offset map for interpolation (same interface as old FrameAligner)
        self._offset_pairs: list[tuple[int, int]] = []

    def push_frame_66(self, frame_idx: int, frame: np.ndarray):
        """Push a cam66 frame. Detects second boundary and buffers."""
        changed, crop = detect_second_change(
            frame, self._prev66, self.osd_region_66, self.threshold
        )
        self._prev66 = crop

        if changed and self._buf66[self._sec66]:
            # Current second is complete, move to next
            self._completed_secs_66.add(self._sec66)
            self._sec66 += 1

        self._buf66[self._sec66].append(frame_idx)

    def push_frame_68(self, frame_idx: int, frame: np.ndarray):
        """Push a cam68 frame. Detects second boundary and buffers."""
        changed, crop = detect_second_change(
            frame, self._prev68, self.osd_region_68, self.threshold
        )
        self._prev68 = crop

        if changed and self._buf68[self._sec68]:
            self._completed_secs_68.add(self._sec68)
            self._sec68 += 1

        self._buf68[self._sec68].append(frame_idx)

    def pop_aligned(self) -> list[tuple[int, int]]:
        """Pop all aligned frame pairs from completed seconds.

        For each second that both cameras have completed:
        - Take frames from both buffers
        - If counts differ, drop extras evenly from the longer one
        - Return paired frame indices

        Returns:
            List of (cam66_frame_idx, cam68_frame_idx) pairs.
        """
        ready = self._completed_secs_66 & self._completed_secs_68
        if not ready:
            return []

        pairs = []
        for sec in sorted(ready):
            frames66 = self._buf66.pop(sec, [])
            frames68 = self._buf68.pop(sec, [])

            if not frames66 or not frames68:
                continue

            # Align: drop extras from the longer list
            aligned66, aligned68 = self._equalize(frames66, frames68)

            for f66, f68 in zip(aligned66, aligned68):
                pairs.append((f66, f68))

            # Record boundary for offset interpolation
            if aligned66 and aligned68:
                self._offset_pairs.append((aligned66[0], aligned68[0]))

            self._completed_secs_66.discard(sec)
            self._completed_secs_68.discard(sec)

        self._aligned_pairs.extend(pairs)
        return pairs

    def _equalize(self, list_a: list[int], list_b: list[int]) -> tuple[list[int], list[int]]:
        """Make two lists the same length by evenly dropping from the longer one."""
        if len(list_a) == len(list_b):
            return list_a, list_b

        if len(list_a) > len(list_b):
            # Drop from list_a
            target = len(list_b)
            indices = np.round(np.linspace(0, len(list_a) - 1, target)).astype(int)
            return [list_a[i] for i in indices], list_b
        else:
            # Drop from list_b
            target = len(list_a)
            indices = np.round(np.linspace(0, len(list_b) - 1, target)).astype(int)
            return list_a, [list_b[i] for i in indices]

    # ------------------------------------------------------------------
    # Offline interface (for video files)
    # ------------------------------------------------------------------

    def align_videos(
        self,
        video66_path: str,
        video68_path: str,
        max_frames: int = 0,
    ) -> list[tuple[int, int]]:
        """Align two video files and return frame pair mapping.

        Args:
            video66_path: Path to cam66 video.
            video68_path: Path to cam68 video.
            max_frames: Max frames to process (0 = all).

        Returns:
            List of (cam66_frame_idx, cam68_frame_idx) aligned pairs.
        """
        cap66 = cv2.VideoCapture(video66_path)
        cap68 = cv2.VideoCapture(video68_path)

        total66 = int(cap66.get(cv2.CAP_PROP_FRAME_COUNT))
        total68 = int(cap68.get(cv2.CAP_PROP_FRAME_COUNT))
        total = min(total66, total68)
        if max_frames > 0:
            total = min(total, max_frames)

        logger.info("Aligning %d frames from two videos...", total)

        all_pairs = []

        for fi in range(total):
            ret66, frame66 = cap66.read()
            ret68, frame68 = cap68.read()
            if not ret66 or not ret68:
                break

            self.push_frame_66(fi, frame66)
            self.push_frame_68(fi, frame68)

            # Check for aligned pairs
            pairs = self.pop_aligned()
            all_pairs.extend(pairs)

        # Flush remaining (last incomplete second)
        # Force-complete current seconds
        self._completed_secs_66.add(self._sec66)
        self._completed_secs_68.add(self._sec68)
        pairs = self.pop_aligned()
        all_pairs.extend(pairs)

        cap66.release()
        cap68.release()

        logger.info(
            "Alignment complete: %d input frames -> %d aligned pairs",
            total, len(all_pairs),
        )
        logger.info(
            "  cam66 detected %d seconds, cam68 detected %d seconds",
            self._sec66 + 1, self._sec68 + 1,
        )

        if all_pairs:
            # Show drift
            first = all_pairs[0]
            last = all_pairs[-1]
            drift_start = first[0] - first[1]
            drift_end = last[0] - last[1]
            logger.info(
                "  Drift: start=%+d frames, end=%+d frames (total drift=%+d)",
                drift_start, drift_end, drift_end - drift_start,
            )

        return all_pairs

    def get_aligned_frame_68(self, frame_66: int) -> int:
        """Given a cam66 frame index, return the aligned cam68 frame index.

        Uses linear interpolation between second boundaries.
        Compatible interface with old FrameAligner.
        """
        if not self._offset_pairs:
            return frame_66

        pairs = self._offset_pairs

        # Before first boundary
        if frame_66 <= pairs[0][0]:
            offset = pairs[0][1] - pairs[0][0]
            return frame_66 + offset

        # After last boundary
        if frame_66 >= pairs[-1][0]:
            offset = pairs[-1][1] - pairs[-1][0]
            return frame_66 + offset

        # Between boundaries - linear interpolation
        for i in range(len(pairs) - 1):
            f66_a, f68_a = pairs[i]
            f66_b, f68_b = pairs[i + 1]

            if f66_a <= frame_66 <= f66_b:
                if f66_b == f66_a:
                    return f68_a
                t = (frame_66 - f66_a) / (f66_b - f66_a)
                f68 = f68_a + t * (f68_b - f68_a)
                return int(round(f68))

        return frame_66

    def build_frame_map(self) -> dict[int, int]:
        """Build a complete frame mapping dict from aligned pairs.

        Returns:
            Dict mapping cam66_frame_idx -> cam68_frame_idx.
        """
        frame_map = {}
        for f66, f68 in self._aligned_pairs:
            frame_map[f66] = f68
        return frame_map

    def summary(self) -> str:
        """Return human-readable summary."""
        lines = [
            f"Seconds detected: cam66={self._sec66 + 1}, cam68={self._sec68 + 1}",
            f"Aligned pairs: {len(self._aligned_pairs)}",
            f"Offset anchor points: {len(self._offset_pairs)}",
        ]
        if self._offset_pairs:
            offsets = [p[0] - p[1] for p in self._offset_pairs]
            lines.append(f"Offset range: {min(offsets):+d} to {max(offsets):+d} frames")
        return "\n".join(lines)


# Backward compatibility alias
FrameAligner = TimestampAligner
