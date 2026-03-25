"""Frame synchronization middleware for dual-camera tennis tracking.

Aligns two unsynchronized camera streams before feeding into the pipeline.
This is a MIDDLEWARE — it sits between video input and all downstream processing.

The cameras have independent internal clocks that drift relative to each other.
Frame N in cam66 may NOT correspond to the same real-world moment as frame N
in cam68. This middleware finds and corrects the offset.

Method: Cross-correlate ball detection density signals from a calibration run,
then apply the fixed offset when reading frames.

Two modes:
  - Offline (video files): auto-detect offset then yield aligned pairs
  - Realtime (RTSP): apply known offset, fast camera buffers while waiting

Usage (offline):
    sync = FrameSync(video66_path, video68_path, max_frames=3000)
    for frame_idx, frame66, frame68 in sync:
        # frame66 and frame68 are from the same real-world moment
        run_tracknet(frame66, frame68)

Usage (realtime):
    sync = FrameSyncLive(offset=-65)
    sync.push("cam66", frame66)
    sync.push("cam68", frame68)
    for frame66, frame68 in sync.pop_aligned():
        process(frame66, frame68)
"""

import cv2
import numpy as np
import logging
from typing import Iterator, Optional

logger = logging.getLogger(__name__)


def _detect_second_tick(curr_gray: np.ndarray, prev_gray: Optional[np.ndarray],
                        threshold: float = 15.0) -> bool:
    """Detect if the OSD seconds digit changed."""
    if prev_gray is None:
        return True
    return cv2.absdiff(curr_gray, prev_gray).astype(np.float32).mean() > threshold


def _crop_osd_seconds(frame: np.ndarray, region: tuple = (3, 33, 560, 610)) -> np.ndarray:
    """Crop the OSD seconds region and convert to grayscale."""
    y1, y2, x1, x2 = region
    crop = frame[y1:y2, x1:x2]
    if len(crop.shape) == 3:
        crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    return crop


def _equalize_lists(a: list, b: list) -> tuple[list, list]:
    """Make two lists same length by evenly sampling from the longer one."""
    if len(a) == len(b):
        return a, b
    if len(a) > len(b):
        target = len(b)
        indices = np.round(np.linspace(0, len(a) - 1, target)).astype(int)
        return [a[i] for i in indices], b
    else:
        target = len(a)
        indices = np.round(np.linspace(0, len(b) - 1, target)).astype(int)
        return a, [b[i] for i in indices]


def find_frame_offset(
    video66_path: str,
    video68_path: str,
    max_frames: int = 1500,
    max_lag: int = 200,
) -> int:
    """Find frame offset between two cameras using motion cross-correlation.

    Uses frame differencing (no ML model needed) to create a motion signal
    for each camera, then cross-correlates to find the best alignment.

    Returns:
        offset: cam66 frame N corresponds to cam68 frame (N + offset).
                Negative offset means cam68 is behind cam66.
    """
    def extract_motion(path, n):
        cap = cv2.VideoCapture(path)
        ret, prev = cap.read()
        prev_g = cv2.cvtColor(cv2.resize(prev, (320, 180)), cv2.COLOR_BGR2GRAY)
        motion = []
        for _ in range(1, n):
            ret, frame = cap.read()
            if not ret:
                break
            g = cv2.cvtColor(cv2.resize(frame, (320, 180)), cv2.COLOR_BGR2GRAY)
            motion.append(cv2.absdiff(g, prev_g).astype(float).mean())
            prev_g = g
        cap.release()
        return np.array(motion)

    logger.info("Finding frame offset (motion cross-correlation, %d frames)...", max_frames)
    m66 = extract_motion(video66_path, max_frames)
    m68 = extract_motion(video68_path, max_frames)

    # Smooth to create density signal
    from scipy.ndimage import uniform_filter1d
    d66 = uniform_filter1d(m66, 20)
    d68 = uniform_filter1d(m68, 20)

    best_corr = -1
    best_lag = 0
    for lag in range(-max_lag, max_lag + 1):
        if lag >= 0:
            a, b = d66[lag:], d68[:len(d66) - lag]
        else:
            a, b = d66[:len(d66) + lag], d68[-lag:]
        n = min(len(a), len(b))
        if n < 100:
            continue
        a, b = a[:n], b[:n]
        sa = (a - a.mean()) / (a.std() + 1e-8)
        sb = (b - b.mean()) / (b.std() + 1e-8)
        corr = np.mean(sa * sb)
        if corr > best_corr:
            best_corr = corr
            best_lag = lag

    logger.info("Frame offset: %+d frames (%+.1fs), corr=%.4f", best_lag, best_lag / 25.0, best_corr)
    return best_lag


class FrameSync:
    """Offline frame synchronization for two video files.

    Auto-detects or uses a known frame offset, then yields aligned frame pairs.
    cam66 is the reference — cam68 is shifted by offset frames.

    Usage:
        sync = FrameSync("cam66.mp4", "cam68.mp4", max_frames=3000)
        for idx, f66, f68 in sync:
            # f66 and f68 are from the same real-world moment
            pass
    """

    def __init__(
        self,
        video66_path: str,
        video68_path: str,
        max_frames: int = 0,
        offset: Optional[int] = None,
        auto_detect: bool = True,
    ):
        self.video66_path = video66_path
        self.video68_path = video68_path
        self.max_frames = max_frames
        self.total_output_frames = 0

        if offset is not None:
            self.offset = offset
            logger.info("FrameSync: using provided offset=%+d", offset)
        elif auto_detect:
            detect_frames = min(max_frames, 1500) if max_frames > 0 else 1500
            self.offset = find_frame_offset(video66_path, video68_path, detect_frames)
        else:
            self.offset = 0
            logger.info("FrameSync: no offset (disabled)")

    def __iter__(self) -> Iterator[tuple[int, np.ndarray, np.ndarray]]:
        """Yield (output_idx, frame66, frame68) aligned pairs."""
        cap66 = cv2.VideoCapture(self.video66_path)
        cap68 = cv2.VideoCapture(self.video68_path)

        total66 = int(cap66.get(cv2.CAP_PROP_FRAME_COUNT))
        total68 = int(cap68.get(cv2.CAP_PROP_FRAME_COUNT))

        # Determine start positions based on offset
        if self.offset >= 0:
            # cam68 is behind: skip first `offset` frames of cam66
            start66 = self.offset
            start68 = 0
        else:
            # cam66 is behind: skip first `-offset` frames of cam68
            start66 = 0
            start68 = -self.offset

        n_pairs = min(total66 - start66, total68 - start68)
        if self.max_frames > 0:
            n_pairs = min(n_pairs, self.max_frames)

        logger.info(
            "FrameSync: offset=%+d, cam66 starts at %d, cam68 starts at %d, %d pairs",
            self.offset, start66, start68, n_pairs,
        )

        cap66.set(cv2.CAP_PROP_POS_FRAMES, start66)
        cap68.set(cv2.CAP_PROP_POS_FRAMES, start68)

        for i in range(n_pairs):
            ret66, frame66 = cap66.read()
            ret68, frame68 = cap68.read()
            if not ret66 or not ret68:
                break
            yield i, frame66, frame68
            self.total_output_frames += 1

        cap66.release()
        cap68.release()

        logger.info("FrameSync complete: %d aligned pairs output", self.total_output_frames)

    def summary(self) -> str:
        return f"FrameSync: offset={self.offset:+d}, {self.total_output_frames} pairs"


class FrameSyncLive:
    """Real-time frame synchronization for RTSP streams.

    Uses a fixed frame offset. The faster camera's frames are buffered
    until the slower camera catches up.

    Usage:
        sync = FrameSyncLive(offset=-65)  # cam68 is 65 frames behind
        # In camera read loops:
        sync.push_66(frame66)
        sync.push_68(frame68)
        for f66, f68 in sync.pop_aligned():
            process(f66, f68)
    """

    def __init__(self, offset: int = 0, max_buffer: int = 200):
        """
        Args:
            offset: cam66 frame N pairs with cam68 frame (N + offset).
                    Negative = cam68 is behind, needs to catch up.
            max_buffer: max frames to buffer before dropping old ones.
        """
        self.offset = offset
        self.max_buffer = max_buffer

        self._buf66: dict[int, np.ndarray] = {}  # frame_idx -> frame
        self._buf68: dict[int, np.ndarray] = {}
        self._count66 = 0
        self._count68 = 0
        self._next_output = 0

    def push_66(self, frame: np.ndarray):
        """Push a cam66 frame (auto-increments frame counter)."""
        self._buf66[self._count66] = frame
        self._count66 += 1
        self._cleanup()

    def push_68(self, frame: np.ndarray):
        """Push a cam68 frame (auto-increments frame counter)."""
        self._buf68[self._count68] = frame
        self._count68 += 1
        self._cleanup()

    def pop_aligned(self) -> list[tuple[np.ndarray, np.ndarray]]:
        """Pop all available aligned frame pairs.

        Returns list of (frame66, frame68) where both correspond
        to the same real-world moment.
        """
        pairs = []
        while True:
            # cam66 frame idx that we want to output
            idx66 = self._next_output
            # corresponding cam68 frame idx
            idx68 = self._next_output + self.offset

            if idx66 in self._buf66 and idx68 in self._buf68:
                f66 = self._buf66.pop(idx66)
                f68 = self._buf68.pop(idx68)
                pairs.append((f66, f68))
                self._next_output += 1
            else:
                break

        return pairs

    def _cleanup(self):
        """Remove old buffered frames to prevent memory growth."""
        cutoff = self._next_output - 10  # keep some history
        for idx in list(self._buf66.keys()):
            if idx < cutoff:
                del self._buf66[idx]
        for idx in list(self._buf68.keys()):
            if idx < cutoff:
                del self._buf68[idx]
