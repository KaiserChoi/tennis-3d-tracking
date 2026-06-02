"""Pure detection pairing helpers for cross-camera tracking."""

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class DetectionPairingResult:
    pairs: list[tuple[dict[str, Any], dict[str, Any]]]
    remaining_first: list[dict[str, Any]]
    remaining_second: list[dict[str, Any]]


def _capture_time(detection: dict[str, Any]) -> float:
    return detection.get("capture_ts", detection["timestamp"])


def pair_by_capture_time(
    first: list[dict[str, Any]],
    second: list[dict[str, Any]],
    *,
    match_window: float,
    max_queue_size: int = 32,
    keep_tail_size: int = 16,
) -> DetectionPairingResult:
    """Greedily pair two camera queues by nearest capture time.

    The behavior intentionally matches the historic Orchestrator loop:
    pairs are chosen in first-queue order, a second-queue item can be used
    once, equality with the match window is outside the window, and oversized
    remainder queues keep only their freshest tail.
    """
    pairs: list[tuple[dict[str, Any], dict[str, Any]]] = []
    used_first: set[int] = set()
    used_second: set[int] = set()

    for i, d1 in enumerate(first):
        t1 = _capture_time(d1)
        best_j, best_dt = -1, match_window
        for j, d2 in enumerate(second):
            if j in used_second:
                continue
            dt = abs(t1 - _capture_time(d2))
            if dt < best_dt:
                best_dt = dt
                best_j = j
        if best_j >= 0:
            pairs.append((d1, second[best_j]))
            used_first.add(i)
            used_second.add(best_j)

    remaining_first = [d for i, d in enumerate(first) if i not in used_first]
    remaining_second = [d for j, d in enumerate(second) if j not in used_second]

    if len(remaining_first) > max_queue_size:
        remaining_first = remaining_first[-keep_tail_size:]
    if len(remaining_second) > max_queue_size:
        remaining_second = remaining_second[-keep_tail_size:]

    return DetectionPairingResult(
        pairs=pairs,
        remaining_first=remaining_first,
        remaining_second=remaining_second,
    )
