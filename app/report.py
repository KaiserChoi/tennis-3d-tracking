"""Match report generation from JSONL tracking data.

Produces structured analysis data (viz_data.json) and self-contained HTML
dashboard reports.  Core analysis functions ported from the research pipeline
(process.py) with improvements for the live tracking system.

Usage:
    from app.report import generate_report
    result = generate_report("recordings/tracking_20260404_093901.jsonl")
"""

from __future__ import annotations

import json
import math
import logging
from bisect import bisect_left, bisect_right
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SINGLES_X = 4.115
BASELINE_Y = 11.885
NET_ZONE = 1.5
SPEED_CALIBRATION = 0.8
SPEED_MIN_RAW = 25
SPEED_MAX_RAW = 180
MEDIAN_WINDOW = 5
BOUNCE_Z_THRESH = 0.25
BOUNCE_Z_MIN = 0.02           # filter z=0 bad triangulation
BOUNCE_DEDUP_FRAMES = 10
MAX_BOUNCES_PER_RALLY = 8
TOP_TRAJECTORIES = 8

# Dashboard reports should reflect what the realtime system actually accepted.
# The older report path re-detected bounces from z minima only inside detected
# rally windows; in live 10-minute sessions that badly undercounted bounces.
LIVE_BOUNCE_RALLY_GAP_SEC = 8.0
LIVE_BOUNCE_CONTEXT_SEC = 1.0
# A report rally is stricter than a raw live rally: short 1-3 bounce clusters
# are often serve faults, pickup/prep bounces, or detector noise in real sessions.
LIVE_BOUNCE_MIN_RALLY_BOUNCES = 4
LIVE_BOUNCE_DEDUP_SEC = 0.7
LIVE_BOUNCE_DEDUP_DIST_M = 1.5
LIVE_BOUNCE_EXTENDED_X = 6.5
LIVE_BOUNCE_EXTENDED_Y = 14.5

REPORTS_DIR = Path("reports")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_tracking(jsonl_path: Path) -> tuple[list[dict], float]:
    """Load JSONL tracking data. Returns (raw_data, t0)."""
    raw = []
    with open(jsonl_path, encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                # Report generation can run while the consumer is appending to
                # the same JSONL. The API fsyncs first, but tolerate a partial
                # tail line so one bad write boundary does not fail the report.
                logger.warning("Skipping malformed JSONL line %d in %s", line_no, jsonl_path)
                continue
            # Skip non-ball rows (e.g. player_pose) — only ball tracking frames have 'ts'
            if d.get("type") == "player_pose":
                continue
            raw.append(d)
    if not raw:
        raise ValueError(f"No ball tracking data in {jsonl_path}")
    return raw, raw[0]["ts"]


def filter_frames(raw_data: list[dict]) -> list[dict]:
    """Filter frames by confidence and camera agreement.

    Ported from process.py — removes noise, high-confidence outliers,
    and frames where cameras disagree significantly.
    """
    filtered = []
    for d in raw_data:
        cam66 = d.get("cam66")
        cam68 = d.get("cam68")
        if not cam66 or not cam68:
            continue

        c66 = cam66.get("conf", 0)
        c68 = cam68.get("conf", 0)

        # Too high confidence = artifact, too low = noise
        if c66 > 150 or c68 > 150:
            continue
        if c66 < 15 and c68 < 15:
            continue

        # Camera world-coord agreement check
        wx66, wy66 = cam66.get("wx", 0), cam66.get("wy", 0)
        wx68, wy68 = cam68.get("wx", 0), cam68.get("wy", 0)
        if math.sqrt((wx66 - wx68) ** 2 + (wy66 - wy68) ** 2) > 8:
            continue

        pos = d.get("smoothed") or d.get("3d")
        if not pos:
            continue

        filtered.append({
            "frame": d["frame"],
            "ts": d["ts"],
            "x": pos["x"],
            "y": pos["y"],
            "z": pos["z"],
            "conf66": c66,
            "conf68": c68,
        })
    return filtered


# ---------------------------------------------------------------------------
# Analysis functions
# ---------------------------------------------------------------------------

def frames_in_rally(filtered: list[dict], t0: float,
                    start: float, end: float) -> list[dict]:
    """Extract frames within a rally time window using binary search."""
    ts_list = [f["ts"] - t0 for f in filtered]
    lo = bisect_left(ts_list, start)
    hi = bisect_right(ts_list, end)
    return filtered[lo:hi]


def compute_speeds(rally_frames: list[dict]) -> list[float]:
    """Compute calibrated speeds (km/h) with sliding median filter."""
    if len(rally_frames) < 2:
        return []

    frame_speeds: list[float | None] = []
    for i in range(1, len(rally_frames)):
        dt = rally_frames[i]["ts"] - rally_frames[i - 1]["ts"]
        if dt <= 0 or dt > 0.5:
            frame_speeds.append(None)
            continue
        dx = rally_frames[i]["x"] - rally_frames[i - 1]["x"]
        dy = rally_frames[i]["y"] - rally_frames[i - 1]["y"]
        dz = rally_frames[i]["z"] - rally_frames[i - 1]["z"]
        frame_speeds.append(math.sqrt(dx ** 2 + dy ** 2 + dz ** 2) / dt * 3.6)

    # Sliding median filter
    speeds = []
    half_w = MEDIAN_WINDOW // 2
    for i in range(len(frame_speeds)):
        w = [frame_speeds[j]
             for j in range(max(0, i - half_w), min(len(frame_speeds), i + half_w + 1))
             if frame_speeds[j] is not None]
        if not w:
            continue
        w.sort()
        s_raw = w[len(w) // 2]
        if s_raw < SPEED_MIN_RAW or s_raw > SPEED_MAX_RAW:
            continue
        speeds.append(round(s_raw * SPEED_CALIBRATION, 1))

    return speeds


def _filter_rolling(bounces: list[dict]) -> list[dict]:
    """Remove rolling-ball sequences: keep only first landing point."""
    if not bounces:
        return bounces
    ROLL_Z = 0.05
    ROLL_GAP = 30
    result = []
    i = 0
    while i < len(bounces):
        result.append(bounces[i])
        if bounces[i]["z"] < ROLL_Z:
            j = i + 1
            while (j < len(bounces)
                   and bounces[j]["z"] < ROLL_Z
                   and bounces[j]["frame"] - bounces[j - 1]["frame"] <= ROLL_GAP):
                j += 1
            i = j
        else:
            i += 1
    return result


def detect_bounces_report(rally_frames: list[dict], shots_count: int) -> list[dict]:
    """Detect bounces (z local minima) within rally frames, with rolling filter."""
    raw_bounces = []
    for i in range(2, len(rally_frames) - 2):
        z = rally_frames[i]["z"]
        if (z > BOUNCE_Z_MIN
                and z < rally_frames[i - 1]["z"]
                and z < rally_frames[i + 1]["z"]
                and z < BOUNCE_Z_THRESH):
            x, y = rally_frames[i]["x"], rally_frames[i]["y"]
            if abs(x) < 7 and abs(y) < 14:
                raw_bounces.append({
                    "x": x, "y": y, "z": z,
                    "frame": rally_frames[i]["frame"],
                })

    # Dedup within 10 frames
    raw_bounces.sort(key=lambda b: b["frame"])
    deduped = []
    i = 0
    while i < len(raw_bounces):
        group = [raw_bounces[i]]
        j = i + 1
        while j < len(raw_bounces) and raw_bounces[j]["frame"] - group[0]["frame"] <= BOUNCE_DEDUP_FRAMES:
            group.append(raw_bounces[j])
            j += 1
        deduped.append(min(group, key=lambda b: b["z"]))
        i = j

    deduped = _filter_rolling(deduped)

    # Cap per rally
    cap = min(shots_count + 2, MAX_BOUNCES_PER_RALLY)
    if len(deduped) > cap:
        deduped.sort(key=lambda b: b["z"])
        deduped = deduped[:cap]

    return deduped


def judge_in_out(bounces: list[dict]) -> list[dict]:
    """Classify bounces as in/out with reason and court side.

    Side convention (V2 coords):
      y < 0 → "near"  (cam66 side)
      y > 0 → "far"   (cam68 side)
    """
    result = []
    for b in bounces:
        x, y = b["x"], b["y"]
        if abs(y) < NET_ZONE:
            out, reason = True, "net"
        elif abs(x) > SINGLES_X:
            out, reason = True, "wide"
        elif abs(y) > BASELINE_Y:
            out, reason = True, "long"
        else:
            out, reason = False, "in"

        side = "near" if y < 0 else "far"

        result.append({
            "x": round(x, 2), "y": round(y, 2), "z": round(b.get("z", 0), 3),
            "out": out, "reason": reason,
            "side": side,
        })
    return result


def _bounce_sort_key(b: dict) -> float:
    if b.get("ts") is not None:
        return float(b["ts"])
    return float(b.get("frame", 0)) / 25.0


def _bounce_distance(a: dict, b: dict) -> float:
    return math.hypot(float(a["x"]) - float(b["x"]), float(a["y"]) - float(b["y"]))


def _choose_best_bounce(group: list[dict]) -> dict:
    return min(group, key=lambda b: (float(b.get("z", 99.0)), _bounce_sort_key(b)))


def _first_present(*values):
    for v in values:
        if v is not None:
            return v
    return None


def _dedup_report_bounces(bounces: list[dict]) -> list[dict]:
    """Deduplicate near-identical bounce reports without suppressing rallies.

    Realtime already has dedup, but dashboard reports may mix live bounce rows
    and z-min fallback rows. Keep this conservative: only merge events close in
    both time/frame and court position.
    """
    if not bounces:
        return []

    ordered = sorted(bounces, key=_bounce_sort_key)
    groups: list[list[dict]] = [[ordered[0]]]
    for b in ordered[1:]:
        head = groups[-1][0]
        dt = abs(_bounce_sort_key(b) - _bounce_sort_key(head))
        frame_dt = abs(int(b.get("frame", 0)) - int(head.get("frame", 0)))
        close_time = dt <= LIVE_BOUNCE_DEDUP_SEC or frame_dt <= BOUNCE_DEDUP_FRAMES
        close_place = _bounce_distance(b, head) <= LIVE_BOUNCE_DEDUP_DIST_M
        if close_time and close_place:
            groups[-1].append(b)
        else:
            groups.append([b])

    return [_choose_best_bounce(g) for g in groups]


def extract_live_bounces(raw_data: list[dict], t0: float) -> list[dict]:
    """Extract accepted realtime bounces stored in tracking JSONL rows."""
    bounces: list[dict] = []
    for idx, row in enumerate(raw_data):
        b = row.get("bounce")
        if not isinstance(b, dict):
            continue

        try:
            x = float(b["x"])
            y = float(b["y"])
            z = float(b.get("z", 0.0))
        except Exception:
            continue
        if not all(math.isfinite(v) for v in (x, y, z)):
            continue

        # Keep plausible out balls, but drop wild triangulation artifacts.
        if abs(x) > LIVE_BOUNCE_EXTENDED_X or abs(y) > LIVE_BOUNCE_EXTENDED_Y:
            continue

        ts_val = _first_present(
            b.get("timestamp"),
            b.get("capture_ts"),
            row.get("capture_ts"),
            row.get("ts"),
        )
        try:
            ts = float(ts_val)
        except Exception:
            continue
        if not math.isfinite(ts):
            continue

        frame_val = b.get("frame_index")
        if frame_val is None:
            frame_val = row.get("frame", idx)
        try:
            frame = int(frame_val)
        except Exception:
            frame = idx

        bounces.append({
            "x": x,
            "y": y,
            "z": z,
            "frame": frame,
            "ts": ts,
            "t_rel": ts - t0,
            "in_court": b.get("in_court"),
            "source": b.get("source_camera", "live"),
        })

    return _dedup_report_bounces(bounces)


def detect_rallies_from_live_bounces(
    live_bounces: list[dict],
    duration: float,
    gap_sec: float = LIVE_BOUNCE_RALLY_GAP_SEC,
    min_bounces: int = LIVE_BOUNCE_MIN_RALLY_BOUNCES,
) -> list[dict]:
    """Segment rallies from accepted live bounce cadence.

    For amateur play, gaps above ~8s usually mean a ball pickup, serve reset, or
    next feed. Requiring at least two bounces avoids counting isolated noise or
    one-off faults as full rallies.
    """
    if not live_bounces:
        return []

    ordered = sorted(live_bounces, key=lambda b: b["t_rel"])
    clusters: list[list[dict]] = []
    cur = [ordered[0]]
    for b in ordered[1:]:
        if b["t_rel"] - cur[-1]["t_rel"] > gap_sec:
            clusters.append(cur)
            cur = [b]
        else:
            cur.append(b)
    clusters.append(cur)

    rallies: list[dict] = []
    for cluster in clusters:
        if len(cluster) < min_bounces:
            continue
        start = max(0.0, cluster[0]["t_rel"] - LIVE_BOUNCE_CONTEXT_SEC)
        end = min(duration, cluster[-1]["t_rel"] + LIVE_BOUNCE_CONTEXT_SEC)
        if end <= start:
            continue
        rallies.append({
            "start": round(start, 1),
            "end": round(end, 1),
            "duration": round(end - start, 1),
            "peak_score": len(cluster),
            "index": len(rallies) + 1,
            "source": "live_bounce",
            "bounce_count": len(cluster),
        })

    return rallies


def count_shots(rally_frames: list[dict]) -> int:
    """Count shots by y-direction reversals."""
    shots = 0
    prev_dy_sign = None
    for i in range(3, len(rally_frames)):
        dy = rally_frames[i]["y"] - rally_frames[i - 3]["y"]
        if abs(dy) < 0.3:
            continue
        cur_sign = 1 if dy > 0 else -1
        if prev_dy_sign is not None and cur_sign != prev_dy_sign:
            shots += 1
        prev_dy_sign = cur_sign
    return shots


# ---------------------------------------------------------------------------
# Rally detection (simplified from Rally_detector.py)
# ---------------------------------------------------------------------------

def detect_rallies_from_tracking(filtered: list[dict], t0: float,
                                  window_sec: float = 5.0,
                                  step_sec: float = 0.5,
                                  score_threshold: int = 12,
                                  merge_gap_sec: float = 3.0,
                                  min_rally_sec: float = 2.0) -> list[dict]:
    """Detect rally segments using multi-feature composite scoring."""
    if not filtered:
        return []

    ts = np.array([f["ts"] - t0 for f in filtered])
    y = np.array([f["y"] for f in filtered])
    z = np.array([f["z"] for f in filtered])
    duration = float(ts[-1])

    # Sliding window scoring
    high_windows = []
    t = 0.0
    while t + window_sec <= duration + step_sec:
        mask = (ts >= t) & (ts < t + window_sec)
        n = int(np.sum(mask))
        if n < 5:
            t += step_sec
            continue

        y_w, z_w, t_w = y[mask], z[mask], ts[mask]
        score = 0

        # Court traversal
        y_min, y_max = float(np.min(y_w)), float(np.max(y_w))
        if y_min < -2 and y_max > 2:
            score += 2
        if (y_max - y_min) > 15:
            score += 3

        # Z-arc count
        n_arcs = 0
        was_above = False
        for zv in z_w:
            if zv > 1.0:
                was_above = True
            elif zv < 0.5 and was_above:
                n_arcs += 1
                was_above = False
        score += min(n_arcs * 2, 10)

        # Y-direction reversals
        if len(y_w) > 10:
            kernel = np.ones(5) / 5
            y_smooth = np.convolve(y_w, kernel, mode="valid")
            if len(y_smooth) > 1:
                dy = np.diff(y_smooth)
                last_dir = 0
                y_rev = 0
                for dv in dy:
                    if abs(dv) > 0.5:
                        curr_dir = int(np.sign(dv))
                        if last_dir != 0 and curr_dir != last_dir:
                            y_rev += 1
                        last_dir = curr_dir
                score += min(y_rev, 8)

        # Tracking continuity
        dt = np.diff(t_w)
        gap_frac = float(np.sum(dt > 0.3) / max(len(dt), 1))
        if gap_frac < 0.1:
            score += 2

        if score >= score_threshold:
            high_windows.append({"t": t, "score": score})

        t += step_sec

    if not high_windows:
        return []

    # Merge consecutive high-score windows
    regions = [[high_windows[0]]]
    for ws in high_windows[1:]:
        if ws["t"] - regions[-1][-1]["t"] <= merge_gap_sec:
            regions[-1].append(ws)
        else:
            regions.append([ws])

    rallies = []
    for region in regions:
        start = region[0]["t"]
        end = region[-1]["t"] + window_sec
        dur = end - start
        if dur >= min_rally_sec:
            scores = [ws["score"] for ws in region]
            rallies.append({
                "start": round(start, 1),
                "end": round(end, 1),
                "duration": round(dur, 1),
                "peak_score": max(scores),
                "index": len(rallies) + 1,
            })

    return rallies


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def generate_report(jsonl_path: str | Path,
                    rallies: list[dict] | None = None,
                    report_name: str | None = None) -> dict:
    """Generate a match analysis report from JSONL tracking data.

    Args:
        jsonl_path: Path to tracking JSONL file.
        rallies: Pre-defined rally segments [{start, end}].
                 If None, rallies are auto-detected.
        report_name: Name for the report directory. Defaults to JSONL filename.

    Returns:
        dict with report data + path to generated files.
    """
    jsonl_path = Path(jsonl_path)
    if not jsonl_path.exists():
        raise FileNotFoundError(f"JSONL not found: {jsonl_path}")

    if report_name is None:
        report_name = jsonl_path.stem.replace("tracking_", "")

    logger.info("Generating report from %s", jsonl_path.name)

    # Load and filter
    raw_data, t0 = load_tracking(jsonl_path)
    filtered = filter_frames(raw_data)
    live_bounces = extract_live_bounces(raw_data, t0)

    if not filtered:
        logger.warning("No valid frames after filtering")
        return {"error": "No valid frames", "raw_count": len(raw_data)}

    duration = raw_data[-1]["ts"] - t0 if raw_data else 0.0

    # Detect or use provided rallies
    rally_source = "provided" if rallies is not None else "offline"
    if rallies is None:
        rallies = detect_rallies_from_live_bounces(live_bounces, duration)
        if rallies:
            rally_source = "live_bounce"
        else:
            rallies = detect_rallies_from_tracking(filtered, t0)

    use_live_bounces = bool(live_bounces)

    logger.info(
        "Frames: %d raw, %d filtered, %d live_bounces, %d rallies (%s), %.0fs duration",
        len(raw_data), len(filtered), len(live_bounces), len(rallies), rally_source, duration,
    )

    # Per-rally analysis
    all_speeds: list[float] = []
    rally_speeds: list[dict] = []
    rally_durations: list[float] = []
    shots_list: list[int] = []
    all_bounces: list[dict] = []
    trajectories: list[dict] = []

    for rally in rallies:
        rf = frames_in_rally(filtered, t0, rally["start"], rally["end"])
        rally_durations.append(rally["end"] - rally["start"])

        speeds = compute_speeds(rf)
        all_speeds.extend(speeds)
        rally_speeds.append({
            "avg": round(sum(speeds) / len(speeds), 1) if speeds else 0,
            "max": round(max(speeds), 1) if speeds else 0,
        })

        rally_live_bounces = [
            b for b in live_bounces
            if rally["start"] <= b["t_rel"] <= rally["end"]
        ] if use_live_bounces else []

        shots = count_shots(rf)
        if rally_live_bounces:
            shots = max(shots, max(1, len(rally_live_bounces) - 1))
        shots_list.append(shots)

        bounces = rally_live_bounces if rally_live_bounces else detect_bounces_report(rf, shots)
        all_bounces.extend(bounces)

    # Cross-rally bounce dedup
    deduped_bounces = _dedup_report_bounces(all_bounces)

    bounces_judged = judge_in_out(deduped_bounces)
    in_c = sum(1 for b in bounces_judged if not b["out"])
    out_c = sum(1 for b in bounces_judged if b["out"])

    # Per-side stats
    near_bounces = [b for b in bounces_judged if b["side"] == "near"]
    far_bounces = [b for b in bounces_judged if b["side"] == "far"]

    # Top trajectories
    for idx, rally in enumerate(rallies):
        rf = frames_in_rally(filtered, t0, rally["start"], rally["end"])
        if len(rf) < 5:
            continue
        step = max(1, len(rf) // 100)
        pts = [{"x": round(rf[i]["x"], 2), "y": round(rf[i]["y"], 2), "z": round(rf[i]["z"], 2)}
               for i in range(0, len(rf), step)]
        trajectories.append({"rally_idx": idx, "points": pts, "duration": rally["end"] - rally["start"]})
    trajectories.sort(key=lambda t: len(t["points"]), reverse=True)
    trajectories = trajectories[:TOP_TRAJECTORIES]

    # Speed stats
    avg_speed = round(sum(all_speeds) / len(all_speeds), 1) if all_speeds else 0
    max_speed = round(max(all_speeds), 1) if all_speeds else 0

    # Speed tiers
    tiers = [0] * 5
    for s in all_speeds:
        if s < 40: tiers[0] += 1
        elif s < 65: tiers[1] += 1
        elif s < 90: tiers[2] += 1
        elif s < 115: tiers[3] += 1
        else: tiers[4] += 1

    # Speed distribution
    bins = list(range(20, 150, 10))
    hist = [0] * len(bins)
    for s in all_speeds:
        idx = min(int((s - 20) / 10), len(bins) - 1)
        if idx >= 0:
            hist[idx] += 1

    # Build output
    output = {
        "session_name": report_name,
        "summary": {
            "total_frames": len(raw_data),
            "filtered_frames": len(filtered),
            "rally_count": len(rallies),
            "total_duration_s": round(duration, 1),
            "total_shots": sum(shots_list),
            "total_bounces": len(bounces_judged),
            "bounces_in": in_c,
            "bounces_out": out_c,
            "avg_speed_kmh": avg_speed,
            "max_speed_kmh": max_speed,
            "rally_source": rally_source,
            "bounce_source": "live" if use_live_bounces else "offline_z_min",
            "live_bounces_available": len(live_bounces),
            "near_side": {
                "label": "Near Side (cam66)",
                "total": len(near_bounces),
                "in": sum(1 for b in near_bounces if not b["out"]),
                "out": sum(1 for b in near_bounces if b["out"]),
            },
            "far_side": {
                "label": "Far Side (cam68)",
                "total": len(far_bounces),
                "in": sum(1 for b in far_bounces if not b["out"]),
                "out": sum(1 for b in far_bounces if b["out"]),
            },
        },
        "rallies": [
            {**rally, "shots": shots_list[i], "speed_avg": rally_speeds[i]["avg"],
             "speed_max": rally_speeds[i]["max"], "duration": rally_durations[i]}
            for i, rally in enumerate(rallies)
        ],
        "speed_tiers": {
            "names": ["Slow", "Medium", "Fast", "Power", "Max"],
            "ranges": ["<40", "40-65", "65-90", "90-115", "115+"],
            "counts": tiers,
        },
        "speed_distribution": {
            "bins": [f"{b}-{b + 10}" for b in bins],
            "counts": hist,
        },
        "bounces": bounces_judged,
        "trajectories": trajectories,
        # rally_stats for Chart.js template compatibility
        "rally_stats": {
            "durations": [round(d, 1) for d in rally_durations],
            "shots_per_rally": shots_list,
            "avg_speeds": [rs["avg"] for rs in rally_speeds],
            "max_speeds": [rs["max"] for rs in rally_speeds],
        },
    }

    # Save to reports directory
    report_dir = REPORTS_DIR / report_name
    report_dir.mkdir(parents=True, exist_ok=True)

    data_path = report_dir / "viz_data.json"
    with open(data_path, "w") as f:
        json.dump(output, f, indent=2)

    # Generate HTML dashboard from template
    _generate_dashboard_html(report_dir, output)

    logger.info("Report saved: %s (%d rallies, %d bounces, avg %.1f km/h)",
                report_dir, len(rallies), len(bounces_judged), avg_speed)

    return {
        "status": "ok",
        "report_name": report_name,
        "path": str(report_dir),
        "data_path": str(data_path),
        "dashboard_url": f"/reports/{report_name}/dashboard.html",
        "summary": output["summary"],
    }


def _generate_dashboard_html(report_dir: Path, viz_data: dict):
    """Generate self-contained HTML dashboard by injecting data into template."""
    template_path = Path(__file__).parent / "api" / "templates" / "report_template.html"
    if not template_path.exists():
        logger.warning("Report template not found: %s", template_path)
        return

    template = template_path.read_text(encoding="utf-8")
    lines = template.split("\n")

    data_json = json.dumps(viz_data)
    s = viz_data["summary"]
    session_name = viz_data["session_name"]

    # Inject DATA
    for i, line in enumerate(lines):
        if line.strip().startswith("const DATA"):
            lines[i] = f"const DATA = {data_json};"
            break

    # Title
    for i, line in enumerate(lines):
        if "<title>" in line:
            lines[i] = f"<title>Tennis Analysis - {session_name}</title>"
            break

    # Header subtitle
    for i, line in enumerate(lines):
        if "Multi-Camera Tracking" in line:
            near = s.get("near_side", {})
            far = s.get("far_side", {})
            side_str = ""
            if near:
                side_str = f' &middot; Near {near.get("total",0)} / Far {far.get("total",0)}'
            lines[i] = (f'  <p>{session_name} &middot; '
                        f'{s["rally_count"]} rallies &middot; '
                        f'{s["total_bounces"]} bounces &middot; '
                        f'avg {s["avg_speed_kmh"]} km/h{side_str}</p>')
            break

    # Footer
    for i, line in enumerate(lines):
        if "Noise filter" in line:
            bd = f'{s["bounces_in"]} IN / {s["bounces_out"]} OUT'
            lines[i] = (f'  {s["total_frames"]} raw &rarr; {s["filtered_frames"]} valid '
                        f'&middot; {s["rally_count"]} rallies '
                        f'&middot; {s["total_bounces"]} bounces ({bd})')
            break

    out_path = report_dir / "dashboard.html"
    out_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Dashboard HTML: %s", out_path)
