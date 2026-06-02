# Single-Cam Bounce Detector

## Goal

Add an experimental single-camera bounce mode for the realtime dashboard,
starting with `cam68`. The current production default remains stereo 3D
bounce detection. Single-camera mode is for A/B testing and field tuning
before it becomes a fallback or primary source.

The guiding idea is to avoid finding local extrema across the entire noisy
2D trajectory. Instead, first cut the stream into a meaningful flight window
using the net line in image coordinates, then find the bounce inside that
window.

## Current Production Context

- Default realtime production mode is stereo: `cam66 + cam68` detections are
  paired, triangulated into 3D, smoothed, then passed to `HybridBounceDetector`.
- Stereo bounce existence is decided from 3D trajectory shape, especially the
  `z` dip / V-shape / parabolic evidence.
- Landing coordinates may use single-camera homography, but the event decision
  is still based on stereo 3D.
- The dashboard now has a bounce mode switch, with `stereo`, `mono_cam68`, and
  `mono_cam66`. The first serious single-camera implementation should target
  `mono_cam68`.

## Proposed Cam68 Net-Line Window Method

For `cam68`, use the net pixel line as the segmentation boundary:

```text
net_line_y = 260
```

Use hysteresis around the net line to avoid repeated open/close events when the
detected ball jitters near the line:

```text
upper_threshold = net_line_y - hysteresis_px
lower_threshold = net_line_y + hysteresis_px
```

Suggested starting value:

```text
hysteresis_px = 8
```

### Upper-Half Recorder

This recorder handles a trajectory window that starts in the upper image half.

```text
Start recording when y < net_line_y - hysteresis_px.
Keep appending detections while the ball is in this flight window.
When y is continuously > net_line_y + hysteresis_px for confirm_frames frames,
close the window and process it.
Inside the completed window, the bounce candidate is the first local maximum
of pixel_y.
```

Reasoning: in image coordinates, larger `y` is lower on screen. For the upper
half, the ball moves down toward the bounce, so contact appears as a local
maximum in `pixel_y`.

### Lower-Half Recorder

This recorder handles a trajectory window that starts in the lower image half.

```text
Start recording when y > net_line_y + hysteresis_px.
Keep appending detections while the ball is in this flight window.
When y is continuously < net_line_y - hysteresis_px for confirm_frames frames,
close the window and process it.
Inside the completed window, the bounce candidate is the first local minimum
of pixel_y.
```

Reasoning: for the lower half, the relevant bounce appears as a local minimum
in `pixel_y` under this camera geometry.

## Starting Parameters

These are initial field-test values, not final truth:

```text
camera = cam68
net_line_y = 260
hysteresis_px = 8
confirm_frames = 5
min_segment_len = 8
max_gap_frames = 3
min_prominence_px = 8-12
cooldown_frames = 12
edge_guard_frames = 2
```

Notes:

- `confirm_frames=5` means the crossing must be stable for 5 consecutive
  detections before the window closes.
- `max_gap_frames=3` allows small TrackNet gaps to be interpolated or tolerated.
  Bigger gaps should lower confidence or reject the window.
- `cooldown_frames=12` roughly matches the existing realtime bounce cooldown.

## Processing Pipeline

```text
TrackNet detections
  -> selected cam68 ball point
  -> net-line window recorder
  -> small-gap interpolation / Kalman or SG smoothing
  -> window quality checks
  -> extremum candidate search
  -> direction verification
  -> homography landing projection
  -> BounceEvent
```

## Window Quality Checks

Before accepting a bounce from a completed window, reject or downgrade windows
that look physically unreliable.

Recommended checks:

```text
Reject if the window has fewer than min_segment_len valid points.
Reject if there is a gap longer than max_gap_frames.
Reject if a single-frame jump is too large for plausible ball motion.
Reject if the candidate extremum is within edge_guard_frames of the window edge.
Reject if the smoothed curve residual is too high.
Reject if the extremum prominence is below min_prominence_px.
```

The goal is to avoid treating a single noisy TrackNet spike as a bounce.

## Smoothing And Gap Filling

TrackNet can miss frames and occasionally drift. The detector should not rely
on raw detections only.

Recommended approach:

```text
Use raw points for recorder start/stop logic.
Inside a completed window, fill very small gaps by interpolation.
Apply a light Kalman filter or Savitzky-Golay smoothing to pixel_y.
Find extrema on the smoothed series.
Keep the original measured frame/pixel available for debugging.
```

Large gaps should not be fully hallucinated. They should either reject the
window or mark the bounce as low confidence.

## Direction Verification

The extremum must match the expected motion direction around the bounce.

For the upper-half recorder:

```text
Candidate = first local maximum of pixel_y.
Require y to generally increase before the candidate.
Require y to generally decrease after the candidate.
```

For the lower-half recorder:

```text
Candidate = first local minimum of pixel_y.
Require y to generally decrease before the candidate.
Require y to generally increase after the candidate.
```

This check prevents a random spike from becoming a bounce.

## Landing Projection

Once the bounce frame is selected:

```text
Use the bounce frame's pixel_x / pixel_y.
Project through cam68 homography to world_x / world_y.
Compute in_court from world_x / world_y.
Emit BounceEvent with source_camera = "cam68".
```

If the chosen bounce frame came from an interpolated point, prefer the nearest
real detection frame for homography projection, or mark the event as lower
confidence.

## Dashboard A/B Plan

The dashboard mode switch should be used as the field-test surface:

```text
stereo     = current production stereo 3D Hybrid path
mono_cam68 = cam68 single-camera net-line window detector
mono_cam66 = future / optional single-camera detector for cam66
```

During testing, compare:

```text
recent_bounces count
minimap landing position
in/out correctness
obvious missed bounces
false double-bounces
source_camera field
detect_delay
```

The first success criterion is not perfect scoring. The first success criterion
is whether `mono_cam68` produces visibly plausible bounce events in situations
where stereo 3D misses because cross-camera pairing or triangulation is sparse.

## Implementation Status

V1 is implemented in `SingleCamBounceDetector`:

```text
app/analytics.py
```

The realtime dashboard path wires it through:

```text
app/orchestrator.py
```

Current mode behavior:

```text
stereo     -> existing stereo 3D HybridBounceDetector
mono_cam68 -> cam68 net-line window detector, net_line_y=260
mono_cam66 -> same detector shape, available for testing but not field-tuned
```

Runtime stats are exposed under `/api/status`:

```text
analytics.single_cam_bounce_stats.cam68
analytics.single_cam_bounce_stats.cam66
```

Important stats include:

```text
windows_started_upper
windows_started_lower
windows_closed_upper
windows_closed_lower
windows_processed
accepted
rejected_short
rejected_gap
rejected_jump
rejected_edge
rejected_prominence
rejected_direction
rejected_residual
rejected_court
active_upper
active_lower
last_reject_reason
```

## Open Questions

- Is `net_line_y=260` stable for the exact dashboard input resolution used in
  production, or does it need to be derived from the homography / rendered
  frame size?
- Should recorder start/stop use raw TrackNet points, Kalman points, or both?
- What is the best jump threshold for detecting drift without rejecting real
  fast balls?
- Should single-camera mode write to production `_live_bounces`, or first write
  to a sidecar such as `single_cam_bounces_eval` during early tuning?
- Should stereo mode later use mono cam68 as fallback when 3D points disappear
  for several frames?
