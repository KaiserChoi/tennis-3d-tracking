# Dashboard Realtime Handoff Plan

> Status: planning and validation handoff. This document records the current findings and the implementation plans that should be given to code executors and validators.

## Current Repository Context

- Local repository: `D:\tennis\tennis-3d-tracking`
- Current branch: `codex/deepen-codebase-modules`
- Remote: `git@github.com:KaiserChoi/tennis-3d-tracking.git`
- Current service process observed on port 8000 uses `C:\Users\PC\miniconda3\python.exe`.
- The checkout root is the active branch code location. `D:\tennis\tennis-3d-tracking\.git\worktrees` is Git metadata, not the code directory to edit.

## Reproduced Findings

### Camera 2 NO MODEL

`Camera 2 NO MODEL` is caused by model loading failure, not by the camera stream itself.

Observed chain:

1. Frontend displays `NO MODEL` when a running pipeline has `inference_ready === false`.
2. `cam68` can be running while inference is disabled.
3. Switching to TrackNet uses `model_weight/TrackNet_finetuned.onnx`.
4. The active Python environment is missing `onnxruntime` and `onnx`.
5. Current TrackNet loader falls back to PyTorch when ONNX Runtime import fails.
6. The fallback then calls `torch.load()` on an `.onnx` file and raises `invalid load key, '\x08'`.

Important distinction:

- TrackNet ball model: `model_weight/TrackNet_finetuned.onnx`
- HRNet ball model: `model_weight/hrnet_tennis.onnx`
- YOLO ball model: `yolo_roadmap/best.pt`
- Player pose model: `model_weight/yolo26x-pose.pt`

The YOLO ball detector is not `model_weight/yolo26x-pose.pt`; that file is for player pose.

### Live Dashboard Lag

The dashboard can become janky because `/api/status` returns a very large analytics payload while the page polls it frequently. During inspection, the payload included hundreds of recent bounces, hits, and speed events. The camera preview is also throttled in the pipeline when not recording, which can make the video area look like it is dropping frames.

### 3D Push Chain

The current intended chain is:

`frontend 3D Push button -> /api/3d-display/enable -> orchestrator _ws_enabled -> accepted live bounce -> _record_live_bounce_locked -> _enqueue_ws_bounce_locked -> _ws_push_loop`

The chain should be verified by sequence number, not just by "button enabled" state.

## Plan 1: Make TrackNet and YOLO Both Runnable

### Executor Tasks

1. Install required ONNX dependencies into the same environment that runs the service:

   ```powershell
   & C:\Users\PC\miniconda3\python.exe -m pip install -r requirements.txt
   ```

   If a smaller install is desired:

   ```powershell
   & C:\Users\PC\miniconda3\python.exe -m pip install onnxruntime-gpu onnx
   ```

2. Update TrackNet loading in `app/pipeline/inference.py`.

   Required behavior:

   - If `model_path` ends with `.onnx`, load it only with ONNX Runtime.
   - If ONNX Runtime is missing, raise a clear error such as `onnxruntime is required for ONNX TrackNet model`.
   - Do not fall back to PyTorch for an `.onnx` file.
   - PyTorch fallback is allowed only when the model path is a real `.pt` checkpoint.

3. Keep runtime model switching paths explicit:

   - `tracknet` -> `model_weight/TrackNet_finetuned.onnx`
   - `hrnet` -> `model_weight/hrnet_tennis.onnx`
   - `yolo_roadmap` -> `yolo_roadmap/best.pt`

4. Improve the dashboard badge tooltip so model-loading failures are visible from the UI.

### Acceptance Criteria

- `C:\Users\PC\miniconda3\python.exe` can import `onnxruntime`.
- `onnxruntime.get_available_providers()` returns at least `CPUExecutionProvider`; CUDA is preferred when available.
- Switching to YOLO works:

  ```powershell
  Invoke-RestMethod -Method Post http://127.0.0.1:8000/api/model/switch/yolo_roadmap
  ```

  Then `/api/status` shows `cam68.inference_ready=true` and `detector_stats.type=yolo_roadmap`.

- Switching to TrackNet works:

  ```powershell
  Invoke-RestMethod -Method Post http://127.0.0.1:8000/api/model/switch/tracknet
  ```

  Then `/api/status` shows `cam68.inference_ready=true`, and logs include `TrackNet using ONNX Runtime`.

- The logs no longer show `invalid load key, '\x08'` when TrackNet is selected.
- The dashboard does not show `Camera 2 NO MODEL` after a successful TrackNet or YOLO switch.

## Plan 2: Simplify Minimap Bounce Rendering

### Executor Tasks

1. Use dashboard-accepted live bounces as the only minimap event source.
2. Remove non-bounce visual clutter from the minimap:

   - No star markers.
   - No text labels.
   - No hit markers.
   - No speed-event markers.
   - No red `OUT` text.
   - No GT/debug diamonds in normal mode.

3. Render bounce status with simple shapes:

   - In-court bounce: small green filled circle.
   - Out bounce: small red hollow circle.

4. Highlight the latest visible bounce without text:

   - Latest in-court bounce: larger green filled circle plus subtle translucent ring.
   - Latest out bounce: larger red hollow circle plus subtle translucent ring.

5. Add a minimap refresh action that clears visible old points only.

   It should not clear backend analytics, live bounce history, or 3D push state.

6. Add a minimap flip toggle.

   Required behavior:

   - Flip must be a display transform only.
   - Do not mutate stored bounce `x`, `y`, or `in_court`.
   - Centralize mapping in one helper such as `courtToMinimap`.
   - Treat flip as a 180-degree view rotation unless product explicitly asks for only vertical mirroring.

### Acceptance Criteria

- With multiple live bounces present, the minimap contains only circles and no text.
- In-court points are green filled circles.
- Out points are red hollow circles.
- Refresh removes existing visible markers, and new accepted bounces appear after refresh.
- Refresh does not reduce backend `total_bounces`.
- Latest visible bounce is visually distinguishable.
- Flip preserves in/out classification and only changes display coordinates.
- Flipping on and off returns points to their original displayed positions.

## Plan 3: Improve Main Video Ball Overlay

### Executor Tasks

1. Separate normal viewing overlay from debug overlay.
2. In normal viewing mode, draw only the selected ball position for each camera.
3. Keep TrackNet top-k candidates, YOLO boxes, source labels, confidence, and diagnostics behind a debug toggle.
4. Prefer `latest_overlay_detections` for display when available.
5. Keep overlay drawing lightweight and animation-frame driven.

### Acceptance Criteria

- Normal mode shows a clean ball marker without diagnostic text.
- Debug mode can still show model details for troubleshooting.
- YOLO mode and TrackNet mode both update the main video overlay.
- Overlay state does not flicker when `/api/status` is delayed.
- Missing detections fade out gracefully instead of leaving stale permanent markers.

## Plan 4: Verify YOLO Bounce to 3D Push Synchronization

### Executor Tasks

1. Add or verify telemetry for the full push chain:

   - last accepted bounce sequence
   - last queued WebSocket sequence
   - last sent WebSocket sequence
   - pending queue length
   - send error count
   - last send error

2. Ensure every accepted YOLO live bounce after enabling 3D Push is either queued and sent, or rejected with a clear reason.
3. Add a focused test or diagnostic script that enables 3D Push, injects or waits for one accepted YOLO bounce, and verifies sequence consistency.
4. Ensure queue overflow does not silently hide sync failure.

### Acceptance Criteria

- After enabling 3D Push, a new accepted YOLO bounce increments `last accepted sequence`.
- The same sequence appears in `last queued WebSocket sequence`.
- The same sequence appears in `last sent WebSocket sequence` after WebSocket send succeeds.
- If the remote endpoint is unavailable, `ws_error_count` increments and `ws_last_error` is visible.
- Pending queue length returns toward zero when the remote endpoint is reachable.

## Plan 5: Reduce Dashboard Lag and Dropped Frames

### Executor Tasks

1. Reduce homepage status payload size.

   Recommended approaches:

   - Add lightweight query params such as `/api/status?dashboard=1&event_limit=40`.
   - Or add a dedicated dashboard summary endpoint.
   - Do not send full `recent_hits` and `recent_speed_events` arrays to the homepage unless debug mode asks for them.

2. Slow the dashboard polling interval from 500 ms to around 1000-1500 ms.
3. Keep canvas/video overlay rendering on `requestAnimationFrame`.
4. Improve preview frame delivery.

   The current non-recording preview is throttled by frame stride. Make preview FPS configurable or time-based, targeting about 8-12 FPS for normal live viewing.

5. Add a simple operational check for stale Python/FFmpeg processes, because multiple old workers can make the current server look slow.

### Acceptance Criteria

- Dashboard status response is under 100 KB, ideally under 50 KB, during a normal live session.
- `/api/status` or the new lightweight endpoint responds in under 150 ms locally during normal operation.
- Chrome Network panel does not show stacked pending status requests.
- The main preview visually reaches at least 8-12 FPS when not recording.
- Minimap totals, latest bounce, current model status, and camera badges still update correctly.

## Plan 6: Git and Deployment Hygiene

### Executor Tasks

1. Keep source, tests, docs, and lightweight config tracked.
2. Keep runtime artifacts untracked:

   - `logs/`
   - `reports/`
   - `debug_output/`
   - MediaMTX binaries and zips
   - local model weights

3. Avoid `git add -A` in this repository unless the status has been reviewed.
4. For deployment, prefer a simple flow:

   ```powershell
   git fetch origin
   git pull --ff-only origin main
   ```

   Or pull a named feature branch if deploying before merge.

### Acceptance Criteria

- `git status --short` does not show generated logs, debug output, report folders, or local binary/model artifacts.
- Commits contain intentional code/docs/config changes only.
- The pushed branch is available on GitHub.
- Deployment machine can pull the branch without needing local generated files.

## Suggested Validator Checklist

1. Confirm both model switches:

   ```powershell
   Invoke-RestMethod -Method Post http://127.0.0.1:8000/api/model/switch/yolo_roadmap
   Invoke-RestMethod http://127.0.0.1:8000/api/status
   Invoke-RestMethod -Method Post http://127.0.0.1:8000/api/model/switch/tracknet
   Invoke-RestMethod http://127.0.0.1:8000/api/status
   ```

2. Watch dashboard camera badges after each switch.
3. Verify minimap shapes, refresh, latest highlight, and flip behavior with live bounces.
4. Enable 3D Push and verify accepted bounce sequence moves through queue and send telemetry.
5. Inspect Chrome Network timing and response size for dashboard polling.
6. Review `git status --short` before every commit.
