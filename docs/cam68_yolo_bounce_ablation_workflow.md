# cam68 YOLO Bounce Ablation Workflow

Use this workflow whenever cam68 YOLO bounce/minimap behavior needs to be
validated or shared with another reviewer.

## Goal

Produce one comparable video per test variant, plus one summary JSON, using the
same input clip, model, homography, bounce parameters, and minimap style.

The fixed visual contract is:

- Main view shows the YOLO box, box-center trajectory, and short bounce highlight.
- YOLO boxes must be visibly reviewable: selected boxes are thick cyan boxes
  with a center marker; masked/static boxes are gray.
- Minimap is `rotate180`.
- Minimap shows bounce points only: green filled IN, red hollow OUT, latest bounce highlighted.
- Generated report files stay under `reports/` and are not committed.

## One-Command Run

From the repo root:

```powershell
tools\run_cam68_yolo_bounce_ablation.bat
```

Equivalent direct command:

```powershell
& 'C:\Users\PC\AppData\Local\Programs\Python\Python310\python.exe' -m tools.run_cam68_yolo_bounce_ablation `
  --frames-dir 'D:\tennis-dataset\1001\clip11\cam68_20260404_075325_2min' `
  --model 'D:\tennis\tennis-3d-tracking\yolo_roadmap\best.pt' `
  --homography 'D:\tennis\tennis-3d-tracking\src\homography_matrices.json' `
  --config 'D:\tennis\tennis-3d-tracking\config.yaml' `
  --max-frames 1500 `
  --out-root 'D:\tennis\tennis-3d-tracking\reports'
```

## Outputs

Each run creates:

```text
reports\cam68_yolo_bounce_ablation_<timestamp>\
```

Required files:

```text
A0_yolo_raw_box_trajectory.mp4
A1_yolo_original_static_trajectory.mp4
A2_yolo_dashboard_static_trajectory.mp4
A3_dashboard_integrated_replay.json
A3_dashboard_integrated_replay_video.mp4
A4_dashboard_integrated_minimap_video.mp4
ablation_summary.json
```

## Variant Meaning

| Variant | Shareable description | Chain |
|---|---|---|
| A0 | YOLO direct detection + bounce. No static filtering; top-confidence box center forms the trajectory. | `YOLO.track -> top box -> detect_single_camera_bounces` |
| A1 | YOLO + static filtering + bounce. Static-looking boxes are filtered before trajectory/bounce detection. | `YOLO.track -> track_id/static_count filter -> active box -> detect_single_camera_bounces` |
| A2 | YOLO + integrated optimized dashboard static filtering + bounce. Uses dashboard YOLO detector behavior. | `YoloRoadmapDetector -> selected kept box -> detect_single_camera_bounces` |
| A3 | YOLO + integrated optimized dashboard event-chain test. Confirms bounce events enter minimap data and 3D push payload from the same source. | `A2 selected box -> Orchestrator.compute_single_cam_bounces -> _live_bounces + _ws_bounce_queue` |
| A4 | Final dashboard presentation preview. Shows the final dashboard-style visual result from integrated dashboard events. | `A3 dashboard events -> final minimap/video render` |

## Acceptance Checklist

For every run, the verifier checks:

- All five MP4 files open.
- Each MP4 has 1500 frames at 25 FPS unless the run intentionally used a smaller smoke-test frame count.
- Main view shows YOLO box and trajectory.
- Selected YOLO boxes are clearly visible on normal frames and bounce frames.
- A1 shows masked/static boxes separately from active trajectory boxes.
- Minimap has no text, star markers, legends, or trajectory lines.
- `ablation_summary.json` has one entry for A0 through A4.
- A3 `checks.minimap_and_push_counts_match` is `true`.
- A3 `checks.minimap_and_push_frame_lists_match` is `true`.
- A2, A3, and A4 bounce frame lists are explainably consistent.

## Reporting Template

When sending the videos to someone else, use this short description:

```text
A0: YOLO direct detection + bounce.
A1: YOLO + static filtering + bounce.
A2: YOLO + integrated optimized dashboard static filtering + bounce.
A3: YOLO + integrated optimized dashboard event-chain test, including minimap and 3D push payload audit.
A4: Final dashboard-style preview using the integrated dashboard events.
```

Also attach `ablation_summary.json`; it is the source of truth for counts,
IN/OUT split, bounce frame lists, and A3 push/minimap consistency checks.
