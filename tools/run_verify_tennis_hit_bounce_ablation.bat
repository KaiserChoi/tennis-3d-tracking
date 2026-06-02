@echo off
setlocal
cd /d "%~dp0\.."

python tools\run_verify_tennis_hit_bounce_ablation.py ^
  --camera cam68 ^
  --frames-dir "D:\tennis-dataset\1001\clip11\cam68_20260404_075325_2min" ^
  --ball-model "D:\tennis\tennis-3d-tracking\yolo_roadmap\best.pt" ^
  --person-model "D:\tennis\tennis-3d-tracking\yolo11n.pt" ^
  --homography "D:\tennis\tennis-3d-tracking\src\homography_matrices.json" ^
  --max-frames 1500

endlocal
