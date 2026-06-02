@echo off
setlocal

set "REPO_ROOT=D:\tennis\tennis-3d-tracking"
set "PYTHON=C:\Users\PC\AppData\Local\Programs\Python\Python310\python.exe"

cd /d "%REPO_ROOT%" || exit /b 1

"%PYTHON%" -m tools.run_cam68_yolo_bounce_ablation ^
  --frames-dir "D:\tennis-dataset\1001\clip11\cam68_20260404_075325_2min" ^
  --model "D:\tennis\tennis-3d-tracking\yolo_roadmap\best.pt" ^
  --homography "D:\tennis\tennis-3d-tracking\src\homography_matrices.json" ^
  --config "D:\tennis\tennis-3d-tracking\config.yaml" ^
  --max-frames 1500 ^
  --progress-every 250 ^
  --out-root "D:\tennis\tennis-3d-tracking\reports"

exit /b %ERRORLEVEL%
