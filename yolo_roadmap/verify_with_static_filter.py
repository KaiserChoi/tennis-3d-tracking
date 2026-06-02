import cv2
from ultralytics import YOLO
import os
import numpy as np

# --- 1. 配置与路径 ---
base_path = r"D:\Work\.Master Folders\MR_ShuttleStrike\yolo_tennis"
model_path = os.path.join(base_path, "runs", "detect", "yolo11_tennis_model_v2_optimized", "weights", "best.pt")
input_video = os.path.join(base_path, "test_2.mp4")
output_video = os.path.join(base_path, "output_visual_masked_25fps.mp4")

# --- 2. 核心参数 ---
MOVE_THRESHOLD = 5      # 像素位移阈值
STATIC_FRAME_LIMIT = 3  # 静态存活上限
model = YOLO(model_path)
track_history = {}

# --- 3. 视频准备 ---
cap = cv2.VideoCapture(input_video)
width, height = int(cap.get(3)), int(cap.get(4))
out = cv2.VideoWriter(output_video, cv2.VideoWriter_fourcc(*'mp4v'), 25, (width, height))

print(f"🚀 视觉提示机制启动：绿色 = 活跃 | 灰色 = 已屏蔽")

while cap.isOpened():
    success, frame = cap.read()
    if not success: break

    # 创建一个用于绘制半透明层的 overlay
    overlay = frame.copy()
    
    results = model.track(frame, persist=True, conf=0.2, device=0, verbose=False)

    if results[0].boxes.id is not None:
        boxes = results[0].boxes.xywh.cpu().numpy()
        track_ids = results[0].boxes.id.int().cpu().numpy()
        
        for box, track_id in zip(boxes, track_ids):
            x, y, w, h = box
            x1, y1, x2, y2 = int(x - w/2), int(y - h/2), int(x + w/2), int(y + h/2)
            
            # 更新轨迹逻辑
            if track_id not in track_history:
                track_history[track_id] = [x, y, 0]
            
            last_x, last_y, static_count = track_history[track_id]
            distance = np.sqrt((x - last_x)**2 + (y - last_y)**2)
            
            if distance < MOVE_THRESHOLD:
                static_count += 1
            else:
                static_count = 0
            
            track_history[track_id] = [x, y, static_count]

            # --- 4. 分支绘图逻辑 ---
            if static_count < STATIC_FRAME_LIMIT:
                # 【活跃目标】 鲜艳的绿色框
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"Ball {track_id}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            else:
                # 【屏蔽目标】 绘制在 overlay 上的灰色填充框
                cv2.rectangle(overlay, (x1, y1), (x2, y2), (100, 100, 100), -1) # 填充灰色
                cv2.putText(frame, "MASKED", (x1, y2 + 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)

    # --- 5. 合并半透明层 ---
    # alpha 是原始画面的权重，beta 是覆盖层的权重
    alpha = 0.8
    cv2.addWeighted(overlay, 1 - alpha, frame, alpha, 0, frame)

    out.write(frame)
    cv2.imshow("Sports Export - Masking Debugger", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"): break

cap.release()
out.release()
cv2.destroyAllWindows()