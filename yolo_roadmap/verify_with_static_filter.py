import cv2
from ultralytics import YOLO
import os
import numpy as np

# --- 1. 閰嶇疆涓庤矾寰?---
base_path = r"D:\Work\.Master Folders\MR_ShuttleStrike\yolo_tennis"
model_path = os.path.join(base_path, "runs", "detect", "yolo11_tennis_model_v2_optimized", "weights", "best.pt")
input_video = os.path.join(base_path, "test_2.mp4")
output_video = os.path.join(base_path, "output_visual_masked_25fps.mp4")

# --- 2. 鏍稿績鍙傛暟 ---
MOVE_THRESHOLD = 5      # 鍍忕礌浣嶇Щ闃堝€?
STATIC_FRAME_LIMIT = 3  # 闈欐€佸瓨娲讳笂闄?
model = YOLO(model_path)
track_history = {}

# --- 3. 瑙嗛鍑嗗 ---
cap = cv2.VideoCapture(input_video)
width, height = int(cap.get(3)), int(cap.get(4))
out = cv2.VideoWriter(output_video, cv2.VideoWriter_fourcc(*'mp4v'), 25, (width, height))

print("Static filter visualizer started: green = active, gray = masked.")

while cap.isOpened():
    success, frame = cap.read()
    if not success: break

    # 鍒涘缓涓€涓敤浜庣粯鍒跺崐閫忔槑灞傜殑 overlay
    overlay = frame.copy()

    results = model.track(frame, persist=True, conf=0.2, device=0, verbose=False)

    if results[0].boxes.id is not None:
        boxes = results[0].boxes.xywh.cpu().numpy()
        track_ids = results[0].boxes.id.int().cpu().numpy()

        for box, track_id in zip(boxes, track_ids):
            x, y, w, h = box
            x1, y1, x2, y2 = int(x - w/2), int(y - h/2), int(x + w/2), int(y + h/2)

            # 鏇存柊杞ㄨ抗閫昏緫
            if track_id not in track_history:
                track_history[track_id] = [x, y, 0]

            last_x, last_y, static_count = track_history[track_id]
            distance = np.sqrt((x - last_x)**2 + (y - last_y)**2)

            if distance < MOVE_THRESHOLD:
                static_count += 1
            else:
                static_count = 0

            track_history[track_id] = [x, y, static_count]

            # --- 4. 鍒嗘敮缁樺浘閫昏緫 ---
            if static_count < STATIC_FRAME_LIMIT:
                # 銆愭椿璺冪洰鏍囥€?椴滆壋鐨勭豢鑹叉
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"Ball {track_id}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            else:
                # 銆愬睆钄界洰鏍囥€?缁樺埗鍦?overlay 涓婄殑鐏拌壊濉厖妗?
                cv2.rectangle(overlay, (x1, y1), (x2, y2), (100, 100, 100), -1) # 濉厖鐏拌壊
                cv2.putText(frame, "MASKED", (x1, y2 + 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)

    # --- 5. 鍚堝苟鍗婇€忔槑灞?---
    # alpha 鏄師濮嬬敾闈㈢殑鏉冮噸锛宐eta 鏄鐩栧眰鐨勬潈閲?
    alpha = 0.8
    cv2.addWeighted(overlay, 1 - alpha, frame, alpha, 0, frame)

    out.write(frame)
    cv2.imshow("Sports Export - Masking Debugger", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"): break

cap.release()
out.release()
cv2.destroyAllWindows()
