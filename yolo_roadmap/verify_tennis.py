import cv2
from ultralytics import YOLO
import os

# 1. 路径与配置
base_path = r"D:\Work\.Master Folders\MR_ShuttleStrike\yolo_tennis"
model_path = os.path.join(base_path, "runs", "detect", "yolo11_tennis_model_v2_optimized", "weights", "best.pt")
input_video = os.path.join(base_path, "test_4.mp4")
output_video = os.path.join(base_path, "output_tennis_25fps_v3.mp4")

# 2. 加载模型 (调用 4080m 算力)
model = YOLO(model_path)

# 3. 读取视频
cap = cv2.VideoCapture(input_video)
width = int(cap.get(cv2.COMM_PROP_FRAME_WIDTH) if hasattr(cv2, 'COMM_PROP_FRAME_WIDTH') else cap.get(3))
height = int(cap.get(cv2.COMM_PROP_FRAME_HEIGHT) if hasattr(cv2, 'COMM_PROP_FRAME_HEIGHT') else cap.get(4))

# 4. 设置视频写入器
# 使用 'mp4v' 编码器，帧率强制设定为 25
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video, fourcc, 25, (width, height))

if not cap.isOpened():
    print(f"❌ 无法打开源视频: {input_video}")
    exit()

print(f"🚀 开始检测并合成视频...")
print(f"🎥 设定帧率: 25 FPS | 输出路径: {output_video}")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # 5. 推理 (利用 4080m 的 device=0)
    # stream=True 适合长视频处理，节省显存
    results = model.predict(frame, conf=0.25, device=0, stream=True)

    for r in results:
        # 绘制检测框和标签
        annotated_frame = r.plot()

        # 6. 写入本地文件
        out.write(annotated_frame)

        # 7. 实时显示 (可选，按 'q' 键可停止处理并保存)
        cv2.imshow("Real-time Tennis Detection", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        print("停止处理，正在保存已完成部分...")
        break

# 释放资源
cap.release()
out.release()
cv2.destroyAllWindows()

print(f"✅ 合成完毕！文件已保存至: {output_video}")