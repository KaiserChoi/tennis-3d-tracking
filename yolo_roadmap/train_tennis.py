from ultralytics import YOLO

# 1. 加载预训练模型
model = YOLO("yolo11n.pt") 

if __name__ == '__main__':
    # 2. 开始优化版训练
    results = model.train(
        data="D:\\Work\\.Master Folders\\MR_ShuttleStrike\\yolo_tennis\\Tennis Ball Detection MR_v1/data.yaml",
        
        # --- 1. 基础与周期设置 ---
        epochs=300,            # 增加轮数，让模型充分学习
        patience=50,           # 早停机制，连续50轮没提升就停止，防过拟合
        imgsz=960,             # 如果显存够且原始图片清晰，可以尝试提至 960 看有无改善
        batch=32,              
        device=0,
        workers=8,
        name="yolo11_tennis_model_v2_optimized",

        # --- 2. 优化器与学习率 ---
        optimizer='AdamW',     # AdamW 对小数据集和Transformer/CNN混合架构更友好
        cos_lr=True,           # 开启余弦退火学习率，让训练后期更平滑
        lr0=0.001,             # 初始学习率 (AdamW推荐值)

        # --- 3. 损失函数权重调整 ---
        box=8.5,               # 默认是 7.5。调高 Box 权重，逼迫模型把框画得更紧，提升 mAP50-95
        cls=0.5,               # 因为只有“网球”一个类别，分类难度极低，降低分类损失比重

        # --- 4. 强力数据增强 (专治小数据集) ---
        mosaic=1.0,            # 强制开启马赛克增强 (默认通常是开启的)
        mixup=0.15,            # 开启 15% 概率的 MixUp (图像融合)，极大提升泛化能力
        degrees=15.0,          # 允许 ±15 度的随机旋转
        scale=0.5,             # 允许图像缩放 ±50% (网球忽大忽小，强化尺度不变性)
        translate=0.1,         # 允许 10% 的图像平移
        hsv_v=0.5,             # 亮度随机变化增强 50% (模拟阴影和强光)
        hsv_s=0.5,             # 饱和度随机变化增强 50%
        close_mosaic=15        # 在最后15个 epoch 关闭马赛克增强，让模型在真实图像上收尾
    )

    print("训练完成，开始导出不同精度的安卓专用模型...")
    # ... 后续导出代码保持不变 ...

    # 3. 导出 FP32 (基准精度)
    model.export(format="tflite", imgsz=640)

    # 4. 导出 FP16 (8 Gen 3 GPU 加速首选，平衡精度与速度)
    model.export(format="tflite", imgsz=640, half=True)

    # 5. 导出 Int8 (极致速度，激活 8 Gen 3 NPU 加速)
    # YOLO 导出 Int8 会自动从数据集中提取图片进行校准
    # model.export(format="tflite", imgsz=640, int8=True, data="data.yaml")

    print("所有模型已保存至 runs/detect/yolo11_tennis_model/weights/ 文件夹中")