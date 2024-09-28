import sys
import os

yolov5_path = "/Users/rensiwen/Documents/LLMs/yolov5"
sys.path.insert(0, yolov5_path)

print("当前工作目录:", os.getcwd())
print("Python 解释器路径:", sys.executable)
print("Python 搜索路径:", sys.path)

import mss
import numpy as np
import cv2
import torch
import json
import time

print("成功导入基本库")

try:
    from models.experimental import attempt_load
    from utils.general import non_max_suppression, scale_boxes

    print("成功导入 YOLOv5 模块")
except ImportError as e:
    print(f"导入 YOLOv5 模块时出错: {e}")
    import traceback

    traceback.print_exc()
    sys.exit(1)

# 在模型加载之前添加这些行
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"使用设备: {device}")

# Step 1: 初始化模型和参数
try:
    model = attempt_load("yolov5s.pt")
    model = model.to(device)
    model.eval()
    print("成功加载模型到", device)
except Exception as e:
    print(f"加载模型时出错: {e}")
    import traceback

    traceback.print_exc()
    sys.exit(1)

# 确定要截取的屏幕区域（可以根据游戏窗口位置调整）
monitor = {"top": 0, "left": 0, "width": 1920, "height": 1080}

# 定义类别名称（根据模型的训练集）
names = model.module.names if hasattr(model, "module") else model.names

# Step 2: 循环捕获屏幕帧
with mss.mss() as sct:
    while True:
        start_time = time.time()

        # 截屏并转换为 NumPy 数组
        img = np.array(sct.grab(monitor))

        # BGR 转 RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Resize 并归一化
        img_resized = cv2.resize(img_rgb, (640, 480))
        img_normalized = img_resized / 255.0
        img_input = np.transpose(img_normalized, (2, 0, 1))
        img_input = np.expand_dims(img_input, 0)
        img_tensor = torch.from_numpy(img_input).float().to(device)

        # 模型推理
        with torch.no_grad():
            outputs = model(img_tensor)[0]
            # NMS 后处理
            detections = non_max_suppression(outputs, conf_thres=0.25, iou_thres=0.45)[
                0
            ]

        # 如果 detections 在 GPU 上,将其移回 CPU
        if detections is not None:
            detections = detections.cpu()

        # 解析检测结果并生成指令
        instructions = []
        if detections is not None and len(detections):
            # 将检测框尺寸调整回原始图像尺寸
            detections[:, :4] = scale_boxes(
                img_resized.shape[1:], detections[:, :4], img.shape
            ).round()

            for *xyxy, conf, cls in detections:
                label = names[int(cls)]
                x1, y1, x2, y2 = map(int, xyxy)
                confidence = float(conf)

                # 根据检测到的对象生成指令（这里以检测到的对象为例）
                instruction = {
                    "object": label,
                    "bbox": [x1, y1, x2, y2],
                    "confidence": confidence,
                }
                instructions.append(instruction)

                # 在图像上绘制检测框（用于调试）
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    img,
                    f"{label} {confidence:.2f}",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (36, 255, 12),
                    2,
                )

        # 将指令转换为 JSON 格式
        json_data = json.dumps(instructions, ensure_ascii=False)
        print(json_data)

        # 显示图像（用于调试）
        cv2.imshow("Game Capture", img)

        # 控制帧率
        end_time = time.time()
        elapsed_time = end_time - start_time
        delay = max(1, int((1 / 30 - elapsed_time) * 1000))  # 以30帧每秒的速度运行
        if cv2.waitKey(delay) & 0xFF == ord("q"):
            break

        print(f"Frame processed in {elapsed_time:.3f} seconds")
        print(f"Detected {len(instructions)} objects")

cv2.destroyAllWindows()
