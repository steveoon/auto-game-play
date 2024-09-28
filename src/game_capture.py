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
from datetime import datetime

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
monitor = {"top": 0, "left": 0, "width": 2560, "height": 1440}

# 定义类别名称（根据模型的训练集）
names = model.module.names if hasattr(model, "module") else model.names

# 新增截图保存相关的设置
SCREENSHOT_DIR = "screenshots"
SCREENSHOT_COUNT = 100
SCREENSHOT_INTERVAL = 3  # 每3秒截一张图
GAMEPLAY_DURATION = 300  # 5分钟的游戏时间

# 确保截图保存目录存在
os.makedirs(SCREENSHOT_DIR, exist_ok=True)

# Step 2: 循环捕获屏幕帧
with mss.mss() as sct:
    start_time = time.time()
    screenshot_count = 0

    while True:
        frame_start_time = time.time()

        # 截屏并转换为 NumPy 数组
        img = np.array(sct.grab(monitor))

        # BGR 转 RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # 保存截图
        elapsed_time = time.time() - start_time
        if elapsed_time >= GAMEPLAY_DURATION:
            break

        if (
            screenshot_count < SCREENSHOT_COUNT
            and elapsed_time % SCREENSHOT_INTERVAL < 0.1
        ):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{SCREENSHOT_DIR}/frame_{screenshot_count:03d}_{timestamp}.png"
            cv2.imwrite(filename, img)
            screenshot_count += 1
            print(f"保存截图: {filename}")

        # 修改图像处理部分
        img_model = cv2.resize(img_rgb, (1280, 1280))
        img_normalized = img_model / 255.0
        img_input = np.transpose(img_normalized, (2, 0, 1))
        img_input = np.expand_dims(img_input, 0)
        img_tensor = torch.from_numpy(img_input).float().to(device)

        # 模型推理
        with torch.no_grad():
            outputs = model(img_tensor)[0]
            detections = non_max_suppression(outputs, conf_thres=0.25, iou_thres=0.45)[
                0
            ]

        # 边界框缩放
        if detections is not None:
            detections[:, :4] = scale_boxes(
                img_model.shape[:2], detections[:, :4], img.shape
            ).round()

        # 如果 detections 在 GPU 上,将其移回 CPU
        if detections is not None:
            detections = detections.cpu()

        # 解析检测结果并生成指令
        instructions = []
        if detections is not None and len(detections):
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

        # 调整显示窗口
        cv2.namedWindow("Game Capture", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Game Capture", 1280, 720)
        cv2.imshow("Game Capture", img)

        # 控制帧率和检查退出条件
        frame_end_time = time.time()
        frame_elapsed_time = frame_end_time - frame_start_time
        delay = max(
            1, int((1 / 15 - frame_elapsed_time) * 1000)
        )  # 以15帧每秒的速度运行
        if cv2.waitKey(delay) & 0xFF == ord("q"):
            break

        print(f"Frame processed in {frame_elapsed_time:.3f} seconds")
        print(f"Detected {len(instructions)} objects")

        # 打印更详细的信息
        for instruction in instructions:
            print(
                f"检测到: {instruction['object']}, 置信度: {instruction['confidence']:.2f}"
            )

    print(f"共保存了 {screenshot_count} 张截图")

cv2.destroyAllWindows()
