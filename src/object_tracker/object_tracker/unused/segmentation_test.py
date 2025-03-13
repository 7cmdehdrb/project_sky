import numpy as np
import torch
import json

# YOLO
from ultralytics.engine.results import Results, Masks, Boxes
from ultralytics import YOLO

# CV
import cv2
from PIL import Image, ImageEnhance


def adjust_sim_image(image_path, stats):
    """
    sim 이미지를 읽어 real 통계에 맞도록 명도, 채도, 대조를 보정하여 저장
    """
    # PIL로 이미지 로드
    image = Image.open(image_path).convert("RGB")
    image_np = np.array(image)

    # 현재 이미지의 채도, 명도, 대조를 OpenCV를 통해 계산 (HSV 사용)
    img_hsv = cv2.cvtColor(image_np, cv2.COLOR_RGB2HSV)
    curr_brightness = np.mean(img_hsv[:, :, 2])
    curr_saturation = np.mean(img_hsv[:, :, 1])
    curr_contrast = np.std(img_hsv[:, :, 2])

    # enhancement factor 계산 (0으로 나누는 경우 방지)
    brightness_factor = stats["avg_brightness"] / (curr_brightness + 1e-8)
    saturation_factor = stats["avg_saturation"] / (curr_saturation + 1e-8)
    contrast_factor = stats["avg_contrast"] / (curr_contrast + 1e-8)

    # PIL ImageEnhance 모듈로 순차적으로 조정
    # 1) 명도 보정
    enhancer = ImageEnhance.Brightness(image)
    image = enhancer.enhance(brightness_factor)

    # 2) 채도 보정
    enhancer = ImageEnhance.Color(image)
    image = enhancer.enhance(saturation_factor)

    # 3) 대조 보정
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(contrast_factor)

    return image


def main():
    model = YOLO(
        "/home/min/7cmdehdrb/ros2_ws/third_party/yolo/runs0225/segment/train/weights/best.pt",
        verbose=True,
    )
    model.eval()

    image_paths = [
        "/home/min/7cmdehdrb/ros2_ws/src/fcn_network/resource/001.png",
        "/home/min/7cmdehdrb/ros2_ws/src/fcn_network/resource/002.png",
        "/home/min/7cmdehdrb/ros2_ws/src/fcn_network/resource/003.png",
        "/home/min/7cmdehdrb/ros2_ws/src/fcn_network/resource/004.png",
    ]

    for image_path in image_paths:
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        with open(
            "/home/min/7cmdehdrb/ros2_ws/third_party/yolo/sim_stats.json", "r"
        ) as f:
            stats = json.load(f)

        image = adjust_sim_image(image_path, stats=stats)

        results: Results = model(image)[0]

        result_img = results.plot()

        cv2.imshow("result", result_img)

        if cv2.waitKey(0) & 0xFF == ord("q"):
            continue

        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
