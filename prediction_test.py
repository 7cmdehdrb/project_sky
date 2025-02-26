from ultralytics import YOLO
from ultralytics.engine.results import Results, Masks, Boxes
import cv2
from matplotlib import pyplot as plt
import os
import sys
import random
from PIL import Image, ImageEnhance
import numpy as np
import json


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


# 학습된 모델 로드
model = YOLO("/home/min/7cmdehdrb/ros2_ws/best.pt")

color_dict = {
    0: (255, 0, 0),  # Red
    1: (0, 255, 0),  # Green
    2: (0, 0, 255),  # Blue
    3: (255, 255, 0),  # Yellow
    4: (255, 0, 255),  # Magenta
    5: (0, 255, 255),  # Cyan
    6: (128, 0, 0),  # Dark Red
    7: (0, 128, 0),  # Dark Green
    8: (0, 0, 128),  # Navy
    9: (128, 128, 0),  # Olive
    10: (128, 0, 128),  # Purple
    11: (0, 128, 128),  # Teal
    12: (192, 192, 192),  # Silver
    13: (255, 165, 0),  # Orange
    14: (0, 0, 0),  # Black
}


image_folder = "/home/min/7cmdehdrb/ros2_ws/real_data"
image_files = os.listdir(image_folder)

# 이미지 파일 목록을 랜덤하게 섞기
random.shuffle(image_files)

# real 이미지의 통계값을 불러오기
with open("sim_stats.json", "r") as f:
    real_stats = json.load(f)


for idx, image_path in enumerate(image_files):
    image_path = os.path.join(image_folder, image_path)

    try:
        # 이미지 로드
        img = cv2.imread(image_path)
        # color_modified_img = adjust_sim_image(image_path, real_stats)
        color_modified_img = img

        if img is None:
            print("❌ 이미지 파일을 찾을 수 없습니다. 경로를 확인하세요.")
            break

        # YOLO 모델을 사용하여 객체 감지
        results: Results = model(color_modified_img)[0]

        cls = results.boxes.cls.cpu().numpy()
        boxes = results.boxes.xyxy.cpu().numpy().astype(int)
        conf = results.boxes.conf.cpu().numpy()
        names: dict = results.names
        masks: Masks = results.masks

        mode = "MASK"

        if mode == "ALL":
            # 결과 이미지 가져오기
            img = results.plot()  # 감지된 객체를 이미지에 그려줌

        elif mode == "MASK":
            masks = results.masks.data.cpu().numpy()

            for idx, mask in enumerate(masks):
                x1, y1, x2, y2 = boxes[idx]  # 바운딩 박스 좌표
                confidence = conf[idx]  # 신뢰도

                if confidence < 0.7:
                    continue

                # 색상 및 텍스트 설정
                class_id = int(cls[idx])
                color = color_dict[class_id]
                thickness = 2  # 선 굵기
                font = cv2.FONT_HERSHEY_SIMPLEX  # 폰트 스타일
                font_scale = 0.5  # 폰트 크기

                # 클래스 및 신뢰도 표시
                label = f"{names[class_id]}"
                cv2.putText(
                    img, label, (x1, y1 - 10), font, font_scale, color, thickness
                )

                mask_binary = (mask * 255).astype(np.uint8)
                contours, _ = cv2.findContours(
                    mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )

                # 컨투어 그리기
                cv2.drawContours(img, contours, -1, color, 2)

        elif mode == "BBOX":
            # 바운딩 박스 그리기
            for i in range(len(boxes)):
                x1, y1, x2, y2 = boxes[i]  # 바운딩 박스 좌표
                confidence = conf[i]  # 신뢰도
                class_id = int(cls[i])  # 클래스 ID

                if confidence < 0.7:
                    continue

                # 색상 및 텍스트 설정
                color = color_dict[class_id]  # 초록색
                thickness = 2  # 선 굵기
                font = cv2.FONT_HERSHEY_SIMPLEX  # 폰트 스타일
                font_scale = 0.5  # 폰트 크기

                # 바운딩 박스 그리기
                cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)

                # 클래스 및 신뢰도 표시
                label = f"{names[class_id]}"
                cv2.putText(
                    img, label, (x1, y1 - 10), font, font_scale, color, thickness
                )

        while True:
            # 이미지 출력
            cv2.imshow("YOLO v11 Object Detection", img)

            # 'n' 키를 누르면 다음 이미지로 이동
            if cv2.waitKey(1) & 0xFF == ord("n"):
                break

    except Exception as e:
        print("❌ 예외 발생:", e)

    # 'q' 키를 누르면 루프 종료
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# 창 닫기
cv2.destroyAllWindows()
