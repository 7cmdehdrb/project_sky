import pyrealsense2 as rs
import numpy as np
import cv2
import os

# 저장 폴더 설정
save_folder = "data"
if not os.path.exists(save_folder):
    os.makedirs(save_folder)  # 폴더가 없으면 생성

# RealSense 파이프라인 설정
pipeline = rs.pipeline()
config = rs.config()

# RGB 스트리밍 활성화 (1280x720, 30FPS)
config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)

# 스트리밍 시작
profile = pipeline.start(config)

# RGB 센서 가져오기
sensor = profile.get_device().query_sensors()[1]  # [0]은 Depth, [1]은 Color

# 자동 노출 끄기 및 수동 값 설정
sensor.set_option(rs.option.enable_auto_exposure, 0)  # 자동 노출 비활성화
sensor.set_option(rs.option.exposure, 100)  # 수동 노출값 (100~200 추천)

def get_next_filename():
    """data 폴더 내에서 가장 큰 숫자의 이미지 파일을 찾아 다음 번호를 생성"""
    existing_files = [f for f in os.listdir(save_folder) if f.startswith("image_") and f.endswith(".png")]
    numbers = [int(f.split("_")[1].split(".")[0]) for f in existing_files if f.split("_")[1].split(".")[0].isdigit()]
    
    next_number = max(numbers) + 1 if numbers else 1
    return f"{save_folder}/image_{next_number:04d}.png"

try:
    while True:
        # 프레임 가져오기
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()

        if not color_frame:
            continue

        # 컬러 이미지를 NumPy 배열로 변환
        color_image = np.asanyarray(color_frame.get_data())

        # 원본 해상도 (1280x720)
        h, w = color_image.shape[:2]  # h=720, w=1280

        # 크롭할 영역 계산 (중앙 기준)
        crop_w, crop_h = 640, 480
        start_x = int((w - crop_w) // 2.05)  # 사용자 설정
        start_y = int((h - crop_h) // 2.7)  # 사용자 설정

        # 크롭 범위가 이미지 크기를 초과하지 않도록 제한
        start_x = max(0, min(start_x, w - crop_w))
        start_y = max(0, min(start_y, h - crop_h))

        # 640x480 영역 크롭
        cropped_image = color_image[start_y:start_y + crop_h, start_x:start_x + crop_w]

        # 화면에 표시
        cv2.imshow('Cropped 640x480 RGB Stream', cropped_image)

        # 키 입력 받기
        key = cv2.waitKey(1) & 0xFF

        # 'q' 키를 누르면 종료
        if key == ord('q'):
            break

        # 's' 키를 누르면 이미지 저장
        elif key == ord('s'):
            filename = get_next_filename()
            cv2.imwrite(filename, cropped_image)
            print(f"이미지 저장됨: {filename}")

finally:
    # 스트리밍 종료
    pipeline.stop()
    cv2.destroyAllWindows()
