from ultralytics import YOLO


def train_model():
    # 1. 모델 로드 (제시해주신 가장 강력한 x 모델)
    model = YOLO("yolo11x-seg.pt")

    # 2. 학습 실행
    results = model.train(
        # --- 기본 설정 ---
        data="data.yaml",  # yaml 파일 경로 (이전 단계의 파일명으로 맞췄습니다)
        epochs=100,  # 총 학습 에포크
        imgsz=640,  # 입력 이미지 크기
        # --- 요청하신 Data Augmentation 설정 ---
        hsv_h=0.005,  # 색상(Hue) 변환: 0.5% 범위 내에서 미세하게 화이트밸런스 조정
        hsv_s=0.15,  # 채도(Saturation) 변환: 15% 범위 내 조절
        hsv_v=0.25,  # 명도(Value) 변환: 25% 범위 내 밝기 조절 (요청하신 밝기 증강)
        bgr=0.0,  # BGR 채널 뒤집기 (사용 안 함)
        fliplr=0.5,  # 좌우 반전 확률 (50%)
        flipud=0.0,  # 상하 반전 확률 (사용 안 함)
        # --- 고성능 학습 환경(96GB VRAM) 맞춤 설정 ---
        batch=32,  # 한 번에 처리할 이미지 수 (96GB이므로 32~64까지 넉넉하게 설정 가능)
        device=0,  # 사용할 GPU 번호 (다중 GPU라면 [0, 1] 형태로 입력)
        workers=8,  # 데이터 로딩에 사용할 CPU 코어 수
        # --- 결과 저장 설정 ---
        project="YOLO11_Seg_Project",
        name="experiment_x_model",
    )

    print("✅ 학습이 완료되었습니다!")


if __name__ == "__main__":
    train_model()
