from ultralytics import YOLO


def train_model():
    model = YOLO("yolo11s-seg.pt")

    results = model.train(
        data="src/test/yolo_dataset/data.yaml",
        # --- 1. 학습 기본 전략 ---
        epochs=300,  # 충분한 학습 기간
        patience=50,  # 진전이 없으면 50 에포크 후 조기 종료
        imgsz=640,
        batch=8,  # 96GB VRAM 활용
        workers=4,
        # --- 2. 최적화 알고리즘 ---
        optimizer="AdamW",  # 대형 모델에 유리한 옵티마이저
        cos_lr=True,  # 부드러운 학습률 감소 (수렴 안정성 확보)
        # --- 3. 데이터 증강 (색상 및 조명) ---
        hsv_h=0.005,
        hsv_s=0.15,
        hsv_v=0.25,
        # --- 4. 데이터 증강 (형태 및 세그멘테이션 특화) ---
        fliplr=0.5,  # 좌우 반전
        degrees=15.0,  # 무작위 회전 (쓰러진 캔/병 대비)
        scale=0.5,  # 무작위 크기 조절 (거리 변화 대비)
        copy_paste=0.3,  # 마스크 복사 붙여넣기 (겹침 현상 대비)
        # --- 결과 저장 ---
        project="YOLO11_Seg_Project",
        name="experiment_optimized",
    )

    print("✅ 최적화된 학습이 완료되었습니다!")


if __name__ == "__main__":
    train_model()
