# src/check_class_names.py
from ultralytics import YOLO

# 모델 로드
model_path = "src/object_tracker/resource/best_hg.pt"
model = YOLO(model_path)

# 클래스 이름 확인
class_names = model.names

print("학습된 클래스 라벨:")
for idx, name in class_names.items():
    print(f"  {idx}: {name}")
