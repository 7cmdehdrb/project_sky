import os
import torch
from PIL import Image, ImageDraw
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Normalize
from torchvision.models.segmentation import fcn_resnet50
from tqdm import tqdm
from torch import nn
import argparse

# Config class
class Config:
    TEST_DIR = "./test_real"
    OUTPUT_DIR = "./outputs"
    MODEL_PATH = os.path.join(OUTPUT_DIR, "best_model.pth")
    DATASET_STATS_PATH = os.path.join(OUTPUT_DIR, "dataset_stats.txt")
    BATCH_SIZE = 2 #원래 64
    CLASS_NAMES = ["can_1", "can_2", "can_3", "cup_1", "cup_2", "cup_3", "mug_1", "mug_2", "mug_3", "bottle_1", "bottle_2", "bottle_3"]  # 클래스 이름 배열
    NUM_CLASSES = len(CLASS_NAMES)


# FCNModel class
class FCNModel(nn.Module):
    def __init__(self):
        super(FCNModel, self).__init__()
        self.model = fcn_resnet50(weights=None)  # pretrained=False는 weights=None으로 대체됨
        self.model.classifier[4] = nn.Conv2d(512, Config.NUM_CLASSES, kernel_size=1)

    def forward(self, x):
        return self.model(x)['out']


# TestDataset class
class TestDataset(Dataset):
    def __init__(self, image_paths, transforms=None):
        self.image_paths = image_paths
        self.transforms = transforms

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        img_path = self.image_paths[index]
        img = Image.open(img_path).convert("RGB")

        img = np.array(img)
        if img.ndim == 2:
            img = np.stack([img] * 3, axis=-1)
        elif img.shape[-1] == 4:
            img = img[..., :3]

        img = img.astype(np.float32) / 255.0
        img = img.transpose(2, 0, 1)

        if self.transforms:
            img = torch.tensor(img, dtype=torch.float32)
            img = self.transforms(img)
        else:
            img = torch.tensor(img, dtype=torch.float32)

        return img, img_path


def load_dataset_stats(stats_path):
    with open(stats_path, "r") as f:
        lines = f.readlines()
        mean = np.array(eval(lines[0].split(":")[-1].strip()))
        std = np.array(eval(lines[1].split(":")[-1].strip()))
    return mean, std


def load_test_data(test_dir, target_class):
    class_dir = os.path.join(test_dir, target_class)
    if not os.path.isdir(class_dir):
        raise ValueError(f"Target class directory '{class_dir}' does not exist.")

    image_paths = [
        os.path.join(class_dir, file)
        for file in os.listdir(class_dir)
        if os.path.isfile(os.path.join(class_dir, file)) and file.lower().endswith(('.png', '.jpg', '.jpeg'))
    ]
    return sorted(image_paths)


def save_overlay_with_distributions(image, previous_dist, current_dist, final_dist, output_path, target_class, alpha=0.3):
    """
    원본 이미지 위에 이전 분포(주황색), 현재 분포(초록색), 최종 분포(파란색)를 오버레이하여 저장.

    Args:
        image (PIL.Image): 원본 RGB 이미지
        previous_dist (numpy.array): 이전 분포 (1D)
        current_dist (numpy.array): 현재 분포 (1D)
        final_dist (numpy.array): 최종 분포 (1D)
        output_path (str): 저장 경로
        target_class (str): 타겟 물체 이름
        alpha (float): 투명도 (0.0 ~ 1.0)
    """
    # 폴더 생성
    overlay_dir = os.path.join(Config.OUTPUT_DIR, f"overlay_images_{target_class}")
    os.makedirs(overlay_dir, exist_ok=True)

    file_name = os.path.basename(output_path)
    output_path = os.path.join(overlay_dir, file_name)

    width, height = image.size
    overlay = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    def normalize_and_clip(dist, color):
        if dist is not None:
            max_val = np.max(dist)
            if max_val > 0:
                normalized_dist = (dist / max_val) * height
                normalized_dist = np.clip(normalized_dist, 0, height)  # 음수 방지 및 최대값 제한
                for x, value in enumerate(normalized_dist):
                    y_top = height - int(value)
                    y_top = np.clip(y_top, 0, height)  # y_top 값 클리핑
                    draw.rectangle([x, y_top, x + 1, height], fill=(*color, int(255 * alpha)))

    # 이전 분포: 주황색
    normalize_and_clip(previous_dist, (255, 165, 0))

    

    # 최종 분포: 파란색
    normalize_and_clip(final_dist, (0, 0, 255))
    
    # 현재 분포: 초록색
    normalize_and_clip(current_dist, (0, 255, 0))

    combined = Image.alpha_composite(image.convert("RGBA"), overlay)
    combined = combined.convert("RGB")
    combined.save(output_path)


def predict_save_and_generate_distribution(model, data_loader, target_class_idx, device, target_class):
    model.eval()
    results = []
    distributions = []
    previous_distribution = None  # 이전 step 분포 저장 변수
    gain = 2.0  # 지수 함수의 gain 값
    gamma = 0.7  # 이전 분포와 현재 분포의 가중치를 위한 값(현재 확율을 얼마나 반영할지)

    with torch.no_grad():
        for imgs, img_paths in tqdm(data_loader):
            imgs = imgs.to(device)
            outputs = model(imgs)

            preds = outputs[:, target_class_idx, :, :].cpu().numpy()
            for i, path in enumerate(img_paths):
                pred = preds[i]

                # 이미지 저장용 2D 예측 결과
                pred_img = (pred * 255)
                pred_img = np.clip(pred_img, 0, 255).astype(np.uint8)
                results.append((pred_img, path))

                # 비선형 가중치 적용
                max_val = np.max(pred)
                if max_val > 0:
                    normalized_pred = pred / max_val  # 0~1로 정규화
                    weighted_pred = normalized_pred * np.exp(-gain * (1 - normalized_pred))  # 지수 함수로 가중치 적용
                else:
                    weighted_pred = pred  # max_val이 0이면 그대로 사용

                # y축으로 합산하여 현재 분포 생성
                current_distribution = np.sum(weighted_pred, axis=0)

                if previous_distribution is not None:
                    # 이전 분포와 현재 분포의 가중 합으로 최종 분포 생성
                    final_distribution = gamma * current_distribution + (1 - gamma) * previous_distribution
                else:
                    final_distribution = current_distribution

                distributions.append((final_distribution, path))

                original_img = Image.open(path).convert("RGB")

                # Save overlayed image with previous, current, and final distributions
                save_overlay_with_distributions(
                    original_img, previous_distribution, current_distribution, final_distribution, path, target_class
                )

                # Update previous_distribution for the next step
                previous_distribution = final_distribution

    return results, distributions


def save_predictions(predictions, target_class):
    output_class_dir = os.path.join(Config.OUTPUT_DIR, f"test_predictions_{target_class}")
    os.makedirs(output_class_dir, exist_ok=True)

    for idx, (pred, path) in enumerate(predictions):
        file_name = os.path.basename(path)
        save_path = os.path.join(output_class_dir, f"pred_{idx + 1}_{file_name}")
        pred_image = Image.fromarray(pred)
        pred_image.save(save_path)

    print(f"Saved predictions for '{target_class}' in '{output_class_dir}'")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=str, required=True, help="Target class name for prediction.")
    args = parser.parse_args()

    target_class = args.target
    if target_class not in Config.CLASS_NAMES:
        raise ValueError(f"Invalid target class '{target_class}'. Must be one of {Config.CLASS_NAMES}.")

    target_class_idx = Config.CLASS_NAMES.index(target_class)

    test_x = load_test_data(Config.TEST_DIR, target_class)
    mean, std = load_dataset_stats(Config.DATASET_STATS_PATH)
    print(f"Using normalization: mean={mean}, std={std}")

    transforms = Normalize(mean=mean.tolist(), std=std.tolist())
    test_dataset = TestDataset(test_x, transforms=transforms)
    test_loader = DataLoader(test_dataset, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=4)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = FCNModel()
    state_dict = torch.load(Config.MODEL_PATH, map_location=device)
    filtered_state_dict = {k: v for k, v in state_dict.items() if "aux_classifier" not in k}
    model.load_state_dict(filtered_state_dict, strict=False)
    model = model.to(device)

    predictions, distributions = predict_save_and_generate_distribution(model, test_loader, target_class_idx, device, target_class)
    save_predictions(predictions, target_class)


if __name__ == "__main__":
    main()
