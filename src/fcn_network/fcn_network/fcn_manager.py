import os
import rclpy
from rclpy.node import Node
import torch
import torch.nn as nn
import numpy as np
from torch import Tensor
from torchvision.models.segmentation import fcn_resnet50
from torchvision.transforms import Normalize
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

# ROS 2 모듈 (환경에 맞춰 유지)
from ament_index_python.packages import get_package_share_directory

# from your_custom_module import Manager, Node (이 부분은 기존 코드에 맞춰서 사용하세요)
from typing import Tuple


class FCNModel(nn.Module):
    """
    ResNet50 기반의 Fully Convolutional Network (FCN) 모델.
    layer_cnt개의 클래스(채널)를 출력하도록 마지막 분류기(classifier)가 수정되었습니다.
    """

    def __init__(self, layer_cnt: int = 12):
        super(FCNModel, self).__init__()
        # Pretrained 가중치 없이 기본 모델 뼈대 생성
        self.model = fcn_resnet50(weights=None)

        # 출력 채널 수를 layer_cnt개로 맞추기 위해 1x1 합성곱 레이어 수정
        self.model.classifier[4] = nn.Conv2d(512, layer_cnt, kernel_size=1)

    def forward(self, x: Tensor) -> Tensor:
        # FCN 출력 중 메인 결과인 'out' 텐서만 반환
        return self.model(x)["out"]


class FCNManager:
    """
    FCN 모델의 로드, 이미지 전처리, 추론, 후처리를 총괄하는 매니저 클래스.
    """

    def __init__(
        self,
        node: Node,
        fcn_gain: float,
        fcn_gamma: float,
        model_path: str,
        fcn_image_transform: bool = True,
        layer_cnt: int = 12,
    ):
        self._node: Node = node
        self._layer_cnt = layer_cnt

        self._device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self._last_results_data: np.ndarray = None

        # >>> 1. 파라미터 및 변환기(Transformer) 초기화 >>>
        self._do_transform = fcn_image_transform
        self._gain = fcn_gain
        self._gamma = fcn_gamma

        self._transformer = Normalize(
            mean=[0.6490130594444274, 0.638542329009374, 0.609406019484202],
            std=[0.17405719094747593, 0.17758162412947812, 0.20408730509485729],
        )

        # self._transformer = Normalize(
        #     mean=[0.6650218532477484, 0.6619035726250543, 0.636150321943495],
        #     std=[0.1857568159886662, 0.18750179093471428, 0.2114251715800321],
        # )

        # 긴빠이(ginppai) 로직에 사용될 X축 구간 분할 기준 인덱스 (기본값)
        self._peak_boundaries = [0, 185, 320, 455, 640]
        # <<< 1. 파라미터 및 변환기(Transformer) 초기화 <<<

        # >>> 2. 가중치 파일 탐색 및 모델 셋업 >>>
        self._model = self._setup_model(model_path)
        # <<< 2. 가중치 파일 탐색 및 모델 셋업 <<<

    def _setup_model(self, model_path: str) -> FCNModel:
        """가중치를 로드하고 필터링한 뒤, 모델을 평가 모드(eval)로 설정합니다."""
        model = FCNModel(layer_cnt=self._layer_cnt)
        state_dict: dict[str, torch.Tensor] = torch.load(
            model_path, map_location=self._device
        )

        # 보조 분류기(aux_classifier) 가중치 제외
        filtered_state_dict = {
            k: v for k, v in state_dict.items() if "aux_classifier" not in k
        }

        model.eval()
        model.load_state_dict(filtered_state_dict, strict=False)
        return model.to(self._device)

    # ==========================================================
    # 추가 요구사항 2: find_top_peaks_ginppai 인덱스 Setter / Getter
    # ==========================================================
    @property
    def peak_boundaries(self) -> list:
        """현재 설정된 X축 분할 구간 인덱스 리스트를 반환합니다."""
        return self._peak_boundaries

    @peak_boundaries.setter
    def peak_boundaries(self, boundaries: list):
        """
        가변적인 구역을 나누기 위한 경계 인덱스를 설정합니다.
        길이는 2 이상이어야 하며 오름차순이어야 합니다.
        """
        if len(boundaries) < 2:
            raise ValueError(
                "경계 인덱스는 최소 2개 이상의 요소(시작점과 끝점)로 구성되어야 합니다."
            )
        if boundaries != sorted(boundaries):
            raise ValueError("경계 인덱스는 오름차순으로 정렬되어 있어야 합니다.")

        self._peak_boundaries = boundaries

    # ==========================================================

    def preprocess_image(self, img: np.ndarray) -> Tensor:
        """
        (기존 post_process_raw_image 역할)
        원시 이미지를 모델이 요구하는 텐서 형태로 전처리합니다.
        """
        # 채널 수를 3채널(RGB)로 통일
        if img.ndim == 2:
            img = np.stack([img] * 3, axis=-1)
        elif img.shape[-1] == 4:
            img = img[..., :3]

        # 0~1 정규화 및 차원 변경: (H, W, C) -> (C, H, W)
        img = img.astype(np.float32) / 255.0
        img = img.transpose(2, 0, 1)

        tensor_img = torch.tensor(img, dtype=torch.float32)

        if self._do_transform:
            tensor_img = self._transformer(tensor_img)

        return tensor_img

    def predict(self, np_image: np.ndarray) -> np.ndarray:
        """입력 이미지를 모델에 통과시켜 2D 결과 맵을 반환합니다."""
        tensor_img = self.preprocess_image(np_image).to(self._device)
        tensor_img = tensor_img.unsqueeze(0)  # 배치 차원 추가

        # 중요: 추론 시 불필요한 연산과 메모리 낭비를 막기 위해 no_grad() 적용
        with torch.no_grad():
            outputs: Tensor = self._model(tensor_img)

        outputs = outputs.squeeze(0)  # 배치 차원 제거
        return outputs.cpu().numpy()

    def get_1d_pdm(
        self, result: np.ndarray, target_class_idx: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        2D 결과 맵을 세로축(Y) 기준으로 압축하여 1차원 프로파일을 생성합니다.
        이전 프레임과의 이동 평균(EMA)을 통해 값이 급격하게 튀는 것을 방지합니다.
        return: (원본 타겟 클래스 맵, 1D PDM 배열)
        """

        target_map = result[target_class_idx]

        # 확신도가 높은 값에 가중치를 부여하는 지수 함수 적용
        normalized_result = target_map * np.exp(-self._gain * (1 - target_map))

        data = np.sum(normalized_result, axis=0)

        # 프레임 간 스무딩 (부드러운 전환 효과)
        if self._last_results_data is not None:
            data = data * self._gamma + (1 - self._gamma) * self._last_results_data
        self._last_results_data = data

        return target_map, data

    def apply_weights(self, data: list, weights: list) -> np.ndarray:
        """동적으로 계산된 구역의 최대값 데이터에 각각 커스텀 가중치를 곱해줍니다."""
        print(f"원본 데이터: {data}")
        print(f"적용할 가중치: {weights}")

        if len(data) != len(weights):
            raise ValueError(
                f"데이터 구역 수({len(data)})와 가중치 길이({len(weights)})가 동일해야 합니다."
            )

        return np.array(data) * np.array(weights)

    def post_process_results(
        self, results: np.ndarray, weights: list, target_class_idx: int
    ) -> Tuple[np.ndarray, list, np.ndarray, int, np.ndarray]:
        """
        전체 결과 맵(2D)에서 최종 메인 구역과 인접 구역을 동적으로 파악합니다.

        :return: (1차원 PDM 배열, 인접 구역 리스트, 가중치가 적용된 N구역 최댓값, 메인 타겟 인덱스, 원본 타겟 클래스 맵)
        """

        # 경계선 개수에서 1을 빼면 실제 구역(Column)의 개수가 됩니다.
        num_peaks = len(self._peak_boundaries) - 1
        target_map, one_d_pdm = self.get_1d_pdm(results, target_class_idx)

        # 가변 구역(긴빠이)의 최댓값 탐색 및 가중치 적용
        max_peak_data = self.find_top_peaks_ginppai(one_d_pdm)

        if len(max_peak_data) < len(weights):
            # Max peak가 Weights 보다 적으면, weights의 초과분은 무시하
            weights = weights[: len(max_peak_data)]

        weighted_peak_data = self.apply_weights(max_peak_data, weights)

        # 가장 값이 높은 구역을 최종 메인 타겟으로 선정
        top_peak_idx = int(np.argmax(weighted_peak_data))

        # 메인 타겟 바로 양옆의 인접 구역 인덱스 추출 (가변 길이 대응)
        res = [
            idx
            for idx in range(top_peak_idx - 1, top_peak_idx + 2)
            if 0 <= idx < num_peaks and idx != top_peak_idx
        ]

        return one_d_pdm, res, weighted_peak_data, top_peak_idx, target_map

    def find_top_peaks_ginppai(self, data_1d: np.ndarray) -> list:
        """
        1차원 데이터를 설정된 가변 구역(boundaries)으로 나누고 각 구역의 최댓값을 리스트로 반환합니다.
        """
        b = self._peak_boundaries
        peaks = []

        try:
            for i in range(len(b) - 1):
                # 각 구역별 데이터를 슬라이싱
                region_data = data_1d[b[i] : b[i + 1]]

                # 구역 크기가 0이거나 데이터가 비어있는 경우 방어 코드
                if len(region_data) == 0:
                    peaks.append(0.0)
                else:
                    peaks.append(float(np.max(region_data)))

        except IndexError as e:
            raise ValueError(
                f"[Error] peak_boundaries가 데이터의 가로 길이를 초과했거나 형태가 잘못되었습니다: {e}"
            )

        return peaks

    def find_top_peaks(
        self, data: np.ndarray, num_peaks=4, smooth_sigma=5, min_distance=10
    ) -> tuple:
        """
        (유틸 함수) 데이터에서 상위 주요 피크를 동적으로 찾는 견고한 로직.
        현재 파이프라인에서는 쓰이지 않으나 예비용으로 유지됨.
        """
        smoothed_data = gaussian_filter1d(data, sigma=smooth_sigma)
        peaks, _ = find_peaks(smoothed_data, distance=min_distance)
        top_peaks = sorted(peaks, key=lambda x: data[x], reverse=True)[:num_peaks]
        return top_peaks, data[top_peaks]
