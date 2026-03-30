import numpy as np
import onnxruntime as ort
from dataclasses import dataclass
from typing import List, Union


@dataclass
class PolicyAction:
    """정책 모델의 추론 결과를 담는 데이터 클래스"""

    action_type: int
    target_column: int  # 0, 1, 2, 3 (목표 열)
    raw_output: np.ndarray  # 클리핑(Clip) 되기 전의 원본 모델 출력값 (디버깅/로깅 용도)


class RLPolicyManager:
    """RL 정책 모델(ONNX)을 로드하고 관측 상태를 관리하며 행동을 추론하는 매니저 클래스"""

    def __init__(self, model_path: str):
        # 1. 모델 경로를 받아와서 GPU(또는 CPU)에 로드
        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if ort.get_device() == "GPU"
            else ["CPUExecutionProvider"]
        )

        try:
            self._session = ort.InferenceSession(model_path, providers=providers)
        except Exception as e:
            raise RuntimeError(
                f"[Error] ONNX 모델 로드 실패. 경로를 확인하세요: {model_path}\n상세: {e}"
            )

        # --- 관측(Observation) 상태 버퍼 초기화 ---
        self._column_distribution = None  # FCN -> float 배열
        self._front_object_distance = None  # 각 ROW 거리 (이산적)
        self._front_object = None  # 각 ROW 맨 앞에 있는 물체 클래스

        self._target_id = 0.0

        # 내부에서만 관리될 이전 스텝의 행동 값 (초기화)
        self._action_policy = 0.0
        self._action_column = 0.0

    def reset_states(self):
        """새로운 작업(Episode) 시작 시, 과거 행동 기록을 0으로 초기화합니다."""
        self._action_policy = 0.0
        self._action_column = 0.0

    # ==========================================================
    # 2. Getter & Setter (각 관측값을 독립적으로 관리)
    # Numpy 배열로 강제 변환하여 데이터 타입 불일치 에러를 방지합니다.
    # ==========================================================

    @property
    def column_distribution(self) -> np.ndarray:
        return self._column_distribution

    @column_distribution.setter
    def column_distribution(self, val: Union[List[float], np.ndarray]):
        self._column_distribution = np.array(val, dtype=np.float32)

    @property
    def front_object_distance(self) -> np.ndarray:
        return self._front_object_distance

    @front_object_distance.setter
    def front_object_distance(self, val: Union[List[float], np.ndarray]):
        self._front_object_distance = np.array(val, dtype=np.float32)

    @property
    def front_object(self) -> np.ndarray:
        return self._front_object

    @front_object.setter
    def front_object(self, val: Union[List[int], np.ndarray]):
        self._front_object = np.array(val, dtype=np.float32)

    @property
    def target_id(self) -> float:
        return self._target_id

    @target_id.setter
    def target_id(self, val: float):
        self._target_id = float(val)

    # 외부에서 현재 들고 있는 값을 읽을 수만 있도록 Getter(Property)만 남겨둡니다.
    @property
    def action_policy(self) -> float:
        return self._action_policy

    @property
    def action_column(self) -> float:
        return self._action_column

    # ==========================================================
    # 3. 모델 추론 요청 함수
    # ==========================================================
    def request_action(self) -> PolicyAction:
        """
        현재 세팅된 관측값들을 모아 모델에 전달하고 다음 행동을 반환합니다.
        """
        # (1) 데이터 Concat (1D 벡터 생성) - 이 때 직전 스텝의 action_policy, action_column이 들어감
        obs = np.concatenate(
            [
                self._column_distribution,  # [4]
                self._front_object_distance,  # [4]
                self._front_object,  # [4]
                [self._target_id],  # [1]
                [self._action_policy],  # [1]
                [self._action_column],  # [1]
            ],
            dtype=np.float32,
        )

        # (2) 배치 차원 추가 (N, 15)
        obs_batch = np.expand_dims(obs, axis=0)

        # (3) 모델 추론 (ONNX Runtime)
        action_output = self._session.run(["actions"], {"obs": obs_batch})[0]

        # (4) 결과 파싱 및 클리핑(안전장치)
        raw_policy = action_output[0, 0]
        raw_column = action_output[0, 1]

        clipped_policy = int(np.clip(raw_policy, 0, 2))
        clipped_column = int(np.clip(raw_column, 0, 4))

        # (5) ★ 핵심: 다음 스텝(t+1)의 입력으로 사용하기 위해 내부 상태를 자동 갱신 ★
        self._action_policy = float(clipped_policy)
        self._action_column = float(clipped_column)

        # (6) Dataclass로 깔끔하게 포장하여 리턴
        return PolicyAction(
            action_type=clipped_policy,
            target_column=clipped_column,
            raw_output=action_output[0],
        )
