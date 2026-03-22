# ROS2
import rclpy
from rclpy.node import Node
from rclpy.time import Time
from rclpy.duration import Duration
from rclpy.qos import QoSProfile, qos_profile_system_default

# Message
from std_msgs.msg import *
from geometry_msgs.msg import *
from sensor_msgs.msg import *
from nav_msgs.msg import *
from visualization_msgs.msg import *
from builtin_interfaces.msg import Duration as BuiltinDuration

# TF
from tf2_ros import *

# Python
import sys
import os
import copy
import numpy as np
from enum import Enum
import time
import threading
from typing import List, Dict, Optional, Tuple
from abc import ABC, abstractmethod

# Custom
from base_package.header import PointCloudTransformer
from base_package.image_manager import ImageManager
from custom_msgs.msg import BoundingBox, BoundingBoxMultiArray


import numpy as np


class PrehensileDecisionNetwork:
    """
    [Prehensile Decision Network]
    타겟 물체를 충돌 없이 잡을 수 있는지(Prehensile) 판별하는 네트워크 (구현 필요).
    입력: object-prehensile state (n, 4) -> [x, y, z, f] / f ∈ {1, 0, -1}
    출력이 0.5보다 크면 잡을 수 있는 것으로 간주
    """

    def __init__(self, decision_threshold: float = 0.5):
        # 내부 상태는 모두 protected(_)로 선언
        self._decision_threshold = decision_threshold
        self._is_model_loaded = False  # 실제 모델 로드 여부 플래그

    @property
    def decision_threshold(self) -> float:
        return self._decision_threshold

    @decision_threshold.setter
    def decision_threshold(self, value: float):
        self._decision_threshold = value

    @property
    def is_model_loaded(self) -> bool:
        return self._is_model_loaded

    def predict(self, state_p: np.ndarray) -> float:
        """
        :param state_p: (M, 4) 형태의 Numpy 배열. [X, Y, Z, ID Feature]
                        ID Feature는 타겟(1), 평가대상 객체(0), 그 외(-1)로 구성
        :return: 0.0 ~ 1.0 사이의 확률(가치) 값
        """
        # TODO: 실제 ONNX/PyTorch 모델 추론 코드로 교체
        # 현재는 심플하게 0.0 ~ 1.0 사이의 랜덤 값을 반환
        random_prob = np.random.uniform(0.0, 1.0)
        return float(random_prob)

    def is_prehensile(self, state_p: np.ndarray) -> bool:
        """
        예측된 확률값이 임계치(0.5)보다 큰지 Boolean으로 직관적으로 리턴합니다[cite: 139].
        """
        return self.predict(state_p) > self._decision_threshold


class PriorityDecisionNetwork:
    """
    [Priority Decision Network]
    트리 탐색 시 어떤 물체를 먼저 치우는 것이 효율적인지 가치(Value)를 예측하는 네트워크
    입력: object-priority state (n, 4) -> [x, y, z, f] / f ∈ {1, 0, -1}
    출력: 행동의 기대 누적 보상 (가치)
    """

    def __init__(self, prune_threshold: float = 0.1):
        self._prune_threshold = prune_threshold  # 논문의 delta(\delta) 값 [cite: 206]
        self._is_model_loaded = False

    @property
    def prune_threshold(self) -> float:
        return self._prune_threshold

    @prune_threshold.setter
    def prune_threshold(self, value: float):
        self._prune_threshold = value

    @property
    def is_model_loaded(self) -> bool:
        return self._is_model_loaded

    def predict(self, state_q: np.ndarray) -> float:
        """
        :param state_q: (M, 4) 형태의 Numpy 배열. [X, Y, Z, ID Feature]
                        ID Feature는 타겟(1), 치울 후보 객체(0), 그 외(-1)
        :return: 행동의 기대 누적 보상 (가치). MCTS의 UCT 계산 등에 사용
        """
        # TODO: 실제 모델 추론 코드로 교체
        # 논문에서 Node-action Value는 [0, 1] 범위로 클리핑 또는 정규화하여 사용하므로[cite: 194],
        # 0.0 ~ 1.0 범위의 랜덤 Float 값을 반환하도록 합니다.
        random_value = np.random.uniform(0.0, 1.0)
        return float(random_value)

    def should_prune(self, state_q: np.ndarray) -> bool:
        """
        예측된 가치(Value)가 임계치(\delta)보다 낮으면 Non-prehensile로 간주하여 가지치기(Prune) 대상으로 판별합니다 [cite: 204-207].
        """
        return self.predict(state_q) < self._prune_threshold


class ObservationManager:
    """
    센서 및 노드들로부터 들어오는 관측 데이터를 모으고,
    MCTS에 필요한 3D 상태(Segmentation)로 가공하는 클래스입니다.
    """

    def __init__(self, node: Node, *args, **kwargs):
        # 1. Raw Data Buffers
        self._node: Node = node

        # >>>>> Subscriptions <<<<<
        self._point_cloud_sub = self._node.create_subscription(
            PointCloud2,
            "/camera/camera1/depth/color/points",  # 실제 사용하는 뎁스 카메라 PC 토픽명으로 변경
            self._pc_callback,
            qos_profile=qos_profile_system_default,
        )
        self._depth_image_manager = ImageManager(
            self._node,
            subscribed_topics=[
                {
                    "topic_name": "/camera/camera1/depth/image_rect_raw",
                    "callback": self._depth_callback,
                },
            ],
            published_topics=[],
            *args,
            **kwargs,
        )
        self._volume_marker_sub = self._node.create_subscription(
            MarkerArray,
            "/grid_markers",
            self._volume_marker_callback,
            qos_profile=qos_profile_system_default,
        )
        self._segmented_bbox_sub = self._node.create_subscription(
            BoundingBoxMultiArray,
            "real_time_segmentation_node" + "/segmented_bbox",
            self._segmented_bbox_callback,
            qos_profile=qos_profile_system_default,
        )

        # >>>>> ROS2 Messages <<<<<

        self._point_cloud_msg: Optional[PointCloud2] = None
        self._depth_image_msg: Optional[Image] = None
        self._volume_marker_array_msg: Optional[MarkerArray] = None
        self._segmentation_msg: Optional[BoundingBoxMultiArray] = None

        # >>>>> Processed Data <<<<<
        self._raw_pointcloud: Optional[np.ndarray] = None  # 호출해야 업데이트됨
        self._depth_image: Optional[np.ndarray] = None  # 호출해야 업데이트됨
        self._detected_objects: List[dict] = []  # 자동으로 업데이트 됨
        self._grid_volumes: Dict[str, dict] = {}  # 자동으로 업데이트 됨

        # >>>>> System Variables <<<<<

        # closest_object_node.py 기준 컬럼 경계선
        self._boundary = [170, 300, 460]

        # col_idx를 key로, 해당 컬럼 내 객체 ID들을 거리가 가까운 순으로 정렬한 리스트
        self._column_sorted_objects: Dict[int, List[int]] = {}

        # row_idx를 key로, 해당 행 내 객체 ID들을 거리가 가까운 순으로 정렬한 리스트
        self._row_sorted_objects: Dict[str, List[int]] = {}

        # 객체 ID를 key로, 세그멘테이션된 PointCloud 배열 저장
        self._segmented_pointclouds: Dict[int, np.ndarray] = {}

    @property
    def boundary(self) -> List[int]:
        return self._boundary

    @boundary.setter
    def boundary(self, value: List[int]):
        self._boundary = value

    @property
    def column_sorted_objects(self) -> Dict[int, List[int]]:
        """
        return: {
            0: [obj_id1, obj_id2, ...],  # 컬럼 0에서 가장 가까운 객체 ID부터 순서대로
            1: [obj_id3, obj_id4, ...],  # 컬럼 1에서 가장 가까운 객체 ID부터 순서대로
            2: [obj_id5, obj_id6, ...],  # 컬럼 2에서 가장 가까운 객체 ID부터 순서대로
            3: [obj_id7, obj_id8, ...],  # 컬럼 3에서 가장 가까운 객체 ID부터 순서대로
        }
        """

        return self._column_sorted_objects

    @property
    def row_sorted_objects(self) -> Dict[int, List[int]]:
        """
        return: {
            "A": [obj_id1, obj_id2, ...],  # Row 'A'에서 가장 가까운 객체 ID부터 순서대로
            "B": [obj_id3, obj_id4, ...],  # Row 'B'에서 가장 가까운 객체 ID부터 순서대로
            "C": [obj_id5, obj_id6, ...],  # Row 'C'에서 가장 가까운 객체 ID부터 순서대로
            "D": [obj_id7, obj_id8, ...],  # Row 'D'에서 가장 가까운 객체 ID부터 순서대로
        }
        """
        return self._row_sorted_objects

    # >>> Callbacks for ROS2 Subscriptions >>>

    def _pc_callback(self, msg: PointCloud2):
        """PointCloud2 메시지를 수신"""
        self._point_cloud_msg = msg

    def _depth_callback(self, msg: Image):
        """Depth 이미지 메시지를 수신"""
        self._depth_image_msg = msg

    def _volume_marker_callback(self, msg: MarkerArray):
        """Grid 마커 메시지를 수신"""
        self._volume_marker_array_msg = msg
        self._update_grid_volumes()

    def _segmented_bbox_callback(self, msg: BoundingBoxMultiArray):
        """객체 검출 결과 메시지를 수신"""
        self._segmentation_msg = msg
        self._update_segmentation()

    # <<< ROS2 Callbacks <<<

    # >>> Pose-Processing Methods >>>

    def _update_pointcloud(self):
        """함수가 호출될 때만, Numpy 배열로 변환하여 self._raw_pointcloud에 저장"""
        if self._point_cloud_msg is None:
            self._node.get_logger().warn(
                "아직 PointCloud2 메시지를 수신하지 못했습니다."
            )
            self._raw_pointcloud = None
            return

        pc = PointCloudTransformer.pointcloud2_to_numpy(
            msg=self._point_cloud_msg, rgb=False
        )
        self._raw_pointcloud = pc

    def _update_depth(self):
        """함수가 호출될 때만, Numpy 배열로 변환하여 self._depth_image에 저장"""
        if self._depth_image_msg is None:
            self._node.get_logger().warn(
                "아직 Depth 이미지 메시지를 수신하지 못했습니다."
            )
            self._depth_image = None
            return None

        np_depth = self._depth_image_manager.decode_message(
            image_msg=self._depth_image_msg, desired_encoding="16UC1"
        )
        np_depth = self._depth_image_manager.crop_image(
            img=np_depth
        )  # 크롭 로직 내재화
        self._depth_image = np_depth

    def _update_grid_volumes(self):
        """
        MarkerArray 메시지를 기반으로, Grid의 각 Cell에 해당하는 3D 부피 정보를 self._grid_volumes에 저장
        {'A0': {'center': [...], 'scale': [...]}, ...}
        """
        marker_info = {}

        if self._volume_marker_array_msg is None:
            self._node.get_logger().warn("아직 Grid 마커 메시지를 수신하지 못했습니다.")
            self._grid_volumes = marker_info
            return None

        def decode_marker_id(marker_id: str) -> Tuple[str, int]:
            """인코딩된 마커 ID를 ROW, COL로 분리함"""
            # ((ord(self._row_id) - 64) * 10) + self._col_id + 2000
            try:
                numeric_id = int(marker_id)
                row_num = (numeric_id - 2000) // 10
                col_num = (numeric_id - 2000) % 10
                row_char = chr(row_num + 64)  # 1 -> 'A', 2 -> 'B', ...
                return row_char, col_num
            except Exception as e:
                self._node.get_logger().error(
                    f"마커 ID 디코딩 실패: {marker_id}, 오류: {e}"
                )
                return None, None

        for marker in self._volume_marker_array_msg.markers:
            marker: Marker

            if marker.ns == "grid_volume":
                row, col = decode_marker_id(marker.id)
                position: Point = marker.pose.position
                scale: Vector3 = marker.scale

                marker_info[f"{row}{col}"] = {
                    "center": [position.x, position.y, position.z],
                    "scale": [scale.x, scale.y, scale.z],
                }

        self._grid_volumes = marker_info

    def _update_segmentation(self):
        """
        객체 검출 결과 메시지를 수신, 파싱, 저장
        [{'id': int, 'mask': np.ndarray}, ...]
        """
        if self._segmentation_msg is None:
            self._node.get_logger().warn(
                "아직 객체 검출 결과 메시지를 수신하지 못했습니다."
            )
            self._detected_objects = []
            return

        detected_objects = []
        for bbox in self._segmentation_msg.data:
            bbox: BoundingBox

            """
            int32 id
            string cls
            float32 conf
            float32[] bbox
            int32 mask_row
            int32 mask_col
            int32[] mask_data
            """

            data = {
                "id": bbox.id,
                "mask": (
                    np.array(bbox.mask_data).reshape((bbox.mask_row, bbox.mask_col))
                    if bbox.mask_row > 0 and bbox.mask_col > 0
                    else None
                ),
            }

            detected_objects.append(data)

        self._detected_objects = detected_objects

    # <<< Pose-Processing Methods <<<

    def _remove_outliers(self, depth_array: np.ndarray) -> np.ndarray:
        """closest_object_node.py의 아웃라이어 제거 로직 차용"""
        return depth_array[depth_array < 1240]

    def process_column_objects(self):
        """
        Depth 이미지와 객체 검출 결과를 기반으로, 각 컬럼별로 가장 가까운 객체 ID를 추출하여
        self._column_sorted_objects에 저장합니다.
        """

        self._update_depth()  # 최신 Depth 이미지 업데이트

        num_cols = len(self._boundary) + 1
        columns_data = {i: [] for i in range(num_cols)}

        if self._depth_image is None or not self._detected_objects:
            self._column_sorted_objects = {i: [] for i in range(num_cols)}
            return

        # 원본 코드의 보정 로직 (좌우 패딩/크롭 등 형태를 맞추기 위함)
        depth_img = self._depth_image
        if depth_img.shape[1] > 40:
            zero_pixel = np.zeros((depth_img.shape[0], 40), dtype=depth_img.dtype)
            depth_img = np.hstack([depth_img, zero_pixel])[:, 40:]

        for obj in self._detected_objects:
            mask = obj["mask"].astype(bool)
            mask_depth = depth_img[mask]
            mask_depth = mask_depth[mask_depth > 0]
            mask_depth = self._remove_outliers(mask_depth)

            if len(mask_depth) == 0:
                continue

            mean_distance = np.mean(mask_depth)
            mask_x = np.where(mask)[1]

            if len(mask_x) == 0:
                continue

            center_x = np.mean(mask_x)

            # X 픽셀 기준 컬럼 인덱스 찾기
            col_idx = num_cols - 1
            for i, b_val in enumerate(self._boundary):
                if center_x < b_val:
                    col_idx = i
                    break

            columns_data[col_idx].append({"id": obj["id"], "distance": mean_distance})

        # 컬럼별로 거리(distance) 기준 오름차순 정렬 후 ID만 추출
        self._column_sorted_objects.clear()
        for col_idx, obj_list in columns_data.items():
            obj_list.sort(key=lambda x: x["distance"])
            self._column_sorted_objects[col_idx] = [item["id"] for item in obj_list]

    def process_row_objects(self):
        """
        Depth 이미지와 객체 검출 결과를 기반으로, 각 '행(Row)'별로 가장 가까운 객체 ID를 추출하여
        self._row_sorted_objects에 저장합니다. (process_column_objects의 Y축 버전)
        """
        self._update_depth()  # 최신 Depth 이미지 업데이트

        # Y축 경계값이 필요하므로, 없으면 기본값 설정 (클래스 __init__에 추가하는 것을 권장)
        if not hasattr(self, "row_boundary"):
            # 예: 세로 해상도가 480 픽셀일 때 4등분하는 예시 경계값
            self.row_boundary = [120, 240, 360]

        num_rows = len(self.row_boundary) + 1
        rows_data = {i: [] for i in range(num_rows)}

        if self._depth_image is None or not self._detected_objects:
            self._row_sorted_objects = {i: [] for i in range(num_rows)}
            return

        # 원본 코드의 보정 로직 (좌우 패딩/크롭 등 형태를 맞추기 위함)
        depth_img = self._depth_image
        if depth_img.shape[1] > 40:
            zero_pixel = np.zeros((depth_img.shape[0], 40), dtype=depth_img.dtype)
            depth_img = np.hstack([depth_img, zero_pixel])[:, 40:]

        for obj in self._detected_objects:
            mask: np.ndarray = obj["mask"].astype(bool)

            # 크기 불일치 방어 로직 (Numpy 에러 방지용)
            if depth_img.shape != mask.shape:
                continue

            mask_depth = depth_img[mask]
            mask_depth = mask_depth[mask_depth > 0]
            mask_depth = self._remove_outliers(mask_depth)

            if len(mask_depth) == 0:
                continue

            mean_distance = np.mean(mask_depth)

            # Y 픽셀 기준 (mask의 0번째 인덱스가 Y축(행)을 의미함)
            mask_y = np.where(mask)[0]

            if len(mask_y) == 0:
                continue

            center_y = np.mean(mask_y)

            # Y 픽셀 기준 로우(Row) 인덱스 찾기
            row_idx = num_rows - 1
            for i, b_val in enumerate(self.row_boundary):
                if center_y < b_val:
                    row_idx = i
                    break

            rows_data[row_idx].append({"id": obj["id"], "distance": mean_distance})

        # 로우별로 거리(distance) 기준 오름차순 정렬 후 ID만 추출
        if not hasattr(self, "row_sorted_objects"):
            self._row_sorted_objects = {}

        self._row_sorted_objects.clear()
        for row_idx, obj_list in rows_data.items():
            obj_list.sort(key=lambda x: x["distance"])
            self._row_sorted_objects[row_idx] = [item["id"] for item in obj_list]

    def execute_3d_segmentation(self) -> bool:
        """
        정렬된 컬럼 정보와 3D Grid 마커(Volume) 정보를 결합하여 PointCloud를 분할합니다.
        측면(Lateral) 접근 환경의 특성상[cite: 6], 거리가 가까울수록
        앞쪽 Row(예: 'A', 'B', 'C' 순)에 위치한다는 휴리스틱을 적용합니다.
        """
        # 최신 PointCloud 데이터를 Numpy 배열로 업데이트 (호출 시점에 변환 수행)
        self._update_pointcloud()

        if self._raw_pointcloud is None or not self._grid_volumes:
            return False

        self._segmented_pointclouds.clear()
        row_identifiers = [
            "A",
            "B",
            "C",
            "D",
        ]  # GridManager의 row_id 정책에 맞게 확장 가능

        # 각 컬럼별로 앞(가장 가까운)에서부터 차례대로 Row ID를 부여하여 3D 부피와 매칭
        for col_idx, sorted_ids in self._column_sorted_objects.items():
            for depth_rank, obj_id in enumerate(sorted_ids):
                if depth_rank >= len(row_identifiers):
                    break  # 미리 정의된 Row 개수를 초과하면 무시

                # 예: 컬럼 0에서 가장 가까운 객체 -> 'A0', 두 번째 -> 'B0'
                row_id = row_identifiers[depth_rank]
                target_marker_id = f"{row_id}{col_idx}"

                if target_marker_id in self._grid_volumes:
                    vol_info = self._grid_volumes[target_marker_id]
                    center = vol_info["center"]
                    scale = vol_info["scale"]

                    x_min, x_max = center[0] - scale[0] / 2, center[0] + scale[0] / 2
                    y_min, y_max = center[1] - scale[1] / 2, center[1] + scale[1] / 2
                    z_min, z_max = center[2] - scale[2] / 2, center[2] + scale[2] / 2

                    pts = self._raw_pointcloud
                    mask = (
                        (pts[:, 0] >= x_min)
                        & (pts[:, 0] <= x_max)
                        & (pts[:, 1] >= y_min)
                        & (pts[:, 1] <= y_max)
                        & (pts[:, 2] >= z_min)
                        & (pts[:, 2] <= z_max)
                    )

                    self._segmented_pointclouds[obj_id] = pts[mask]

        return len(self._segmented_pointclouds) > 0

    def reconstruct_full_pointcloud(
        self, target_id: str, select_id: str = None, present_objects: List[str] = None
    ) -> Optional[np.ndarray]:
        """
        분할된 PointCloud 조각들을 모두 합쳐서 원본과 동일한 형태로 재구성합니다.
        present_objects가 주어지면, 해당 리스트에 있는 객체들만 조립하여 '가상의 상태'를 만듭니다.
        """
        if not self._segmented_pointclouds:
            return None

        # MCTS 탐색 중이 아니라, 최초 1회 실행(초기 상태 세팅)일 때만 업데이트 수행
        if present_objects is None:
            self.process_column_objects()
            self.execute_3d_segmentation()
            present_objects = [str(k) for k in self._segmented_pointclouds.keys()]

        all_points = []
        for obj_id, points in self._segmented_pointclouds.items():
            # ★ 핵심: MCTS 트리 상에서 이미 치워진 물체라면 합치지 않고 패스!
            if str(obj_id) not in present_objects:
                continue

            if points is None or len(points) == 0:
                continue

            if str(obj_id) == str(target_id):
                id_feature = 1
            elif str(obj_id) == str(select_id):
                id_feature = 0
            else:
                id_feature = -1

            id_column = np.full((points.shape[0], 1), id_feature)
            points_with_id = np.hstack([points, id_column])
            all_points.append(points_with_id)

        return np.vstack(all_points) if all_points else None


class MCTSNode:
    """
    GP3 MCTS 트리의 각 상태(State)를 나타내는 노드 클래스입니다[cite: 179].
    """

    def __init__(self, state_objects: List[str], parent=None, action_taken: str = None):
        self._state_objects = state_objects  # 현재 상태에 남아있는 객체 ID 리스트
        self._parent: "MCTSNode" = parent  # 부모 노드
        self._action_taken: str = (
            action_taken  # 부모 노드에서 이 노드로 오기 위해 치운 객체 ID
        )

        self._children: Dict[str, "MCTSNode"] = (
            {}
        )  # Action(치운 객체 ID)을 Key로 가지는 자식 노드들
        self._untried_actions: List[str] = (
            []
        )  # 아직 탐색하지 않은 유효한 행동(객체 ID) 리스트

        self._visits: int = 0  # N(X): 노드 방문 횟수 [cite: 194]
        self._value: float = 0.0  # F(X): 노드 가치 [cite: 194]
        self._action_values: Dict[str, float] = (
            {}
        )  # G(X, a): 노드-행동 가치 [cite: 194]

        self._is_terminal: bool = False
        self._is_success: bool = False

    # >>> Getter / Setter >>>
    @property
    def state_objects(self) -> List[str]:
        return self._state_objects

    @property
    def parent(self) -> "MCTSNode":
        return self._parent

    @property
    def action_taken(self) -> str:
        return self._action_taken

    @property
    def children(self) -> Dict[str, "MCTSNode"]:
        return self._children

    @property
    def untried_actions(self) -> List[str]:
        return self._untried_actions

    @untried_actions.setter
    def untried_actions(self, actions: List[str]):
        self._untried_actions = actions

    @property
    def visits(self) -> int:
        return self._visits

    @property
    def value(self) -> float:
        return self._value

    @value.setter
    def value(self, val: float):
        self._value = val

    @property
    def action_values(self) -> Dict[str, float]:
        return self._action_values

    @property
    def is_terminal(self) -> bool:
        return self._is_terminal

    @is_terminal.setter
    def is_terminal(self, val: bool):
        self._is_terminal = val

    @property
    def is_success(self) -> bool:
        return self._is_success

    @is_success.setter
    def is_success(self, val: bool):
        self._is_success = val

    # <<< Getter / Setter <<<

    def add_child(self, action: str, child_node: "MCTSNode"):
        self._children[action] = child_node
        self._action_values[action] = 0.0

    def increment_visits(self):
        self._visits += 1

    def is_fully_expanded(self) -> bool:
        return len(self._untried_actions) == 0


class MCTSManager:
    """
    [개선 포인트 2]
    Node 계층과의 강한 결합을 끊고, 순수 관측 처리 및 알고리즘 수행을 담당합니다.
    """

    def __init__(self, node: Node):
        self._node: Node = node

        self._target_id: str = None  # MCTS 탐색 시 최종 타겟 객체 ID

        # GP3 하이퍼파라미터
        self._exploration_constant: float = 1.414  # c 값 [cite: 201]
        self._max_iterations: int = 100  # MCTS 최대 반복 횟수
        self._max_depth: int = 10  # 최대 탐색 깊이 (안전장치)

        self.observer = ObservationManager(node=self._node)
        self.prehensile_net = PrehensileDecisionNetwork()
        self.priority_net = PriorityDecisionNetwork()

    # >>> Getter / Setter >>>
    @property
    def target_id(self) -> str:
        return self._target_id

    @target_id.setter
    def target_id(self, val: str):
        self._target_id = val

    # <<< Getter / Setter <<<

    def _build_hypothetical_pc(
        self, present_objects: List[str], target: str, select: str = None
    ) -> bool:
        """
        MCTS 탐색 전 관측 데이터를 기반으로 현재 상태를 세팅합니다.

        """
        observation = self.observer.reconstruct_full_pointcloud(
            target_id=target, select_id=select, present_objects=present_objects
        )
        return observation

    def run_mcts(self) -> List[str]:
        """
        MCTS 알고리즘을 실행하여 최적의 객체 제거 순서(Action Sequence)를 반환합니다.
        """
        # 1. 초기 상태 세팅
        self.observer.process_column_objects()
        self.observer.execute_3d_segmentation()

        initial_objects = [str(k) for k in self.observer._segmented_pointclouds.keys()]
        if self._target_id not in initial_objects:
            self._node.get_logger().error("Target object가 시야에 없습니다.")
            return []

        root_node = MCTSNode(state_objects=initial_objects)
        self._init_node_actions(root_node)

        # 2. 4단계 사이클 반복 [cite: 89]
        for _ in range(self._max_iterations):
            # Step 1: Selection
            node = self._selection(root_node)

            # Step 2: Expansion
            if not node.is_terminal and not node.is_fully_expanded():
                node = self._expansion(node)

            # Step 3: Simulation
            reward = self._simulation(node)

            # Step 4: Backpropagation
            self._backpropagation(node, reward)

        # 3. 최적의 경로 추출 (가장 방문 횟수가 많은 자식 선택)
        return self._extract_best_sequence(root_node)

    def _init_node_actions(self, node: MCTSNode):
        """노드에서 수행 가능한 행동(치울 수 있는 객체)들을 Priority Network로 필터링하여 초기화합니다."""
        # 1. 터미널 조건 검사: 타겟이 지금 바로 잡히는가? (Success) [cite: 181]
        pc_target = self._build_hypothetical_pc(
            node.state_objects, target=self._target_id
        )
        if pc_target is not None and self.prehensile_net.is_prehensile(pc_target):
            node.is_terminal = True
            node.is_success = True
            node.value = 1.0
            return

        # 2. 유효한 행동(치울 객체) 추출
        valid_actions = []
        for obj_id in node.state_objects:
            if obj_id == self._target_id:
                continue

            # 우선순위 네트워크를 통해 Pruning (가치가 delta 이하인 객체는 가지치기) [cite: 204-207]
            pc_priority = self._build_hypothetical_pc(
                node.state_objects, target=self._target_id, select=obj_id
            )
            if pc_priority is not None and not self.priority_net.should_prune(
                pc_priority
            ):
                valid_actions.append(obj_id)

        if not valid_actions:
            node.is_terminal = True  # 뺄 수 있는게 없으면 실패(Dead-end)
            node.is_success = False
            node.value = 0.0

        node.untried_actions = valid_actions

    def _selection(self, node: MCTSNode) -> MCTSNode:
        """UCT 공식을 사용하여 자식 노드를 선택합니다 [cite: 200-203]."""
        current = node
        while not current.is_terminal and current.is_fully_expanded():
            best_action = None
            best_uct = -float("inf")

            for action, child in current.children.items():
                if child.visits == 0:
                    uct = float("inf")
                else:
                    # UCT 공식: G(X, a) + c * sqrt(2 * ln(N(X)) / N(child))
                    g_val = current.action_values[action]
                    exploration = self._exploration_constant * math.sqrt(
                        2 * math.log(current.visits) / child.visits
                    )
                    uct = g_val + exploration

                if uct > best_uct:
                    best_uct = uct
                    best_action = action

            current = current.children[best_action]
        return current

    def _expansion(self, node: MCTSNode) -> MCTSNode:
        """아직 시도하지 않은 행동 중 하나를 골라 자식 노드를 생성합니다 [cite: 208-209]."""
        action = node.untried_actions.pop(0)  # 단순 순차 추출. 휴리스틱으로 정렬 가능

        # 행동 적용: 선택된 객체 제거 [cite: 193]
        new_state_objects = copy.deepcopy(node.state_objects)
        new_state_objects.remove(action)

        child_node = MCTSNode(
            state_objects=new_state_objects, parent=node, action_taken=action
        )

        # 자식 노드가 유효한지(치우려는 객체를 현재 잡을 수 있는지) 판별 [cite: 182]
        pc_action_obj = self._build_hypothetical_pc(node.state_objects, target=action)
        if pc_action_obj is None or not self.prehensile_net.is_prehensile(
            pc_action_obj
        ):
            child_node.is_terminal = True
            child_node.is_success = False
            child_node.value = 0.0
        else:
            self._init_node_actions(child_node)

        node.add_child(action, child_node)
        return child_node

    def _simulation(self, node: MCTSNode) -> float:
        """Priority Network를 이용한 탐욕(Greedy) 롤아웃을 수행합니다 [cite: 210-215]."""
        current_objects = copy.deepcopy(node.state_objects)
        depth = 0

        # 이미 터미널 노드라면 즉시 반환
        if node.is_terminal:
            return node.value

        while depth < self._max_depth:
            # 타겟이 잡히는지 확인
            pc_target = self._build_hypothetical_pc(
                current_objects, target=self._target_id
            )
            if pc_target is not None and self.prehensile_net.is_prehensile(pc_target):
                return 1.0  # 성공

            best_action = None
            best_val = -float("inf")

            # Priority Network로 최적의 다음 객체 탐색
            for obj_id in current_objects:
                if obj_id == self._target_id:
                    continue
                pc_priority = self._build_hypothetical_pc(
                    current_objects, target=self._target_id, select=obj_id
                )
                if pc_priority is None:
                    continue

                val = self.priority_net.predict(pc_priority)
                if val > best_val:
                    best_val = val
                    best_action = obj_id

            if best_action is None:
                return 0.0  # 더 이상 치울 수 없음 (실패)

            # 치울 객체가 Prehensile 한지 검증
            pc_action_obj = self._build_hypothetical_pc(
                current_objects, target=best_action
            )
            if pc_action_obj is None or not self.prehensile_net.is_prehensile(
                pc_action_obj
            ):
                return 0.0  # 치우려는 물체를 잡을 수 없음 (실패)

            # 객체 제거 후 다음 깊이로 이동
            current_objects.remove(best_action)
            depth += 1

        return 0.0  # 최대 깊이 초과 (실패)

    def _backpropagation(self, node: MCTSNode, reward: float):
        """시뮬레이션 결과를 바탕으로 부모 노드들의 가치를 업데이트합니다 [cite: 216-217]."""
        current = node
        current_reward = reward

        while current is not None:
            current.increment_visits()

            if current.parent is not None:
                action = current.action_taken
                # G(X, a) 업데이트: max(-0.1 + F(X_child), 0) [cite: 196]
                current.parent.action_values[action] = max(-0.1 + current_reward, 0.0)

                # F(X) 업데이트: max(G(X, a)) [cite: 196]
                current.parent.value = (
                    max(current.parent.action_values.values())
                    if current.parent.action_values
                    else 0.0
                )

            # 논문에서는 성공 시 1.0에서 스텝마다 0.1씩 감소시킨 값을 뒤로 전파합니다[cite: 214].
            if current_reward > 0.0:
                current_reward = max(current_reward - 0.1, 0.0)

            current = current.parent

    def _extract_best_sequence(self, root: MCTSNode) -> List[str]:
        """탐색 완료 후 가장 최적의 시퀀스(방문 횟수가 가장 많은 자식들의 경로)를 추출합니다."""
        sequence = []
        current = root
        while current.children:
            # 방문 횟수가 가장 많은 자식 선택 (Robust Child)
            best_action = max(
                current.children.keys(), key=lambda a: current.children[a].visits
            )
            sequence.append(best_action)
            current = current.children[best_action]

            # 만약 타겟을 집을 수 있는 노드에 도달했다면 종료
            if current.is_success:
                break

        return sequence
