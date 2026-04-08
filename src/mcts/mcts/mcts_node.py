# ROS2
import rclpy
from rclpy.node import Node
from rclpy.time import Time
from rclpy.duration import Duration
from rclpy.callback_groups import ReentrantCallbackGroup, MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.qos import QoSProfile, qos_profile_system_default

# Message
from std_msgs.msg import *
from geometry_msgs.msg import *
from sensor_msgs.msg import *
from nav_msgs.msg import *
from visualization_msgs.msg import *
from builtin_interfaces.msg import Duration as BuiltinDuration
from custom_msgs.srv import GetPolicyAction

# TF
from tf2_ros import *

# Python
import sys
import os
import copy
import random
import math
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
from mcts.mcts_manager import MCTS, GridState


class ObservationManager:
    """
    센서 및 노드들로부터 들어오는 관측 데이터를 모으고,
    MCTS에 필요한 3D 상태(Segmentation)로 가공하는 클래스입니다.
    """

    def __init__(self, node: Node, *args, **kwargs):
        # 1. Raw Data Buffers
        self._node: Node = node

        # >>>>> Subscriptions <<<<<
        # 가장 가까운 이미지를 파악하기 위한 Depth 이미지 구독
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
        # 필드 내 존재하는 물체의 위치를 파악하기 위한 MarkerArray 구독
        self._volume_marker_sub = self._node.create_subscription(
            MarkerArray,
            "/grid_markers",
            self._volume_marker_callback,
            qos_profile=qos_profile_system_default,
        )
        # 위에서 파악한 물체 위치에 라벨을 붙이기 위한 세그멘테이션 결과 구독
        self._segmented_bbox_sub = self._node.create_subscription(
            BoundingBoxMultiArray,
            "real_time_segmentation_node" + "/segmented_bbox",
            self._segmented_bbox_callback,
            qos_profile=qos_profile_system_default,
        )

        # >>>>> ROS2 Messages <<<<<

        self._depth_image_msg: Optional[Image] = None
        self._volume_marker_array_msg: Optional[MarkerArray] = None
        self._segmentation_msg: Optional[BoundingBoxMultiArray] = None

        # >>>>> Processed Data <<<<<
        self._depth_image: Optional[np.ndarray] = None  # 호출해야 업데이트됨
        self._detected_objects: List[dict] = []  # 자동으로 업데이트 됨
        self._grid_volumes: np.ndarray = None  # 자동으로 업데이트 됨

        # >>>>> System Variables <<<<<

        # closest_object_node.py 기준 컬럼 경계선
        """
        [0, 128, 256, 384, 512, 640] 
        [0, 170, 300, 460, 640]
        """
        self._boundary = [0, 170, 300, 460, 640]
        # self._boundary = [0, 170, 270, 384, 480, 640]

        # col_idx를 key로, 해당 컬럼 내 객체 ID들을 거리가 가까운 순으로 정렬한 리스트
        self._column_sorted_objects: Dict[int, List[int]] = {}

    # >>> Getter / Setter >>>

    @property
    def boundary(self) -> List[int]:
        return self._boundary

    @boundary.setter
    def boundary(self, value: List[int]):
        self._boundary = value

    # <<< Getter / Setter <<<

    @property
    def column_sorted_objects(self) -> Dict[int, List[int]]:
        # 컬럼별로 가장 가까운 객체 ID부터 순서대로 정렬된 딕셔너리 반환
        """
        return: {
            0: [obj_id1, obj_id2, ...],  # 컬럼 0에서 가장 가까운 객체 ID부터 순서대로
            1: [obj_id3, obj_id4, ...],  # 컬럼 1에서 가장 가까운 객체 ID부터 순서대로
            2: [obj_id5, obj_id6, ...],  # 컬럼 2에서 가장 가까운 객체 ID부터 순서대로
            3: [obj_id7, obj_id8, ...],  # 컬럼 3에서 가장 가까운 객체 ID부터 순서대로
        }
        """

        return self._column_sorted_objects

    # >>> Callbacks for ROS2 Subscriptions >>>

    def _depth_callback(self, msg: Image):
        """Depth 이미지 메시지를 수신"""
        self._depth_image_msg = msg
        self._update_depth()

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

        zero_pixel = np.zeros((480, 40), dtype=np.uint16)
        np_depth = np.hstack([np_depth, zero_pixel])[:, 40:]

        self._depth_image = np_depth

    def _update_grid_volumes(self):
        """
        MarkerArray 메시지를 기반으로, Grid의 각 Cell에 해당하는 3D 부피 정보를 self._grid_volumes에 저장
        {'A0': {'center': [...], 'scale': [...]}, ...}
        """

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

        gird_size = (
            (4, 5) if len(self._boundary) == 6 else (3, 4)
        )  # boundary 길이에 따라 그리드 크기 결정
        grid_matrix = np.zeros(gird_size)

        """
        예시 그리드
        [
            [0, 0, 0, 0, 0]
            [0, 0, 0, 0, 0]
            [0, 0, 0, 0, 0]
            [0, 0, 0, 0, 0]
        ] -> (4, 5) 크기의 그리드
        """

        if self._volume_marker_array_msg is None:
            self._node.get_logger().warn("아직 Grid 마커 메시지를 수신하지 못했습니다.")
            self._grid_volumes = grid_matrix
            return None

        for marker in self._volume_marker_array_msg.markers:
            marker: Marker

            if marker.ns == "grid_volume":
                
                row, col = decode_marker_id(marker.id)
                row_int = ord(row) - ord("A")  # 0~4
                col_int = int(col)  # 0~5

                # (4, 5) 크기의 그리드에서, 해당하는 인덱스 값을 1로 변환
                grid_matrix[row_int, col_int] = 1

        self._grid_volumes = grid_matrix

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
                "cls": str(bbox.cls),
                "conf": bbox.conf,
                "bbox": bbox.bbox,
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

    def _process_column_objects(self):
        """
        Depth 이미지와 객체 검출 결과를 기반으로, 각 컬럼별로 가장 가까운 객체 ID를 추출하여
        self._column_sorted_objects에 저장합니다.
        """

        num_cols = len(self._boundary) - 1  # 경계선 개수 - 1 = 컬럼 개수
        columns_data = {i: [] for i in range(num_cols)}

        # 원본 코드의 보정 로직 (좌우 패딩/크롭 등 형태를 맞추기 위함)
        depth_img = np.copy(self._depth_image)

        for obj in self._detected_objects:
            obj: Dict[str, np.ndarray]

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
            # self._boundary = [0, 170, 300, 460, 640] 기준으로 컬럼 경계선이 정의되어 있다고 가정
            col_idx = None
            for i in range(len(self._boundary) - 1):
                if self._boundary[i] <= center_x < self._boundary[i + 1]:
                    col_idx = i
                    break

            if col_idx is not None:
                columns_data[col_idx].append(
                    {
                        "id": obj["id"],
                        "distance": mean_distance,
                    }
                )

        self._column_sorted_objects = {
            col_idx: [obj["id"] for obj in sorted(objects, key=lambda x: x["distance"])]
            for col_idx, objects in columns_data.items()
        }
        print(f"컬럼별로 가장 가까운 객체 ID 리스트: {self._column_sorted_objects}")

    def get_observation(self, target_object_id: int) -> np.ndarray:
        self._process_column_objects()

        if self._grid_volumes is None or len(self._column_sorted_objects) == 0:
            self._node.get_logger().warn(
                "관측값을 구성하는 데 필요한 데이터가 아직 준비되지 않았습니다."
            )
            return None

        # 1. 원본 배열과 동일한 크기의 0(비어있음)으로 채워진 결과 배열 생성
        obs = np.zeros_like(self._grid_volumes, dtype=np.int32)
        _, cols = self._grid_volumes.shape

        for c in range(cols):
            # 해당 컬럼에서 물체가 차 있는(1인) 행의 인덱스를 가져옴
            # 행 인덱스는 0(가장 앞쪽)부터 시작하여 오름차순으로 정렬되어 있음
            occupied_rows = np.where(self._grid_volumes[:, c] == 1)[0]
            num_occupied = len(occupied_rows)

            if num_occupied == 0:
                continue

            # 해당 컬럼에 등록된 물체 리스트 가져오기
            col_objs = self._column_sorted_objects.get(c, [])
            num_objs = len(col_objs)

            if num_occupied == num_objs:
                # [케이스 1] 갯수가 일치할 경우: 순서대로 단순 매핑
                for r, obj_id in zip(occupied_rows, col_objs):
                    obs[r, c] = 2 if obj_id == target_object_id else 1
            else:
                # [케이스 2] 갯수가 다를 경우: 뒤에서부터 매핑하고 나머지는 -1 처리
                rev_occupied = occupied_rows[::-1]
                rev_objs = col_objs[::-1]

                target_in_col = target_object_id in col_objs
                target_assigned = False

                for i, r in enumerate(rev_occupied):
                    if i < num_objs:
                        # 뒤에서부터 물체 아이디를 매핑
                        obj_id = rev_objs[i]
                        if obj_id == target_object_id:
                            obs[r, c] = 2
                            target_assigned = True
                        else:
                            obs[r, c] = 1
                    else:
                        # 매핑할 물체가 부족하면 나머지는 알 수 없음(-1) 처리
                        obs[r, c] = -1

                # (2는 우선순위를 가짐) 조건 반영:
                # 센서/비전의 불일치로 타겟이 누락될 위기라도, 해당 컬럼에 타겟이 존재한다고
                # 인식되었다면 가장 앞쪽의 남아있는 칸을 2로 강제 덮어씌워 타겟 위치를 잃지 않도록 보장.
                if target_in_col and not target_assigned:
                    front_most_row = rev_occupied[-1]  # 남은 공간 중 가장 앞쪽(입구)
                    obs[front_most_row, c] = 2

        """
        0: 비어있음, 1: 일반 물체, 2: 타겟(목표), -1: 알 수 없음
        np.array(
            [
                [1, 1, 1, -1, 1],  # 가장 앞쪽 (입구)
                [1, 1, -1, 1, 1],
                [1, 1, -1, 1, 1], 
                [1, 2, 1, 1, 1],  # 가장 깊숙한 곳
            ]
        )
        """

        return obs


class MCTSROSNode(Node):
    def __init__(self):
        super().__init__("mcts_node")

        self._observation_manager = ObservationManager(self)
        self._mcts_engine = MCTS(time_limit=1.0, exploration_constant=20.0)

        # self._result_publisher = self.create_publisher(
        #     Int32MultiArray,
        #     self.get_name() + "/closest_object_ids",
        #     qos_profile_system_default,
        # )

        # self._cnt_pub = self.create_publisher(
        #     Int32,
        #     "/fcn_service_node/cnt",
        #     qos_profile=qos_profile_system_default,
        # )
        # self._cnt = 0

        # 비동기 서비스 처리를 위한 콜백 그룹
        self.srv_cb_group = ReentrantCallbackGroup()
        self.timer_cb_group = MutuallyExclusiveCallbackGroup()

        self._closest_object_sub = self.create_subscription(
            Int32MultiArray,
            "closest_object_classifier" + "/closest_object_ids",
            callback=self._closest_object_callback,
            qos_profile=qos_profile_system_default,
            callback_group=self.srv_cb_group,
        )

        self.srv = self.create_service(
            GetPolicyAction,
            "get_policy_action",
            self.handle_mcts_request,
            callback_group=self.srv_cb_group,
        )

        self._closest_object_list: List[int] = (
            []
        )  # 가장 가까운 객체 ID 리스트 (컬럼 순서대로)

        # self._timer = self.create_timer(
        #     1.0, self._publish_cnt, callback_group=self.timer_cb_group
        # )

    def _closest_object_callback(self, msg: Int32MultiArray):
        """가장 가까운 객체 ID 리스트를 수신하여 업데이트"""
        self._closest_object_list = msg.data

    # def _publish_cnt(self):
    #     """현재 카운트 값을 주기적으로 퍼블리시하는 헬퍼 함수 (디버깅용)"""
    #     cnt_msg = Int32()
    #     cnt_msg.data = self._cnt
    #     self._cnt_pub.publish(cnt_msg)

    def handle_mcts_request(
        self, request: GetPolicyAction.Request, response: GetPolicyAction.Response
    ):
        # 1. 요청 파라미터에서 타겟 클래스 인덱스 추출

        # self._cnt += 1

        target_id: int = request.target_id

        # 2. 관측값 업데이트 및 가공
        new_grid: np.ndarray = self._observation_manager.get_observation(
            target_object_id=target_id
        )

        # 3. State 재초기화
        grid_state = GridState(grid=new_grid, steps=0)
        self.get_logger().info(grid_state.print_grid())  # 초기 상태 출력 (디버깅용)

        # 4. MCTS 탐색 수행
        first_action: int = None
        while rclpy.ok() and not grid_state.is_terminal():
            best_action = self._mcts_engine.search(grid_state)
            if first_action is None:
                _, c = best_action
                first_action = c  # 첫 번째 액션의 컬럼 인덱스 저장

            grid_state = grid_state.take_action(best_action)
            self.get_logger().info(grid_state.print_grid())

        # 여러번 실행을 했고, 첫 번째 액션이 존재한다면 탐색이 정상적으로 이루어졌다고 판단
        trigger = grid_state.steps != 0 and first_action is not None
        random_action = random.choice(
            [i for i, obj_id in enumerate(self._closest_object_list) if obj_id != -1]
        )

        if trigger is True:
            response.target_column = first_action
            self.get_logger().info(
                f"MCTS 탐색 완료: 첫 번째 액션 컬럼 {first_action} 선택 (타겟 ID: {target_id})"
            )
        else:
            response.target_column = random_action
            self.get_logger().warn(
                f"MCTS 탐색 실패: 유효한 액션을 찾지 못했습니다. 랜덤 액션 컬럼 {random_action} 선택 (타겟 ID: {target_id})"
            )

        response.action_type = 0
        response.one_d_pdm = []
        response.one_d_image = Image()
        response.two_d_image = Image()

        return response


def main(args=None):
    rclpy.init(args=args)
    node = MCTSROSNode()

    # 비동기 서비스 처리(추론)와 타이머(시각화)가 동시에 돌아가기 위해 멀티스레드 사용
    executor = MultiThreadedExecutor()
    executor.add_node(node)

    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
