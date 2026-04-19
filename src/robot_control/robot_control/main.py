# ROS2
import rclpy
from rclpy.node import Node
from rclpy.time import Time
from rclpy.task import Future
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
import datetime
import cv2
import os
import copy
import numpy as np
from enum import Enum
import time
import threading
from abc import ABC, abstractmethod

# Custom
from rotutils import *
from base_package.transform_manager import TransformManager
from robot_control.action_sequence import (
    ActionSequence,
    GraspActionSequence,
    SweepActionSequence,
    AxisDirection,
)
from robot_control.controller import UR5eController, RobotiqController
from custom_msgs.srv import GetPolicyAction, GetNextDropCell

import json
from base_package.image_manager import ImageManager
from rclpy.node import Node

# from your_package.srv import GetFCNResult (실제 사용하는 패키지에 맞게 import 필요)


class ImageSaver:
    def __init__(self, node: Node):
        # 기본 인자
        self._node = node

        # >>> 로그용 인자 >>>
        self._log_dir = "/home/irol/DRL-Occluded-Object-Search/ssal"
        # <<< 로그용 인자 <<<

        # >>> 로깅 시작 >>>
        os.makedirs(self._log_dir, exist_ok=True)

        # >>> ROS Subscriber & Publisher 초기화 >>>
        self._raw_image: Image = None

        self._image_manager = ImageManager(
            node=self._node,
            subscribed_topics=[
                {
                    "topic_name": "/camera/camera1/color/image_raw",
                    "callback": self._callback_raw_image,
                }
            ],
            published_topics=[],
        )

    def _callback_raw_image(self, msg: Image):
        self._raw_image = msg

    def _post_process_images(self, msg: Image, ignore_none: bool = False) -> np.ndarray:

        if msg is None:
            self._node.get_logger().warn("Received None image, returning None.")
            return None

        np_image = self._image_manager.decode_message(
            image_msg=msg, desired_encoding="bgr8"
        )
        if np_image.shape[0] != 480 or np_image.shape[1] != 640:
            np_image = self._image_manager.crop_image(img=np_image)

        return np_image

    def log(self):
        # Images from Subscribers
        raw_image = self._post_process_images(self._raw_image)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        cv2.imwrite(os.path.join(self._log_dir, f"image_{timestamp}.png"), raw_image)


class DRLClient:
    def __init__(self, node: Node, target_class_idx: int = 0):
        self._node = node

        self._target_class_idx = target_class_idx

        self.req_cnt = 0  # 누락되었던 요청 횟수 카운터 초기화 추가

        self._client = self._node.create_client(GetPolicyAction, "get_policy_action")

        self._node.get_logger().info("DRL 서비스 서버 대기 중...")
        while not self._client.wait_for_service(timeout_sec=1.0):
            self._node.get_logger().info("DRL 서비스 서버 대기 중...")

        self._node.get_logger().info("🟢 DRL 서비스 서버 확인 완료!")
        self._node.get_logger().info("DRL 모듈 초기화 완료!")

    @property
    def target_class_idx(self) -> int:
        return self._target_class_idx

    @target_class_idx.setter
    def target_class_idx(self, val: int):
        self._target_class_idx = int(val)

    def send_request_sync(self) -> GetPolicyAction.Response:
        self._node.get_logger().info(f"▶️ [{self.req_cnt}]번째 동기식 추론 요청 전송...")

        req = GetPolicyAction.Request()
        req.index = self.req_cnt  # Episode 구분을 위한 인덱스 추가
        req.target_id = int(self._target_class_idx)

        self._node.get_logger().info(
            f"▶️ 요청 내용: Target Class Index = {self._target_class_idx}"
        )

        try:
            # call_async 대신 동기식 call() 메서드 사용 (응답이 올 때까지 블로킹됨)
            result: GetPolicyAction.Response = self._client.call(req)

            action_type: int = result.action_type
            target_column: int = result.target_column

            self._node.get_logger().info(
                f"✅ [{self.req_cnt}]번째 응답 수신: Action Type = {action_type}, Target Column = {target_column}"
            )

            self.req_cnt += 1

            return result

        except Exception as e:
            self._node.get_logger().error(f"❌ [{self.req_cnt}]번째 요청 실패: {e}")
            return None


class TargetObjectPicker:
    def __init__(self, node: Node, transform_manager: TransformManager):
        self._node = node
        self._transform_manager = transform_manager

        self._cloest_object_ids: List[int] = []
        self._cloest_object_distances: List[float] = []

        self._sub = self._node.create_subscription(
            MarkerArray,
            "/grid_markers",
            self._marker_callback,
            qos_profile=qos_profile_system_default,
        )
        self._c_object_ids_sub = self._node.create_subscription(
            Int32MultiArray,
            "/closest_object_classifier/closest_object_ids",
            self._closest_object_ids_callback,
            qos_profile=qos_profile_system_default,
        )
        self._c_object_dist_sub = self._node.create_subscription(
            Float32MultiArray,
            "/front_object_distance",
            callback=self._cloest_distance_callback,
            qos_profile=qos_profile_system_default,
        )

        self._msg: MarkerArray = None

    @property
    def closest_object_ids(self) -> List[int]:
        return self._cloest_object_ids

    @property
    def closest_object_distances(self) -> List[float]:
        return self._cloest_object_distances

    def check_distance_valid(self, idx: int, action: int) -> bool:
        if action == 0:
            return True  # Grasp 액션은 항상 유효하다고 간주 (거리 비교 없이 실행)

        if action == 1:
            # 오른쪽으로 미는 액션. 오른쪽 index의 물체가 더 가깝거나, 같으면 무효
            # 혹은, 오른쪽 index가 범위를 벗어나면, 무효
            target_dist = self._cloest_object_distances[idx]
            right_idx = idx + 1

            if right_idx >= len(self._cloest_object_distances):
                return False

            right_dist = self._cloest_object_distances[right_idx]
            if (
                np.abs(right_dist - target_dist) <= 0.1
            ):  # Threshold for distance comparison
                return False

            return True

        elif action == 2:
            # 왼쪽으로 미는 액션. 왼쪽 index의 물체가 더 가깝거나, 같으면 무효
            # 혹은, 왼쪽 index가 범위를 벗어나면, 무효
            target_dist = self._cloest_object_distances[idx]
            left_idx = idx - 1

            if left_idx < 0:
                return False

            left_dist = self._cloest_object_distances[left_idx]
            if (
                np.abs(left_dist - target_dist) <= 0.1
            ):  # Threshold for distance comparison
                return False

            return True

        else:
            # 그냥 무효
            return False

    def _closest_object_ids_callback(self, msg: Int32MultiArray):
        self._cloest_object_ids = msg.data

    def _cloest_distance_callback(self, msg: Float32MultiArray):
        self._cloest_object_distances = msg.data

    def _marker_callback(self, msg: MarkerArray):
        self._msg = msg

    def _decode_marker_id(self, marker_id: str) -> Tuple[str, int]:
        # 인코딩 공식: ((ord(self._row_id) - 64) * 10) + self._col_id + 2000
        marker_id_int = int(marker_id)
        row_id = (marker_id_int - 2000) // 10 + 64
        col_id = (marker_id_int - 2000) % 10

        # TODO: 테스트 용도!
        row_id = (marker_id_int - 0) // 10 + 64
        col_id = (marker_id_int - 0) % 10

        return chr(row_id), col_id

    def get_target_object_by_column(self, column_id: int) -> Marker:
        object_in_column = {}

        for marker in self._msg.markers:
            marker: Marker

            if marker.ns == "grid_volume":  # "grid_volume":
                row, col = self._decode_marker_id(marker.id)
                if col == column_id:
                    object_in_column[row] = marker

        return object_in_column[sorted(object_in_column.keys())[0]]

    def get_target_object_by_row(self, row_id: str) -> Marker:
        object_in_row = {}

        for marker in self._msg.markers:
            marker: Marker

            if marker.ns == "grid_volume":  # "grid_volume":
                row, col = self._decode_marker_id(marker.id)

                if row == row_id:
                    object_in_row[col] = marker

        return object_in_row[sorted(object_in_row.keys())[0]]

    def post_process_target_object(self, marker: Marker) -> Marker:
        # 예시: 좌표 변환
        new_frame = "world"

        transformed_pose = self._transform_manager.transform_pose(
            pose=marker.pose,
            target_frame=new_frame,
            source_frame=marker.header.frame_id,
        )

        new_marker = copy.deepcopy(marker)
        new_marker.pose = transformed_pose.pose
        new_marker.header.frame_id = new_frame

        return new_marker


class DropGridSyncClient:
    def __init__(self, node: Node):
        self._node = node

        self._drop_cnt = 0
        self._client = self._node.create_client(GetNextDropCell, "request_drop_cell")

        self._node.get_logger().info("DropGrid 서비스 서버 대기 중...")
        while not self._client.wait_for_service(timeout_sec=1.0):
            self._node.get_logger().info("DropGrid 서비스 서버 대기 중...")

        self._node.get_logger().info("🟢 DropGrid 서비스 서버 확인 완료!")
        self._node.get_logger().info("DropGridSyncClient 초기화 완료!")

    def request_next_drop_cell_sync(self) -> GetNextDropCell.Response:
        req = GetNextDropCell.Request()
        req.index = self._drop_cnt

        try:
            result: GetNextDropCell.Response = self._client.call(req)

            if result.success:
                self._node.get_logger().info(
                    f"✅ 다음 드롭 셀 응답 수신: Row ID = {result.row_id}, Col ID = {result.col_id}"
                )
            else:
                self._node.get_logger().warn("빈 그리드가 없습니다 (모든 셀이 채워짐).")

            self._drop_cnt += 1

            return result

        except Exception as e:
            self._node.get_logger().error(f"❌ 드롭 셀 요청 실패: {e}")
            return None


class FakeActionCreator:
    def __init__(self, file_path: str):
        # 파일 경로에서 객체 이름과 인덱스 매핑 정보를 읽어와서 SmartNameDict에 저장
        """
        {
            "actions": [
                {
                    "action_type": 0,
                    "target_column": 2
                },
                {
                    "action_type": 1,
                    "target_column": 3
                }
            ]
        }
        """

        data = json.load(open(file_path, "r"))
        self._idx = 0
        self._actions = data["actions"]

    def get_action(self) -> Tuple[int, int]:
        action_info = self._actions[self._idx]
        self._idx += 1

        return action_info["action_type"], action_info["target_column"]


class MainControlNode(Node):

    class State(Enum):
        SEARCH = 0
        ACTION = 1
        END = 2
        FINISHED = 999

    def __init__(self, target_class_idx: int = 0):
        super().__init__("main_control_node")

        # 초기 상태 설정
        self._target_class_idx = target_class_idx
        self._is_finished = False
        self._state = self.State.SEARCH

        self._image_saver = ImageSaver(node=self)

        # UR5eController 인스턴스
        self._ur5e_controller = UR5eController(node=self)

        self._fake_action_creator = FakeActionCreator(
            file_path="/home/irol/DRL-Occluded-Object-Search/src/robot_control/resource/fake_action.json"
        )
        self._use_fake_action = False  # True로 설정하면 DRLClient 대신 FakeActionCreator에서 액션을 가져와서 실행 (테스트 용도)

        # RobotiqController 인스턴스 (현재는 None으로 전달, 실제 구현 필요)
        self._robotiq_controller = RobotiqController(
            node=self,
        )

        # ActionSequence 인스턴스
        self._grasp_action_sequence = GraspActionSequence(
            node=self,
            ur_controller=self._ur5e_controller,
            gripper_controller=self._robotiq_controller,
            target_point=None,  # 실제 타겟 포인트는 DRL 모듈에서 받아와야 하므로 초기값은 None
            direction=AxisDirection.POS_X,
        )
        self._sweep_right_action_sequence = SweepActionSequence(
            node=self,
            ur_controller=self._ur5e_controller,
            gripper_controller=self._robotiq_controller,
            target_point=None,  # 실제 타겟 포인트는 DRL 모듈에서 받아와야 하므로 초기값은 None
            direction=AxisDirection.POS_X,
            sweep_direction=AxisDirection.NEG_Y,  # 오른쪽으로 스윕
            sweep_distance=0.1,  # 스윕 거리 (예시값, 실제로는 DRL 모듈에서 받아와야 할 수도 있음)
            offset_distance=0.06,  # 타겟 포인트에서 스윕 시작 지점까지의 오프셋 거리 (예시값, 실제로는 DRL 모듈에서 받아와야 할 수도 있음)
        )
        self._sweep_left_action_sequence = SweepActionSequence(
            node=self,
            ur_controller=self._ur5e_controller,
            gripper_controller=self._robotiq_controller,
            target_point=None,  # 실제 타겟 포인트는 DRL 모듈에서 받아와야 하므로 초기값은 None
            direction=AxisDirection.POS_X,
            sweep_direction=AxisDirection.POS_Y,  # 왼쪽으로 스윕
            sweep_distance=0.1,  # 스윕 거리 (예시값, 실제로는 DRL 모듈에서 받아와야 할 수도 있음)
            offset_distance=0.06,  # 타겟 포인트에서 스윕 시작 지점까지의 오프셋 거리 (예시값, 실제로는 DRL 모듈에서 받아와야 할 수도 있음)
        )

        self._sequences: dict[int, ActionSequence] = {
            0: self._grasp_action_sequence,
            1: self._sweep_right_action_sequence,
            2: self._sweep_left_action_sequence,
        }

        self._transform_manager = TransformManager(node=self)

        self._drl_client = DRLClient(node=self, target_class_idx=self._target_class_idx)
        self._drop_client = DropGridSyncClient(node=self)
        self._target_picker = TargetObjectPicker(
            node=self, transform_manager=self._transform_manager
        )

        # >>> System Variables >>>

        self._methods = {
            self.State.SEARCH: self._drl_search,
            self.State.ACTION: self._execute_action,
            self.State.END: self._end,
            self.State.FINISHED: self._finished,
        }

        self._exp_index: int = 0
        self._action_type: int = None
        self._target_column: int = None
        self._drop_cell: Point = None
        # <<< System Variables <<<

    def _drl_search(self):
        """
        0: Grasp
        1: Sweep Right
        2: Sweep Left
        """
        import random

        # 1. DRL 모듈에 동기식 요청 보내기
        # int32 action_type / int32 target_column 응답
        res: GetPolicyAction.Response = self._drl_client.send_request_sync()

        self._action_type: int = res.action_type
        self._target_column: int = res.target_column
        one_d_pdm: List[float] = res.one_d_pdm

        # FOR TEST
        # self._action_type = 1
        # self._target_column = 3

        self._image_saver.log()

        if self._use_fake_action:
            f_action, f_column = self._fake_action_creator.get_action()
            self._action_type = f_action
            self._target_column = f_column

        # 가장 가까운 물체 리스트에, 타겟 물체가 있다면, 강제로 그것을 피킹합니다.
        # 이후, Flag를 변경하여 무한 루프합니다.
        if self._target_class_idx in self._target_picker.closest_object_ids:
            self.get_logger().info(
                f"타겟 클래스 인덱스 {self._target_class_idx}가 가장 가까운 물체 리스트에 존재합니다. 강제로 Grasp 액션을 수행하도록 설정합니다."
            )

            self._is_finished = True
            self._action_type = 0  # Grasp로 강제 설정
            self._target_column = self._target_picker.closest_object_ids.index(
                self._target_class_idx
            )

            self.get_logger().info(f"INDEX: {self._target_column}")

        # 해당 출력이 정상적인지 검사하고, 정상적이지 않다면, one_d_pdm의 기반한 값으로 바꿔치기 합니다.
        elif self._target_picker.closest_object_ids[self._target_column] == -1:
            self.get_logger().error(
                f"DRL 모듈에서 받은 타겟 컬럼 {self._target_column}에 대한 Closest Object ID가 -1로 나타났습니다. one_d_pdm 값을 기반으로 타겟 컬럼을 재설정합니다."
            )

            # one_d_pdm 값을 기반으로 타겟 컬럼을 재설정
            # one_d_pdm에서 가장 높은 값을 가진 인덱스를 타겟 컬럼으로 설정
            self._action_type = 0  # Grasp로 강제 설정
            self._target_column = np.argmax(one_d_pdm)

            self.get_logger().info(f"재설정된 타겟 컬럼: {self._target_column}")

        # check action validation
        is_action_valid = self._target_picker.check_distance_valid(
            idx=self._target_column, action=self._action_type
        )
        if not is_action_valid:
            self.get_logger().warn(
                f"판별된 액션이 유효하지 않습니다. Action Type: {self._action_type}, Target Column: {self._target_column}. 액션을 무시하고 다음 탐색으로 넘어갑니다."
            )
            self._action_type = 0  # Grasp으로 강제 설정

        if self._action_type == 0:
            # Grasp의 경우에만, Drop 좌표를 계산함

            res: GetNextDropCell.Response = (
                self._drop_client.request_next_drop_cell_sync()
            )

            transformed_pose = self._transform_manager.transform_pose(
                pose=res.center_coord.pose,
                target_frame="world",
                source_frame=res.center_coord.header.frame_id,
            )

            self._drop_cell = transformed_pose.pose.position
            self._grasp_action_sequence.drop_point = self._drop_cell

        # 2. TargetObjectPicker에서 타겟 오브젝트 정보 가져와서 ActionSequence에 타겟 포인트로 전달
        target_object_marker: Marker = self._target_picker.get_target_object_by_column(
            self._target_column
        )
        processed_target_object_marker = self._target_picker.post_process_target_object(
            target_object_marker
        )

        # Update
        self._sequences[self._action_type].target_point = (
            processed_target_object_marker.pose.position
        )

        return True

    def _execute_action(self):
        """
        self._action_type에 해당하는 액션 시퀀스 실행
        res가 True가 될 때까지 반복
        """

        res: bool = self._sequences[
            self._action_type
        ].step()  # 액션 시퀀스의 step() 메서드 호출
        return res

    def _end(self):
        return False

    def _finished(self):
        while rclpy.ok():
            continue
        return True

    def _update_state(self):
        # State 변경 로직. 변경 요망: 실제 DRL 모듈의 응답에 따라 상태를 변경하도록 구현 필요
        self._state = self.State(self._state.value + 1)
        if self._state == self.State.END:

            if self._is_finished:
                # Task가 종료될, 경우 Finished 로 이동하여, 무한 루프에 갇힙니다.
                self._state = self.State.FINISHED
            else:
                # Task가 종료되지 않을 경우, SEARCH로 돌아가서, 다시 DRL 모듈에 요청을 보냅니다.
                self._state = self.State.SEARCH

        self.get_logger().info(f"State changed to: {self._state.name}")
        return self._state

    def step(self):
        res: bool = self._methods[self._state]()
        if res:
            self._update_state()


def main(args=None):
    rclpy.init(args=args)

    """
    {
        "can_1": "coca_cola", # 0
        "can_2": "sikhye", # 1
        "can_3": "yello_peach", # 2
        "can_4": "cantata", # 3
        "cup_1": "cup_sky", # 4
        "cup_2": "cup_white", # 5
        "cup_3": "cup_blue", # 6
        "cup_4": "cup_green", # 7
        "mug_1": "mug_black", # 8
        "mug_2": "mug_gray", # 9
        "mug_3": "mug_yello", # 10
        "mug_4": "mug_orange", # 11
        "bottle_1": "alive", # 12
        "bottle_2": "green_tea", # 13
        "bottle_3": "yello_smoothie", # 14 
        "bottle_4": "bottle_red",# 15
        "can_5": "cyder", # 16
    }
    """

    node = MainControlNode(target_class_idx=15)

    th = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    th.start()

    hz = 30.0
    r = node.create_rate(hz)

    WAIT_TIME = 3.0
    for _ in range(int(WAIT_TIME * hz)):
        # 초기화 대기 시간 동안 노드가 정상적으로 실행되고 있는지 확인하기 위해 로그 출력
        r.sleep()

    try:
        while rclpy.ok():

            node.step()
            r.sleep()
    except KeyboardInterrupt:
        node.get_logger().info("KeyboardInterrupt received, shutting down.")
    except Exception as e:
        node.get_logger().error(f"Exception in main loop: {e}")
    finally:
        th.join(timeout=1.0)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
