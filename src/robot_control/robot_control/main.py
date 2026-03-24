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
    SweepLeftActionSequence,
    SweepRightActionSequence,
    AxisDirection,
)
from robot_control.controller import UR5eController, RobotiqController
from custom_msgs.srv import GetPolicyAction, GetNextDropCell


from rclpy.node import Node

# from your_package.srv import GetFCNResult (실제 사용하는 패키지에 맞게 import 필요)


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
        self.req_cnt += 1
        self._node.get_logger().info(f"▶️ [{self.req_cnt}]번째 동기식 추론 요청 전송...")

        req = GetPolicyAction.Request()
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

            return result

        except Exception as e:
            self._node.get_logger().error(f"❌ [{self.req_cnt}]번째 요청 실패: {e}")
            return None


class TargetObjectPicker:
    def __init__(self, node: Node, transform_manager: TransformManager):
        self._node = node
        self._transform_manager = transform_manager

        self._sub = self._node.create_subscription(
            MarkerArray,
            "/grid_markers",
            self._marker_callback,
            qos_profile=qos_profile_system_default,
        )

        self._msg: MarkerArray = None

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

            if marker.ns == "grid_cells":  # "grid_volume":
                row, col = self._decode_marker_id(marker.id)
                if col == column_id:
                    object_in_column[row] = marker

        return object_in_column[sorted(object_in_column.keys())[0]]

    def get_target_object_by_row(self, row_id: str) -> Marker:
        object_in_row = {}

        for marker in self._msg.markers:
            marker: Marker

            if marker.ns == "grid_cells":  # "grid_volume":
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

        self._client = self._node.create_client(GetNextDropCell, "request_drop_cell")

        self._node.get_logger().info("DropGrid 서비스 서버 대기 중...")
        while not self._client.wait_for_service(timeout_sec=1.0):
            self._node.get_logger().info("DropGrid 서비스 서버 대기 중...")

        self._node.get_logger().info("🟢 DropGrid 서비스 서버 확인 완료!")
        self._node.get_logger().info("DropGridSyncClient 초기화 완료!")

    def request_next_drop_cell_sync(self) -> GetNextDropCell.Response:
        req = GetNextDropCell.Request()

        try:
            result: GetNextDropCell.Response = self._client.call(req)

            if result.success:
                self._node.get_logger().info(
                    f"✅ 다음 드롭 셀 응답 수신: Row ID = {result.row_id}, Col ID = {result.col_id}"
                )
            else:
                self._node.get_logger().warn("빈 그리드가 없습니다 (모든 셀이 채워짐).")

            return result

        except Exception as e:
            self._node.get_logger().error(f"❌ 드롭 셀 요청 실패: {e}")
            return None


class MainControlNode(Node):

    class State(Enum):
        SEARCH = 0
        ACTION = 1
        END = 2

    def __init__(self):
        super().__init__("main_control_node")

        # 초기 상태 설정
        self._state = self.State.SEARCH

        # UR5eController 인스턴스
        self._ur5e_controller = UR5eController(node=self)

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
            direction=AxisDirection.POS_Y,  # 예시로 POS_X 방향으로 설정 (실제 방향은 DRL 모듈에서 받아와야 함)
        )
        self._sweep_left_action_sequence = SweepLeftActionSequence(
            node=self,
            ur_controller=self._ur5e_controller,
            gripper_controller=self._robotiq_controller,
            target_point=None,  # 실제 타겟 포인트는 DRL 모듈에서 받아와야 하므로 초기값은 None
        )
        self._sweep_right_action_sequence = SweepRightActionSequence(
            node=self,
            ur_controller=self._ur5e_controller,
            gripper_controller=self._robotiq_controller,
            target_point=None,  # 실제 타겟 포인트는 DRL 모듈에서 받아와야 하므로 초기값은 None
        )

        self._transform_manager = TransformManager(node=self)

        self._drl_client = DRLClient(node=self, target_class_idx=0)
        self._drop_client = DropGridSyncClient(node=self)
        self._target_picker = TargetObjectPicker(
            node=self, transform_manager=self._transform_manager
        )

        # >>> System Variables >>>

        self._methods = {
            self.State.SEARCH: self._drl_search,
            self.State.ACTION: self._execute_action,
            self.State.END: self._end,
        }

        self._action_type: int = None
        self._target_column: int = None
        self._drop_cell: Point = None
        self._temp_target_obejct: Marker = None
        # <<< System Variables <<<

    def _drl_search(self):
        # 1. DRL 모듈에 동기식 요청 보내기
        # int32 action_type / int32 target_column 응답
        res: GetPolicyAction.Response = self._drl_client.send_request_sync()

        self._action_type: int = res.action_type
        self._target_column: int = res.target_column

        self._action_type = (
            0  # TODO: 테스트 용도, 실제로는 DRL 모듈에서 받아온 action_type 사용
        )

        import random

        self._target_column: int = random.randint(
            0, 3
        )  # TODO: 테스트 용도, 실제로는 DRL 모듈에서 받아온 target_column 사용

        if self._action_type == 0:
            # Drop 해야 하기에, 다음 드롭 셀 정보 요청 -> ActionSequence에 타겟 포인트로 전달
            """
            bool success
            string row_id
            int32 col_id
            string frame_id
            geometry_msgs/Point center_coord
            geometry_msgs/Vector3 size
            """
            res: GetNextDropCell.Response = (
                self._drop_client.request_next_drop_cell_sync()
            )

            pose = res.center_coord.pose
            transformed_pose = self._transform_manager.transform_pose(
                pose=pose,
                target_frame="world",
                source_frame=res.center_coord.header.frame_id,
            )

            self._drop_cell = transformed_pose.pose.position
            self._grasp_action_sequence.drop_point = self._drop_cell

        # 2. TargetObjectPicker에서 타겟 오브젝트 정보 가져와서 ActionSequence에 타겟 포인트로 전달
        target_object_marker: Marker = self._target_picker.get_target_object_by_column(
            self._target_column
        )
        self._temp_target_obejct = self._target_picker.post_process_target_object(
            target_object_marker
        )

        # Update
        self._grasp_action_sequence.target_point = (
            self._temp_target_obejct.pose.position
        )

        return True

    def _execute_action(self):
        """
        self._action_type에 해당하는 액션 시퀀스 실행
        res가 True가 될 때까지 반복
        """
        methods = {
            0: self._grasp_action_sequence.step,
            1: self._sweep_left_action_sequence.step,
            2: self._sweep_right_action_sequence.step,
        }

        res: bool = methods[self._action_type]()
        return res

    def _end(self):
        return False

    def _update_state(self):
        # State 변경 로직. 변경 요망: 실제 DRL 모듈의 응답에 따라 상태를 변경하도록 구현 필요
        self._state = self.State(self._state.value + 1)
        if self._state == self.State.END:
            self._state = (
                self.State.SEARCH
            )  # END 상태에서 다시 SEARCH로 돌아가도록 설정 (필요에 따라 변경 가능)
        self.get_logger().info(f"State changed to: {self._state.name}")
        return self._state

    def step(self):
        res: bool = self._methods[self._state]()
        if res:
            self._update_state()


def main(args=None):
    rclpy.init(args=args)

    node = MainControlNode()

    th = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    th.start()

    hz = 30.0
    r = node.create_rate(hz)

    WAIT_TIME = 2.0
    for _ in range(int(WAIT_TIME * hz)):
        # 초기화 대기 시간 동안 노드가 정상적으로 실행되고 있는지 확인하기 위해 로그 출력
        r.sleep()

    # try:
    while rclpy.ok():

        node.step()
        r.sleep()
    # except KeyboardInterrupt:
    #     node.get_logger().info("KeyboardInterrupt received, shutting down.")
    # except Exception as e:
    #     node.get_logger().error(f"Exception in main loop: {e}")
    # finally:
    th.join(timeout=1.0)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
