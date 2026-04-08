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
from abc import ABC, abstractmethod

# Custom
from rotutils import *
from robot_control.controller import UR5eController, RobotiqController


class AxisDirection(Enum):
    """6방향을 (x, y, z) 단위 벡터로 정의합니다."""

    POS_X = (1.0, 0.0, 0.0)
    NEG_X = (-1.0, 0.0, 0.0)
    POS_Y = (0.0, 1.0, 0.0)
    NEG_Y = (0.0, -1.0, 0.0)
    POS_Z = (0.0, 0.0, 1.0)
    NEG_Z = (0.0, 0.0, -1.0)

    def move_point(self, point: Point, distance: float, reverse: bool = False) -> Point:
        """
        Point를 현재 방향으로 distance만큼 이동시킨 새 Point를 반환
        reverse가 True면 반대 방향으로 이동
        """
        new_point = Point()
        # 원본 위치 + (방향 벡터 * 이동 거리)
        new_point.x = point.x + (self.value[0] * distance * (-1.0 if reverse else 1.0))
        new_point.y = point.y + (self.value[1] * distance * (-1.0 if reverse else 1.0))
        new_point.z = point.z + (self.value[2] * distance * (-1.0 if reverse else 1.0))
        return new_point

    def move_pose(self, pose: Pose, distance: float, reverse: bool = False) -> Pose:
        """Pose를 복사한 뒤 position만 현재 방향으로 이동시켜 반환"""
        new_pose = copy.deepcopy(pose)
        new_pose.position = self.move_point(
            point=pose.position, distance=distance, reverse=reverse
        )
        return new_pose


def rotate_direction_z(current_dir: AxisDirection, angle_deg: int) -> AxisDirection:
    """
    현재 정면 방향을 Z축 기준으로 회전한 후의 새로운 방향을 반환합니다.

    :param current_dir: 현재의 AxisDirection
    :param angle_deg: Z축 기준 회전 각도 (90 또는 -90)
    :return: 회전 후의 AxisDirection
    """
    if angle_deg not in (90, -90):
        raise ValueError("회전 각도는 90도 또는 -90도만 지원합니다.")

    # Z축 방향을 바라보고 있다면, Z축 회전을 해도 방향은 변하지 않음
    if current_dir in (AxisDirection.POS_Z, AxisDirection.NEG_Z):
        return current_dir

    if angle_deg == 90:
        # +90도 (반시계 방향, CCW) 회전
        rotation_map = {
            AxisDirection.POS_X: AxisDirection.POS_Y,
            AxisDirection.POS_Y: AxisDirection.NEG_X,
            AxisDirection.NEG_X: AxisDirection.NEG_Y,
            AxisDirection.NEG_Y: AxisDirection.POS_X,
        }
    else:  # angle_deg == -90
        # -90도 (시계 방향, CW) 회전
        rotation_map = {
            AxisDirection.POS_X: AxisDirection.NEG_Y,
            AxisDirection.NEG_Y: AxisDirection.NEG_X,
            AxisDirection.NEG_X: AxisDirection.POS_Y,
            AxisDirection.POS_Y: AxisDirection.POS_X,
        }

    return rotation_map[current_dir]


class ActionSequence:
    """
    전체 액션을 총괄하는 부모 시퀸스 클래스.
    Grasp / Sweep Left / Sweep Right 등의 자식 클래스로 구성될 예정이며,
    자식 클래스는 execute() 메서드를 구현하여 각 액션의 구체적인 동작을 정의한다.

    이러한 구조를 통하여, 전체 프로그램에서 인스턴스를 생성만 해두고
    DRL이 요청하는 Action 인스턴스를 콜만 해도 원하는 액션이 실행되도록 설계할 수 있다.
    """

    class State(Enum):
        HOME = 0
        # 이후는 자식 클래스에서 구체적으로 정의 (예: GRASP_POSE, LIFT, SWEEP_LEFT_POSE 등)

    def __init__(
        self,
        node: Node,
        ur_controller: UR5eController,
        gripper_controller: RobotiqController,
        target_point: Point,
        direction: AxisDirection = AxisDirection.POS_X,
    ):
        self._node: Node = node
        self._ur_controller: UR5eController = ur_controller
        self._gripper_controller: RobotiqController = gripper_controller
        self._target_point: Point = target_point
        self._direction: AxisDirection = direction

        self._drop_point: Point = (
            target_point  # 드롭 위치는 일단 target_point로 설정, 필요에 따라 별도 설정 가능
        )
        self._waypoints: List[Pose] = []  # 액션 수행을 위한 경로의 waypoints 리스트

        self._methods = (
            {}
        )  # 상태별 실행 메서드를 저장하는 딕셔너리, 자식 클래스에서 채워질 예정

        self._state = self.State.HOME  # 초기 상태는 HOME

    @property
    def target_point(self):
        return self._target_point

    @target_point.setter
    def target_point(self, value: Point | np.ndarray | Tuple[float, float, float]):
        self._node.get_logger().info(f"Setting target_point to: {value}")
        if isinstance(value, Point):
            self._target_point = value
        elif isinstance(value, np.ndarray) and value.shape == (3,):
            self._target_point = Point(x=value[0], y=value[1], z=value[2])
        elif isinstance(value, tuple) and len(value) == 3:
            self._target_point = Point(x=value[0], y=value[1], z=value[2])
        elif isinstance(value, list) and len(value) == 3:
            self._target_point = Point(x=value[0], y=value[1], z=value[2])
        else:
            raise ValueError(
                "Invalid type or shape for target_point. Expected Point, np.ndarray of shape (3,), or tuple of 3 floats."
            )

    @property
    def drop_point(self):
        return self._drop_point

    @drop_point.setter
    def drop_point(self, value: Point):
        self._drop_point = value

    @property
    def waypoints(self):
        return self._waypoints

    @waypoints.setter
    def waypoints(self, value: List[Pose]):
        if not isinstance(value, list) or not all(isinstance(p, Pose) for p in value):
            raise ValueError("Waypoints must be a list of Pose objects.")
        self._waypoints = value

    def step(self):
        self._node.get_logger().info(f"Executing step for state: {self._state.name}")

        self._methods[self._state]()

        if self._state.value + 1 >= len(self._methods):
            #  현재 상태가 마지막 상태인 경우, 다음 step에서 다시 HOME부터 시작하도록 초기화
            self._state = self.State(0)
            return True  # 액션 시퀸스 종료 신호
        else:
            self._state = self.State(self._state.value + 1)
            return False  # 액션 시퀸스 진행 중 신호


class GraspActionSequence(ActionSequence):

    class State(Enum):
        HOME = 0  # 시작 자세로 이동
        APPROACH = 1  # 잡기 위한 자세로 이동 -> waypoints에 의하여 여러번 이동되며, 그리퍼 중심 == 물체 중심
        GRASP = 2  # 그리퍼 닫기
        PLACE = 3  # 중간 세이프티 자세로 이동 -> 내려놓는 위치로 이동. 역시 waypoints에 의하여 여러번 이동될 수 있음
        RELEASE = 4  # 그리퍼 열기
        RETURN_HOME = 5  # 홈 자세로 이동

    def __init__(
        self,
        node: Node,
        ur_controller: UR5eController,
        gripper_controller: RobotiqController,
        target_point: Point,
        direction: AxisDirection = AxisDirection.POS_Y,
    ):
        super().__init__(
            node, ur_controller, gripper_controller, target_point, direction
        )

        self._state = self.State.HOME  # 초기 상태는 HOME
        self._methods = {
            self.State.HOME: self._home,
            self.State.APPROACH: self._approach,
            self.State.GRASP: self._grasp,
            self.State.PLACE: self._place,
            self.State.RELEASE: self._release,
            self.State.RETURN_HOME: self._return_home,
        }

    def _home(self):
        """
        홈 자세로 이동 및 실행
        1) ur_controller를 이용하여 홈 자세로 이동하는 경로 계획 및 실행
        2) gripper_controller를 이용하여 그리퍼 열기 명령 발행
        """

        self._ur_controller.moveJ(joint_states=self._ur_controller.home_joints)
        self._gripper_controller.control_gripper(open=True, max_effort=0.0)
        time.sleep(1.0)  # 그리퍼 대기 시간

    def _approach(self):
        """
        잡기 위한 자세로 이동
        1) 홈 포즈에서 약간 떨어진 중간 세이프티 자세로 이동
        2) target_point 앞 (예: 10cm) 위치로 이동
        3) target_point 더 가까운 앞 (예: 2cm) 위치로 이동
        4) target_point 위치로 이동 (그리퍼 중심 == 물체 중심)
         - 위의 1~4는 모두 waypoints(List[Pose])로 결정
        """

        # ur_controller에 정의된 safety_pose 시작
        safety_pose: Pose = self._ur_controller.safety_pose.pose

        # target_point에서 떨어진 앞 위치 (UR 정면의 역방향)
        first_aim_pose = Pose(
            position=self._direction.move_point(
                point=self._target_point, distance=0.1, reverse=True
            ),
            orientation=self._ur_controller.home_orientation,
        )

        # target_point에서, 조금 떨어진 앞 위치 (UR 정면의 역방향)
        second_aim_pose = Pose(
            position=self._direction.move_point(
                point=self._target_point, distance=0.02, reverse=True
            ),
            orientation=self._ur_controller.home_orientation,
        )

        # target_point 위치 (그리퍼 중심 == 물체 중심), Orientation 은 홈 자세와 동일하게 유지
        target_pose = Pose(
            position=self._target_point,
            orientation=self._ur_controller.home_orientation,
        )

        self._ur_controller.plan_and_execute_cartesian_path(
            waypoints=[safety_pose, first_aim_pose, second_aim_pose, target_pose],
            max_retries=20,
        )

    def _grasp(self):
        self._gripper_controller.control_gripper(open=False, max_effort=0.0)
        time.sleep(1.0)  # 그리퍼가 대기 시간

    def _place(self):
        # target_point에서, 조금 떨어진 앞 위치 (UR 정면의 역방향)
        second_aim_pose = Pose(
            position=self._direction.move_point(
                point=self._target_point, distance=0.02, reverse=True
            ),
            orientation=self._ur_controller.home_orientation,
        )
        second_aim_pose.position.z += 0.05  # 높이도 약간 올려서 중간 세이프티 자세로 이동

        # target_point에서, 떨어진 앞 위치 (UR 정면의 역방향)
        first_aim_pose = Pose(
            position=self._direction.move_point(
                point=self._target_point, distance=0.1, reverse=True
            ),
            orientation=self._ur_controller.home_orientation,
        )
        first_aim_pose.position.z += 0.05  # 높이도 약간 올려서 중간 세이프티 자세로 이동

        safety_pose: Pose = self._ur_controller.safety_pose.pose

        second_safety_pose: Pose = self._ur_controller.second_safety_pose.pose

        # Position은 drop_point, Orientation은 홈 자세 +90도
        drop_pose = Pose(
            position=self._drop_point,
            orientation=self._ur_controller.drop_orientation,
        )

        # Drop pose에서 0.05m 뒤로 이동한 위치 (UR -> Drop Grid 역방향)
        first_drop_pose = rotate_direction_z(
            current_dir=self._direction, angle_deg=int(90)
        ).move_pose(pose=drop_pose, distance=0.05, reverse=True)

        self._ur_controller.plan_and_execute_kinematic_path(
            waypoints=[
                second_aim_pose,
                first_aim_pose,
                safety_pose,
                second_safety_pose,
                first_drop_pose,
                drop_pose,
            ],
            max_retries=20,
        )


    def _release(self):
        self._gripper_controller.control_gripper(open=True, max_effort=0.0)
        time.sleep(1.0)  # 그리퍼가 대기 시간

    def _return_home(self):

        # 참조용
        drop_pose = Pose(
            position=self._drop_point,
            orientation=self._ur_controller.drop_orientation,
        )

        # Drop pose에서 0.05m 뒤로 이동한 위치 (UR -> Drop Grid 역방향)
        first_drop_pose = rotate_direction_z(
            current_dir=self._direction, angle_deg=int(90)
        ).move_pose(pose=drop_pose, distance=0.05, reverse=True)

        self._ur_controller.plan_and_execute_kinematic_path(waypoints=[first_drop_pose])

        # 대기 자세로 이동
        self._ur_controller.moveJ(joint_states=self._ur_controller.waiting_joints)

        return None

    # def step(self):

    #     self._node.get_logger().info(f"Executing step for state: {self._state.name}")

    #     self._methods[self._state]()

    #     if self._state == self.State.RETURN_HOME:
    #         self._state = (
    #             self.State.HOME
    #         )  # 다음 액션 시퀸스에서도 HOME부터 시작하도록 초기화
    #         return True  # 액션 시퀸스 종료 신호

    #     else:
    #         self._state = self.State(self._state.value + 1)
    #         return False  # 액션 시퀸스 진행 중 신호


class SweepActionSequence(ActionSequence):

    class State(Enum):
        HOME = 0  # 시작 자세로 이동
        APPROACH = 1  # Left 임으로, 물체 오른쪽으로 근접
        SWEEP = 2  # Sweep Left 수행
        RETURN_HOME = 3  # 홈 자세로 이동. waypoints에 의하여 여러번 이동될 수 있음

    def __init__(
        self,
        node: Node,
        ur_controller: UR5eController,
        gripper_controller: RobotiqController,
        target_point: Point,
        direction: AxisDirection = AxisDirection.POS_Y,
        sweep_direction: AxisDirection = AxisDirection.NEG_X,
        sweep_distance: float = 0.1,
        offset_distance: float = 0.02,
    ):
        super().__init__(
            node, ur_controller, gripper_controller, target_point, direction
        )

        self._sweep_direction = sweep_direction
        self._sweep_distance = sweep_distance
        self._offset_distance = offset_distance

        self._methods = {
            self.State.HOME: self._home,
            self.State.APPROACH: self._approach,
            self.State.SWEEP: self._sweep,
            self.State.RETURN_HOME: self._return_home,
        }

    def _home(self):
        """
        홈 자세로 이동 및 실행
        1) ur_controller를 이용하여 홈 자세로 이동하는 경로 계획 및 실행
        2) gripper_controller를 이용하여 그리퍼 열기 명령 발행
        """

        self._ur_controller.moveJ(joint_states=self._ur_controller.home_joints)
        self._gripper_controller.control_gripper(open=True, max_effort=0.0)
        time.sleep(1.0)  # 그리퍼 대기 시간

    def _approach(self):
        """
        잡기 위한 자세로 이동
        1) 홈 포즈에서 약간 떨어진 중간 세이프티 자세로 이동
        2) target_point 앞+살짝 옆 위치로 이동 + Gripper 직각 회전
        3) target_point 살짝 옆 위치로 이동 + Gripper 직각 회전
        4) 일정 거리 Sweep
        5) (2) 위치로 복귀 (Sweep 전 자세로)
        """

        # ur_controller에 정의된 safety_pose 시작
        safety_pose: Pose = self._ur_controller.safety_pose.pose

        # target_point에서 떨어진 앞 + 살짝 옆 위치 (UR 정면의 역방향)
        p1 = self._direction.move_point(
            point=self._target_point, distance=0.1, reverse=True
        )
        p2 = self._sweep_direction.move_point(
            point=p1,
            distance=self._offset_distance,
            reverse=True,
        )
        aim_pose = Pose(
            position=p2,
            orientation=self._ur_controller.sweep_orientation,
        )

        # target_point에서, 살짝 옆 위치 (UR 정면의 역방향)
        target_pose = Pose(
            position=self._sweep_direction.move_point(
                point=self._target_point,
                distance=self._offset_distance,
                reverse=True,
            ),
            orientation=self._ur_controller.sweep_orientation,
        )

        self._ur_controller.plan_and_execute_cartesian_path(
            waypoints=[safety_pose, aim_pose, target_pose],
            max_retries=20,
        )

    def _sweep(self):
        # return
        target_pose = Pose(
            position=self._sweep_direction.move_point(
                point=self._target_point,
                distance=self._sweep_distance,
                reverse=False,
            ),
            orientation=self._ur_controller.sweep_orientation,
        )

        self._ur_controller.plan_and_execute_cartesian_path(
            waypoints=[target_pose], max_retries=20
        )
        pass

    def _return_home(self):
        # target_point에서 떨어진 앞 + 살짝 옆 위치 (UR 정면의 역방향)
        p1 = self._direction.move_point(
            point=self._target_point, distance=0.1, reverse=True
        )
        p2 = self._sweep_direction.move_point(
            point=p1,
            distance=self._sweep_distance,
            reverse=False,
        )
        aim_pose = Pose(
            position=p2,
            orientation=self._ur_controller.sweep_orientation,
        )

        safety_pose: Pose = self._ur_controller.safety_pose.pose

        waiting_pose: Pose = self._ur_controller.waiting_pose.pose

        self._ur_controller.plan_and_execute_kinematic_path(
            # waypoints=[aim_pose, safety_pose, waiting_pose],
            waypoints=[aim_pose, safety_pose],
            max_retries=20,
        )

        self._ur_controller.moveJ(joint_states=self._ur_controller.waiting_joints)


