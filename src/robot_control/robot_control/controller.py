# ROS2
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.action.client import ClientGoalHandle
from rclpy.time import Time
from rclpy.duration import Duration
from rclpy.qos import QoSProfile, qos_profile_system_default

# Message
from std_msgs.msg import *
from geometry_msgs.msg import *
from sensor_msgs.msg import *
from nav_msgs.msg import *
from visualization_msgs.msg import *
from moveit_msgs.msg import *
from trajectory_msgs.msg import *
from shape_msgs.msg import *
from control_msgs.action import GripperCommand
from builtin_interfaces.msg import Duration as BuiltinDuration

# TF
from tf2_ros import *

# Python
import sys
import os
import numpy as np
from enum import Enum
import time
import threading
import copy
from rotutils import *
from dataclasses import dataclass

# Custom
from moveit2_commander import (
    FK_ServiceManager,
    IK_ServiceManager,
    CartesianPath_ServiceManager,
    KinematicPath_ServiceManager,
    GetPlanningScene_ServiceManager,
    ApplyPlanningScene_ServiceManager,
    ExecuteTrajectory_ServiceManager,
)


# 재시도 로직을 전담하는 헬퍼 함수 (*args와 **kwargs 지원)
def retry_step(
    step_func, max_retries=3, delay=0.5, *args, **kwargs
) -> tuple[bool, Any]:
    for attempt in range(1, max_retries + 1):
        # args와 kwargs를 step_func에 언패킹하여 전달
        res = step_func(*args, **kwargs)
        if res:
            return True, res

        print(
            f"Step 실패, 재시도 중... (최대 {max_retries}회, {attempt}/{max_retries})"
        )
        # 실패 시 딜레이 후 재시도 (필요하다면 여기에 로깅을 추가할 수 있습니다)
        time.sleep(delay)

    return False, None  # 최대 재시도 횟수 초과 시 False


def merge_trajectories(trajectories: List[RobotTrajectory]) -> RobotTrajectory:
    # (기존 작성하신 병합 로직과 동일하므로 내용은 그대로 유지합니다)
    if not trajectories:
        raise ValueError("병합할 궤적 리스트가 비어있습니다.")

    def duration_to_nanosec(duration: BuiltinDuration) -> int:
        return duration.sec * 1_000_000_000 + duration.nanosec

    def nanosec_to_duration(total_nanosec: int, duration_type) -> Any:
        sec = total_nanosec // 1_000_000_000
        nanosec = total_nanosec % 1_000_000_000
        return duration_type(sec=sec, nanosec=nanosec)

    merged_points = []
    accumulated_nanosec = 0

    for idx, traj in enumerate(trajectories):
        points: List[JointTrajectoryPoint] = traj.joint_trajectory.points
        if not points:
            continue

        start_point_idx = 1 if idx > 0 else 0

        for point in points[start_point_idx:]:
            current_point_ns = duration_to_nanosec(point.time_from_start)
            new_time_ns = accumulated_nanosec + current_point_ns

            new_point = JointTrajectoryPoint(
                positions=point.positions,
                velocities=point.velocities,
                accelerations=point.accelerations,
                effort=point.effort,
                time_from_start=nanosec_to_duration(
                    new_time_ns, type(point.time_from_start)
                ),
            )
            merged_points.append(new_point)

        last_point_ns = duration_to_nanosec(points[-1].time_from_start)
        accumulated_nanosec += last_point_ns

    base_traj: RobotTrajectory = trajectories[0]

    new_joint_trajectory = JointTrajectory(
        header=base_traj.joint_trajectory.header,
        joint_names=base_traj.joint_trajectory.joint_names,
        points=merged_points,
    )

    merged_trajectory = RobotTrajectory(
        joint_trajectory=new_joint_trajectory,
        multi_dof_joint_trajectory=base_traj.multi_dof_joint_trajectory,
    )

    return merged_trajectory


class UR5eController:
    def __init__(self, node: Node):
        self._node: Node = node

        # >>>>> Variables <<<<<
        self._planning_group: str = (
            "ur_manipulator"  # MoveIt2에서 설정한 UR5e의 Planning Group 이름
        )
        self._kinematic_tolerance: float = (
            0.01  # IK 솔버의 허용 오차 (예시값, 필요에 따라 조정)
        )
        self._fraction_threshold: float = (
            0.7  # Cartesian Path 계획 시 허용할 최소 경로 완성도 (예시값, 필요에 따라 조정)
        )
        self._default_frame_id: str = (
            "world"  # UR5e의 기본 프레임 ID (MoveIt2 설정에 따라 다를 수 있음)
        )
        self._end_effector_link: str = (
            "gripper_link"  # UR5e의 End Effector 링크 이름 (MoveIt2 설정에 따라 다를 수 있음)
        )
        # <<<<< End of Variables <<<<<

        # >>>>> Predefined Joint States <<<<<
        self._home_joints = JointState(
            header=Header(
                stamp=self._node.get_clock().now().to_msg(),
                frame_id="base_link",
            ),
            name=[
                "shoulder_lift_joint",
                "elbow_joint",
                "wrist_1_joint",
                "wrist_2_joint",
                "wrist_3_joint",
                "shoulder_pan_joint",
            ],
            position=[
                -0.853251652126648,
                -2.4234585762023926,
                -3.0269695721068324,
                -np.pi / 2.0,
                np.pi,
                -1.616389576588766,
            ],
        )
        self._safety_joints = JointState(
            header=Header(
                stamp=self._node.get_clock().now().to_msg(),
                frame_id="base_link",
            ),
            name=[
                "shoulder_lift_joint",
                "elbow_joint",
                "wrist_1_joint",
                "wrist_2_joint",
                "wrist_3_joint",
                "shoulder_pan_joint",
            ],
            position=[
                -0.7980526937250012,
                -2.2779344915978577,
                -3.22939911761183,
                -0.6398843295933805,
                3.150328623414728,
                -0.6956096575289052,
            ],
        )
        self._waiting_joints = JointState(
            header=Header(
                stamp=self._node.get_clock().now().to_msg(),
                frame_id="base_link",
            ),
            name=[
                "shoulder_lift_joint",
                "elbow_joint",
                "wrist_1_joint",
                "wrist_2_joint",
                "wrist_3_joint",
                "shoulder_pan_joint",
            ],
            position=[
                -0.14543800000017093,
                -0.5723896666667043,
                -2.2659790000078974,
                -1.76121310109918,
                3.1378451010107,
                0.04616566666664836,
            ],
        )
        # <<<<< End of Predefined Joint States <<<<<

        # >>>>> ROS Subscribers <<<<<
        self._joint_states: JointState = None
        self._joint_states_sub = self._node.create_subscription(
            JointState,
            "/joint_states",
            self._joint_states_callback,
            qos_profile_system_default,
        )

        self._collision_objects: List[CollisionObject] = None
        self._collision_objects_sub = self._node.create_subscription(
            MarkerArray,
            "/grid_markers",
            self._collision_objects_callback,
            qos_profile_system_default,
        )

        # >>>>> MoveIt2 Service Managers <<<<<
        self._fk_manager = FK_ServiceManager(node)
        self._ik_manager = IK_ServiceManager(node)
        self._cartesian_path_manager = CartesianPath_ServiceManager(
            node,
            planning_group=self._planning_group,
            fraction_threshold=self._fraction_threshold,
        )
        self._kinematic_path_manager = KinematicPath_ServiceManager(
            node, planning_group=self._planning_group
        )
        self._get_planning_scene_manager = GetPlanningScene_ServiceManager(node)
        self._apply_planning_scene_manager = ApplyPlanningScene_ServiceManager(node)
        self._execute_trajectory_manager = ExecuteTrajectory_ServiceManager(node)

    @property
    def joint_states(self) -> JointState:
        self._joint_states.header.stamp = self._node.get_clock().now().to_msg()
        return self._joint_states

    @property
    def home_joints(self) -> JointState:
        self._home_joints.header.stamp = self._node.get_clock().now().to_msg()
        return self._home_joints

    @property
    def safety_joints(self) -> JointState:
        self._safety_joints.header.stamp = self._node.get_clock().now().to_msg()
        return self._safety_joints

    @property
    def waiting_joints(self) -> JointState:
        self._waiting_joints.header.stamp = self._node.get_clock().now().to_msg()
        return self._waiting_joints

    @property
    def home_pose(self) -> PoseStamped | None:
        # FK 서비스를 이용하여 home_joints에 대한 TCP Pose 계산
        return self._fk_manager.run(
            joint_states=self._home_joints,
            end_effector=self._end_effector_link,
        )

    @property
    def safety_pose(self) -> PoseStamped | None:
        # FK 서비스를 이용하여 safety_joints에 대한 TCP Pose 계산
        return self._fk_manager.run(
            joint_states=self._safety_joints,
            end_effector=self._end_effector_link,
        )

    @property
    def waiting_pose(self) -> PoseStamped | None:
        # FK 서비스를 이용하여 waiting_joints에 대한 TCP Pose 계산
        return self._fk_manager.run(
            joint_states=self._waiting_joints,
            end_effector=self._end_effector_link,
        )

    @property
    def home_orientation(self) -> Quaternion:
        home_pose = self.home_pose
        if home_pose is not None:
            return home_pose.pose.orientation
        else:
            self._node.get_logger().warn(
                "홈 자세의 TCP Pose를 계산할 수 없습니다. 기본 orientation을 반환합니다."
            )
            return Quaternion(
                x=0.0, y=0.0, z=0.0, w=1.0
            )  # 기본 단위 쿼터니언 (회전 없음)

    @property
    def drop_orientation(self) -> Quaternion:
        home_orientation = self.home_orientation

        h_roll, h_pitch, h_yaw = euler_from_quaternion(
            [
                home_orientation.x,
                home_orientation.y,
                home_orientation.z,
                home_orientation.w,
            ]
        )

        sx, sy, sz, sw = quaternion_from_euler(
            [
                h_roll,
                h_pitch,
                h_yaw
                + np.deg2rad(
                    90.0
                ),  # safety_joints에서 yaw 방향으로 90도 회전한 orientation 계산
            ]
        )

        return Quaternion(
            x=sx,
            y=sy,
            z=sz,
            w=sw,
        )

    @property
    def sweep_orientation(self) -> Quaternion:
        home_orientation = self.home_orientation

        h_roll, h_pitch, h_yaw = euler_from_quaternion(
            [
                home_orientation.x,
                home_orientation.y,
                home_orientation.z,
                home_orientation.w,
            ]
        )

        # sweep_joints에서 pitch 방향으로 90도 회전한 orientation 계산
        sx, sy, sz, sw = quaternion_from_euler(
            [h_roll, h_pitch + +np.deg2rad(90.0), h_yaw]
        )

        return Quaternion(
            x=sx,
            y=sy,
            z=sz,
            w=sw,
        )

    @property
    def tcp_pose(self) -> PoseStamped | None:
        if self._joint_states is None:
            self._node.get_logger().warn(
                "현재 JointState가 수신되지 않았습니다. TCP Pose를 계산할 수 없습니다."
            )
            return None

        # FK 서비스를 이용하여 현재 관절 상태에 대한 TCP Pose 계산
        return self._fk_manager.run(
            joint_states=self._joint_states,
            end_effector=self._end_effector_link,
        )

    def _joint_states_callback(self, msg: JointState):
        """
        JointState 메시지를 수신하여 현재 로봇의 관절 상태를 업데이트하는 콜백 함수.
        기본: Node
        """
        self._joint_states = msg

    def _create_default_collision_object(self) -> List[CollisionObject]:

        collision_object = CollisionObject(
            id=f"10001",
            header=Header(
                stamp=self._node.get_clock().now().to_msg(),
                frame_id="base_link",
            ),
            operation=CollisionObject.ADD,
            primitives=[
                SolidPrimitive(
                    type=SolidPrimitive.BOX,
                    dimensions=[0.7, 0.7, 0.4],
                ),
            ],
            primitive_poses=[
                Pose(
                    position=Point(x=0.0, y=0.0, z=-0.2),
                    orientation=Quaternion(x=0.0, y=0.0, z=0.0, w=1.0),
                )
            ],
        )

        collision_objects = [collision_object]

        return collision_objects

    def _collision_objects_callback(self, msg: MarkerArray):
        """
        MarkerArray 메시지를 수신하여, 그 중 "grid_volume" 네임스페이스를 가진 마커들을 CollisionObject로 변환하여 리스트에 저장하는 콜백 함수.
        """

        def decode_id(encoded_val: int) -> Tuple[str, int]:
            """
            Marker ID로부터 행과 열 정보를 디코딩하는 함수.
            """
            base_val = encoded_val - 2000
            col_id = base_val % 10
            row_num = base_val // 10
            row_id = chr(row_num + 64)

            return row_id, col_id

        collision_objects = []

        for marker in msg.markers:
            marker: Marker

            if marker.ns == "grid_volume":
                # Marker의 위치와 크기를 CollisionObject로 변환하여 리스트에 저장
                row, col = decode_id(marker.id)

                collision_object = CollisionObject(
                    id=f"{row}{col}",
                    header=marker.header,
                    operation=CollisionObject.ADD,
                    primitives=[
                        SolidPrimitive(
                            type=SolidPrimitive.BOX,
                            dimensions=[marker.scale.x, marker.scale.y, marker.scale.z],
                        ),
                    ],
                    primitive_poses=[marker.pose],
                )
                collision_objects.append(collision_object)

        self._collision_objects = collision_objects

    def _get_header(self) -> Header:
        """
        현재 시점의 Header를 생성하여 반환하는 헬퍼 메서드.
        """
        return Header(
            stamp=self._node.get_clock().now().to_msg(),
            frame_id=self._default_frame_id,
        )

    def get_trajectory_duration(self, traj: RobotTrajectory) -> float:
        """
        RobotTrajectory의 총 실행 시간을 초 단위(float)로 반환한다.
        """
        if not traj.joint_trajectory.points:
            raise ValueError("Trajectory에 point가 없습니다.")

        last_point = traj.joint_trajectory.points[-1]
        duration = last_point.time_from_start

        return duration.sec + duration.nanosec * 1e-9

    def _get_and_apply_planning_scene(self) -> bool:
        """
        현재 로봇의 Planning Scene을 가져와서, 수신된 Collision Objects를 추가한 후 다시 적용하는 메서드.
        """

        def get_scene_with_remove_ops(original_scene: PlanningScene) -> PlanningScene:
            new_scene = copy.deepcopy(original_scene)
            remove_op = CollisionObject.REMOVE
            for obj in new_scene.world.collision_objects:
                obj: CollisionObject
                obj.operation = remove_op
            return new_scene

        """
        A. 현재 Planning Scene 가져오고, 리셋
        """

        def step_a_clear_scene() -> bool:
            """
            A 단계: Planning Scene에서 기존 Collision Objects를 제거하는 단계
            """
            current_scene: PlanningScene = self._get_planning_scene_manager.run()
            scene_to_clear = get_scene_with_remove_ops(current_scene)
            return self._apply_planning_scene_manager.run(
                collision_objects=scene_to_clear.world.collision_objects,
                scene=scene_to_clear,
            )

        def step_b_apply_new_objects() -> bool:
            """
            B 단계: 수신된 Collision Objects를 추가한 씬 생성 및 적용
            """
            current_scene: PlanningScene = self._get_planning_scene_manager.run()

            new_collision_objects = (
                self._collision_objects + self._create_default_collision_object()
            )

            return self._apply_planning_scene_manager.run(
                collision_objects=new_collision_objects,
                scene=current_scene,
            )

        _, _ = retry_step(step_a_clear_scene, max_retries=10)

        _, _ = retry_step(step_b_apply_new_objects, max_retries=10)

    def plan_and_execute_cartesian_path(
        self, waypoints: List[Pose], max_retries: int = 3
    ) -> bool:
        """
        주어진 waypoints 리스트를 따라 Cartesian Path를 계획하고 실행하는 메서드.
        실행 중 예기치 않게 일찍 종료된 경우, 이미 지나온 궤적은 무시하고
        현재 멈춘 위치에서 남은 Waypoint들에 대해서만 재플래닝하여 실행합니다.
        """

        attempt = 0
        current_waypoints = waypoints.copy()  # 원본 보존 및 남은 목표지점 관리용

        while attempt < max_retries:
            if not current_waypoints:
                self._node.get_logger().info(
                    "방문할 남은 Waypoint가 없습니다. 이동을 완료합니다."
                )
                return True

            # 1. Planning Scene 업데이트 (재시도 시 멈춘 위치 반영)
            self._node.get_logger().info(
                f"[{attempt + 1}/{max_retries}] Planning Scene 업데이트 중..."
            )
            _ = self._get_and_apply_planning_scene()

            trajs: List[RobotTrajectory] = []

            # 첫 번째 시작 상태는 현재 로봇의 관절 상태로 초기화
            current_start_state = self._joint_states

            # 2. 남은 current_waypoints에 대하여 Planning (LOOP)
            for i, waypoint in enumerate(current_waypoints):
                self._node.get_logger().info(
                    f"Waypoint {i + 1}/{len(current_waypoints)}에 대한 경로 계획 중..."
                )

                _, traj = retry_step(
                    step_func=self._cartesian_path_manager.run,
                    max_retries=10,
                    delay=0.5,
                    header=self._get_header(),
                    waypoints=[waypoint],
                    joint_states=current_start_state,
                    end_effector=self._end_effector_link,
                )

                # 예외 처리: 만약 특정 구간 플래닝 실패 시 로직 종료
                if not traj:
                    self._node.get_logger().error(
                        f"Waypoint {i + 1}에 대한 Planning에 실패했습니다."
                    )
                    return False

                trajs.append(traj)

                # 다음 루프를 위해 current_start_state를 방금 계산한 traj의 마지막 상태로 갱신
                last_point: JointTrajectoryPoint = traj.joint_trajectory.points[-1]
                current_start_state = JointState(
                    header=self._joint_states.header,
                    name=traj.joint_trajectory.joint_names,
                    position=last_point.positions,
                    velocity=last_point.velocities,
                    effort=last_point.effort,
                )

            # 4. Trajectory 병합
            merged_traj = merge_trajectories(trajs)
            total_duration = self.get_trajectory_duration(merged_traj)

            # 5. Trajectory 실행
            t1 = time.time()
            self._node.get_logger().info("병합된 궤적 실행 중...")
            self._execute_trajectory_manager.run(
                trajectory=merged_traj,
            )
            t2 = time.time()

            elapsed_time = t2 - t1
            wait_time = total_duration - elapsed_time

            if wait_time > 0.1:  # 통신 딜레이 고려 마진
                self._node.get_logger().warn(
                    f"Cartesian 이동이 예상보다 일찍 종료되었습니다. "
                    f"(예상: {total_duration:.2f}초, 실제: {elapsed_time:.2f}초)"
                )

                # 💡 핵심 로직: 실행된 시간을 추적하여 로봇이 몇 번째 Waypoint를 향하던 중이었는지 파악
                accumulated_time = 0.0
                resume_idx = 0
                for idx, t in enumerate(trajs):
                    dur = self.get_trajectory_duration(t)
                    accumulated_time += dur
                    if accumulated_time > elapsed_time:
                        resume_idx = idx
                        break

                # 이미 통과했거나 도달한 Waypoint는 리스트에서 슬라이싱하여 잘라냄
                current_waypoints = current_waypoints[resume_idx:]

                self._node.get_logger().info(
                    f"지나온 궤적을 무시하고, 현재 위치에서 남은 {len(current_waypoints)}개의 Waypoint로 재플래닝을 시도합니다."
                )
                attempt += 1
                continue
            else:
                self._node.get_logger().info(
                    "Cartesian Path 궤적 실행이 성공적으로 완료되었습니다."
                )
                return True

        self._node.get_logger().error(
            f"Cartesian Path 재시도 횟수({max_retries}회)를 초과하여 실패했습니다."
        )
        return False

    def plan_and_execute_kinematic_path(
        self, waypoints: List[Pose], max_retries: int = 3
    ) -> bool:

        attempt = 0
        current_waypoints = waypoints.copy()  # 원본 보존 및 남은 목표지점 관리용

        while attempt < max_retries:
            if not current_waypoints:
                self._node.get_logger().info(
                    "방문할 남은 Waypoint가 없습니다. 이동을 완료합니다."
                )
                return True

            # 1. Planning Scene 업데이트 (재시도 시 멈춘 위치 반영)
            self._node.get_logger().info(
                f"[{attempt + 1}/{max_retries}] Planning Scene 업데이트 중..."
            )
            _ = self._get_and_apply_planning_scene()

            trajs: List[RobotTrajectory] = []

            # 첫 번째 시작 상태는 현재 로봇의 관절 상태로 초기화
            current_start_state = self._joint_states

            # 2. 남은 current_waypoints에 대하여 Planning (LOOP)
            for i, waypoint in enumerate(current_waypoints):
                self._node.get_logger().info(
                    f"Waypoint {i + 1}/{len(current_waypoints)}에 대한 경로 계획 중..."
                )

                goal_robot_states: RobotState = self._ik_manager.run(
                    pose_stamped=PoseStamped(
                        header=self._get_header(),
                        pose=waypoint,
                    ),
                    joint_states=current_start_state,
                    end_effector=self._end_effector_link,
                )

                constraint = self._kinematic_path_manager.get_goal_constraint(
                    goal_joint_states=goal_robot_states.joint_state,
                    tolerance=self._kinematic_tolerance,
                )

                _, traj = retry_step(
                    step_func=self._kinematic_path_manager.run,
                    max_retries=10,
                    delay=0.5,
                    goal_constraints=[constraint],
                    path_constraints=None,
                    joint_states=current_start_state,
                    num_planning_attempts=100,
                    allowed_planning_time=1.0,
                    max_velocity_scaling_factor=1.0,
                    max_acceleration_scaling_factor=1.0,
                )

                # 예외 처리: 만약 특정 구간 플래닝 실패 시 로직 종료
                if not traj:
                    self._node.get_logger().error(
                        f"Waypoint {i + 1}에 대한 Planning에 실패했습니다."
                    )
                    return False

                trajs.append(traj)

                # 다음 루프를 위해 current_start_state를 방금 계산한 traj의 마지막 상태로 갱신
                last_point: JointTrajectoryPoint = traj.joint_trajectory.points[-1]
                current_start_state = JointState(
                    header=self._joint_states.header,
                    name=traj.joint_trajectory.joint_names,
                    position=last_point.positions,
                    velocity=last_point.velocities,
                    effort=last_point.effort,
                )

            # 4. Trajectory 병합
            merged_traj = merge_trajectories(trajs)
            total_duration = self.get_trajectory_duration(merged_traj)

            # 5. Trajectory 실행
            t1 = time.time()
            self._node.get_logger().info("병합된 궤적 실행 중...")
            self._execute_trajectory_manager.run(
                trajectory=merged_traj,
            )
            t2 = time.time()

            elapsed_time = t2 - t1
            wait_time = total_duration - elapsed_time

            if wait_time > 0.1:  # 통신 딜레이 고려 마진
                self._node.get_logger().warn(
                    f"Kinematic 이동이 예상보다 일찍 종료되었습니다. "
                    f"(예상: {total_duration:.2f}초, 실제: {elapsed_time:.2f}초)"
                )

                # 💡 핵심 로직: 실행된 시간을 추적하여 로봇이 몇 번째 Waypoint를 향하던 중이었는지 파악
                accumulated_time = 0.0
                resume_idx = 0
                for idx, t in enumerate(trajs):
                    dur = self.get_trajectory_duration(t)
                    accumulated_time += dur
                    if accumulated_time > elapsed_time:
                        resume_idx = idx
                        break

                # 이미 통과했거나 도달한 Waypoint는 리스트에서 슬라이싱하여 잘라냄
                current_waypoints = current_waypoints[resume_idx:]

                self._node.get_logger().info(
                    f"지나온 궤적을 무시하고, 현재 위치에서 남은 {len(current_waypoints)}개의 Waypoint로 재플래닝을 시도합니다."
                )
                attempt += 1
                continue
            else:
                self._node.get_logger().info(
                    "Kinematic Path 궤적 실행이 성공적으로 완료되었습니다."
                )
                return True

        self._node.get_logger().error(
            f"Kinematic Path 재시도 횟수({max_retries}회)를 초과하여 실패했습니다."
        )
        return False

    def moveJ(self, joint_states: JointState, max_retries: int = 3) -> bool:
        """
        주어진 JointState로 관절 이동하는 메서드.
        실행 중 예기치 않게 일찍 종료된 경우(wait_time > 0),
        현재 위치에서 남은 거리에 대해 재플래닝하여 이동을 재시도합니다.
        """

        goal_constraint = self._kinematic_path_manager.get_goal_constraint(
            goal_joint_states=joint_states, tolerance=self._kinematic_tolerance
        )

        attempt = 0
        while attempt < max_retries:
            # 1. Planning Scene 업데이트
            # (재시도 시, 방금 전까지 이동하다 멈춘 로봇의 최신 상태가 반영됨)
            _ = self._get_and_apply_planning_scene()

            # 2. Joint State에 대하여 Planning
            # 최신 self._joint_states를 시작점으로 사용하므로,
            # 이전에 이동했던 궤적은 자연스럽게 무시되고 "현재 위치 -> 목표 위치"의 새 궤적이 생성됨
            success, traj = retry_step(
                step_func=self._kinematic_path_manager.run,
                max_retries=10,
                delay=0.5,
                goal_constraints=[goal_constraint],
                path_constraints=None,
                joint_states=self._joint_states,
            )

            if not traj:
                self._node.get_logger().error("Trajectory 플래닝에 실패했습니다.")
                return False

            duration = self.get_trajectory_duration(traj)
            t1 = time.time()

            # 3. Trajectory 실행
            self._execute_trajectory_manager.run(
                trajectory=traj,
            )

            t2 = time.time()
            elapsed_time = t2 - t1
            wait_time = duration - elapsed_time

            # 통신 지연 등 미세한 오차를 고려해 0.1초 정도의 마진을 둡니다.
            if wait_time > 0.1:
                self._node.get_logger().warn(
                    f"[{attempt + 1}/{max_retries}] 관절 이동이 예상보다 일찍 종료되었습니다. "
                    f"(예상: {duration:.2f}초, 실제: {elapsed_time:.2f}초)\n"
                    "남은 구간 이동을 위해 현재 위치에서 재플래닝 및 실행을 시도합니다."
                )
                attempt += 1
                continue  # 루프의 처음으로 돌아가 Planning Scene 업데이트부터 다시 시작
            else:
                self._node.get_logger().info(
                    "관절 이동 명령이 예상 시간만큼 충분히 실행되었습니다."
                )
                return True

        self._node.get_logger().error(
            f"관절 이동 재시도 횟수({max_retries}회)를 초과하여 실패했습니다."
        )
        return False


class RobotiqController:
    def __init__(self, node: Node):
        self._node: Node = node
        self._action_client = ActionClient(
            self._node,
            GripperCommand,
            "/gripper/robotiq_gripper_controller/gripper_cmd",
        )

        # 메인 루프(State Machine)에서 상태를 체크하기 위한 내부 플래그
        self._is_finished = False
        self._is_success = False
        self._final_position = 0.0

    def control_gripper(self, open: bool = True, max_effort: float = 0.0):
        """[비동기] 그리퍼 제어 명령을 전송하고, 즉시 리턴하여 블로킹을 방지합니다."""
        position = 0.0 if open else 0.8
        goal_msg = GripperCommand.Goal()
        goal_msg.command.position = position
        goal_msg.command.max_effort = max_effort

        self._node.get_logger().info("Action Server 연결 대기 중...")
        self._action_client.wait_for_server()

        # 새로운 명령을 보내기 전 플래그 초기화
        self._is_finished = False
        self._is_success = False
        self._final_position = 0.0

        self._node.get_logger().info(
            f"▶️ 그리퍼 {'열기' if open else '닫기'} 명령 전송 중..."
        )

        # 1. 비동기 Goal 전송 및 콜백 연결 (여기서 코드 실행이 멈추지 않고 바로 넘어갑니다)
        future: Future = self._action_client.send_goal_async(goal_msg)
        future.add_done_callback(self._goal_response_callback)

    def _goal_response_callback(self, future: Future):
        """Goal 수락/거절 여부를 처리하는 콜백"""
        goal_handle = future.result()

        if not goal_handle.accepted:
            self._node.get_logger().error("❌ 서버가 Goal을 거절했습니다.")
            self._is_finished = True
            self._is_success = False
            return

        self._node.get_logger().info("🟢 Goal 수락됨! 물리적 동작 완료 대기 중...")

        # 2. Result(물리적 동작 완료)를 기다리는 비동기 요청 및 콜백 연결
        result_future: Future = goal_handle.get_result_async()
        result_future.add_done_callback(self._get_result_callback)

    def _get_result_callback(self, future: Future):
        """최종 동작이 완료되었을 때 실행되는 콜백"""
        result = future.result().result
        self._final_position = result.position

        # self._node.get_logger().info(f"✅ 동작 완료! 최종 위치: {self._final_position}")

        # 메인 상태 머신이 다음 단계로 넘어갈 수 있도록 플래그 업데이트
        self._is_success = True
        self._is_finished = True

    # ==========================================
    # 상태 머신(State Machine)에서 읽어갈 프로퍼티들
    # ==========================================
    @property
    def is_finished(self) -> bool:
        """액션이 완전히 끝났는지(성공/실패 무관) 확인합니다."""
        # 1회 읽고 나면 자동으로 초기화하게 하려면 여기서 self._is_finished = False 처리를 해도 좋습니다.
        return self._is_finished

    @property
    def is_success(self) -> bool:
        """액션이 성공적으로 수행되었는지 확인합니다."""
        return self._is_success

    @property
    def final_position(self) -> float:
        """동작 완료 후 그리퍼의 최종 위치를 반환합니다."""
        return self._final_position
