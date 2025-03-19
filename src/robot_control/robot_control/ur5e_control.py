# ROS2
import rclpy
from rclpy.node import Node
from rclpy.time import Time
from rclpy.duration import Duration
from rclpy.task import Future
from rclpy.qos import QoSProfile, qos_profile_system_default

# Message
from std_msgs.msg import *
from geometry_msgs.msg import *
from sensor_msgs.msg import *
from nav_msgs.msg import *
from visualization_msgs.msg import *
from trajectory_msgs.msg import *
from shape_msgs.msg import *
from moveit_msgs.srv import *
from moveit_msgs.action import *
from moveit_msgs.msg import *

# TF
from tf2_ros import *

# Python
import os
import sys
import numpy as np
from abc import ABC, abstractmethod


class ForwardKinematics(object):
    def __init__(self):
        self.joint_order = [
            "shoulder_pan_joint",
            "shoulder_lift_joint",
            "elbow_joint",
            "wrist_1_joint",
            "wrist_2_joint",
            "wrist_3_joint",
        ]
        self.link_order = [
            "base_link_inertia",
            "forearm_link",
            "shoulder_link",
            "upper_arm_link",
            "wrist_1_link",
            "wrist_2_link",
            "wrist_3_link",
        ]

    @staticmethod
    def dh_transform(a, d, alpha, theta):
        """
        Denavit-Hartenberg 변환 행렬 생성 함수.
        :param a: 링크 길이
        :param d: 링크 오프셋
        :param alpha: 링크 간 회전
        :param theta: 조인트 각도
        :return: 4x4 변환 행렬
        """
        return np.array(
            [
                [
                    np.cos(theta),
                    -np.sin(theta) * np.cos(alpha),
                    np.sin(theta) * np.sin(alpha),
                    a * np.cos(theta),
                ],
                [
                    np.sin(theta),
                    np.cos(theta) * np.cos(alpha),
                    -np.cos(theta) * np.sin(alpha),
                    a * np.sin(theta),
                ],
                [0, np.sin(alpha), np.cos(alpha), d],
                [0, 0, 0, 1],
            ]
        )

    @staticmethod
    def forward_kinematics(joint_angles):
        """
        UR5e Forward Kinematics 계산 함수.
        :param joint_angles: 길이 6짜리 NumPy 배열, 조인트 각도 (라디안)
        :return: End Effector Pose (4x4 변환 행렬)
        """

        # UR5e DH 파라미터
        dh_params = [
            # (a_i, d_i, alpha_i, theta_i)
            (0, 0.1625, np.pi / 2, joint_angles[0]),  # Joint 1
            (-0.425, 0, 0, joint_angles[1]),  # Joint 2
            (-0.3922, 0, 0, joint_angles[2]),  # Joint 3
            (0, 0.1333, np.pi / 2, joint_angles[3]),  # Joint 4
            (0, 0.0997, -np.pi / 2, joint_angles[4]),  # Joint 5
            (0, 0.0996, 0, joint_angles[5]),  # Joint 6
        ]

        # 초기 변환 행렬 (Identity Matrix)
        t_matrix = np.eye(4)

        # X, Y축 뒤집기 행렬. 왜 이런지는 모르겠음.
        flip_transform = np.array(
            [[-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
        )

        # 각 DH 파라미터를 적용하여 누적 변환 계산
        for a, d, alpha, theta in dh_params:
            t_matrix = np.dot(
                t_matrix, ForwardKinematics.dh_transform(a, d, alpha, theta)
            )

        return np.dot(flip_transform, t_matrix)

    def parse_robot_trajectory_to_path(
        self, header: Header, joint_trajectory: RobotTrajectory
    ) -> Path:
        """
        shoulder_pan_joint,
        shoulder_lift_joint,
        elbow_joint,
        wrist_1_joint,
        wrist_2_joint,
        wrist_3_joint,
        해당 순서로 메세지를 변환합니다.
        """

        path = Path()
        path.header = header

        joint_names = joint_trajectory.joint_trajectory.joint_names
        joint_order = [joint_names.index(joint) for joint in self.joint_order]

        position_keys = ["x", "y", "z"]
        orientation_keys = ["x", "y", "z", "w"]
        orientation = Quaternion(**dict(zip(orientation_keys, [0.0, 0.0, 0.0, 1.0])))

        for point in joint_trajectory.joint_trajectory.points:
            point: JointTrajectoryPoint

            joint_position = np.array(point.positions)
            joint_position = joint_position[joint_order]

            eef_pose = ForwardKinematics.forward_kinematics(joint_position)[
                :3, 3
            ]  # End Effector Pose, 길이 3 벡터

            eef_pose_stamped = PoseStamped(
                header=header,
                pose=Pose(
                    position=Point(**dict(zip(position_keys, eef_pose))),
                    orientation=orientation,
                ),
            )
            path.poses.append(eef_pose_stamped)

        return path


class SRVClient(object):
    def __init__(self, node: Node, service_name: str, service_type: type):
        self._node = node

        self._service_name = service_name
        self._service_type = service_type

        self._srv = self._node.create_client(self._service_type, self._service_name)
        self._future: Future = None
        self._response = service_type.Response()

        self._is_finished = True

        self._error_code = {
            "NOT_INITIALIZED": 0,
            "SUCCESS": 1,
            "FAILURE": 99999,
            "PLANNING_FAILED": -1,
            "INVALID_MOTION_PLAN": -2,
            "MOTION_PLAN_INVALIDATED_BY_ENVIRONMENT_CHANGE": -3,
            "CONTROL_FAILED": -4,
            "UNABLE_TO_AQUIRE_SENSOR_DATA": -5,
            "TIMED_OUT": -6,
            "PREEMPTED": -7,
            "START_STATE_IN_COLLISION": -10,
            "START_STATE_VIOLATES_PATH_CONSTRAINTS": -11,
            "START_STATE_INVALID": -26,
            "GOAL_IN_COLLISION": -12,
            "GOAL_VIOLATES_PATH_CONSTRAINTS": -13,
            "GOAL_CONSTRAINTS_VIOLATED": -14,
            "GOAL_STATE_INVALID": -27,
            "UNRECOGNIZED_GOAL_TYPE": -28,
            "INVALID_GROUP_NAME": -15,
            "INVALID_GOAL_CONSTRAINTS": -16,
            "INVALID_ROBOT_STATE": -17,
            "INVALID_LINK_NAME": -18,
            "INVALID_OBJECT_NAME": -19,
            "FRAME_TRANSFORM_FAILURE": -21,
            "COLLISION_CHECKING_UNAVAILABLE": -22,
            "ROBOT_STATE_STALE": -23,
            "SENSOR_INFO_STALE": -24,
            "COMMUNICATION_FAILURE": -25,
            "CRASH": -29,
            "ABORT": -30,
            "NO_IK_SOLUTION": -31,
        }

        while not self._srv.wait_for_service(timeout_sec=1.0):
            self._node.get_logger().info(
                f"Service {service_name} not available, waiting again..."
            )

    def send_request(self, request):
        if self.is_finished():

            self._future: Future = self._srv.call_async(request)
            self._future.add_done_callback(self.callback)

            self._is_finished = False

        else:
            self._node.get_logger().warn(
                f"Service {self._service_name} is not finished yet."
            )

    def callback(self, future: Future):
        self._response = future.result()

    def is_finished(self):
        if self._future is None:
            return True

        if self._future.done():
            self._future = None
            return True

        return False

    @property
    def response(self):
        return self._response

    def get_error_code(self, code: int):
        for key, value in self._error_code.items():
            if value == code:
                return key

        return "UNKNOWN"


class MoveitClient(Node):
    """
    /move_group

    Subscribers:
        /parameter_events: rcl_interfaces/msg/ParameterEvent
        /trajectory_execution_event: std_msgs/msg/String

    Publishers:
        /display_contacts: visualization_msgs/msg/MarkerArray
        /display_planned_path: moveit_msgs/msg/DisplayTrajectory
        /motion_plan_request: moveit_msgs/msg/MotionPlanRequest
        /parameter_events: rcl_interfaces/msg/ParameterEvent
        /robot_description_semantic: std_msgs/msg/String
        /rosout: rcl_interfaces/msg/Log

    Service Servers:
        /apply_planning_scene: moveit_msgs/srv/ApplyPlanningScene
        /check_state_validity: moveit_msgs/srv/GetStateValidity
        /clear_octomap: std_srvs/srv/Empty
        /compute_cartesian_path: moveit_msgs/srv/GetCartesianPath
        /compute_fk: moveit_msgs/srv/GetPositionFK
        /compute_ik: moveit_msgs/srv/GetPositionIK
        /get_planner_params: moveit_msgs/srv/GetPlannerParams
        /load_map: moveit_msgs/srv/LoadMap
        /move_group/describe_parameters: rcl_interfaces/srv/DescribeParameters
        /move_group/get_parameter_types: rcl_interfaces/srv/GetParameterTypes
        /move_group/get_parameters: rcl_interfaces/srv/GetParameters
        /move_group/list_parameters: rcl_interfaces/srv/ListParameters
        /move_group/set_parameters: rcl_interfaces/srv/SetParameters
        /move_group/set_parameters_atomically: rcl_interfaces/srv/SetParametersAtomically
        /plan_kinematic_path: moveit_msgs/srv/GetMotionPlan
        /query_planner_interface: moveit_msgs/srv/QueryPlannerInterfaces
        /save_map: moveit_msgs/srv/SaveMap
        /set_planner_params: moveit_msgs/srv/SetPlannerParams

    Service Clients:
        None

    Action Servers:
        /execute_trajectory: moveit_msgs/action/ExecuteTrajectory
        /move_action: moveit_msgs/action/MoveGroup

    Action Clients:
        None

    """

    def __init__(self):
        super().__init__("ur5e_control_node")

        # >>> Variables >>>
        self.planning_scene = None
        self.joint_states = None
        self.fk = ForwardKinematics()
        # <<< Variables <<<

        # >>> ROS Subscribers >>>
        self.joint_states_subscriber = self.create_subscription(
            JointState,
            "/joint_states",
            self.joint_states_callback,
            qos_profile_system_default,
        )
        # <<< ROS Subscribers <<<

        # >>> ROS Publishers >>>
        self.end_effector_pose_publisher = self.create_publisher(
            PoseStamped,
            self.get_name() + "/end_effector/pose",
            qos_profile_system_default,
        )

        self.end_effector_path_publisher = self.create_publisher(
            Path, self.get_name() + "/end_effector/path", qos_profile_system_default
        )
        # <<< ROS Publishers <<<

        # >>> Service Clients >>>
        self.apply_planning_scene_client = SRVClient(
            self, "/apply_planning_scene", ApplyPlanningScene
        )
        self.compute_cartesian_path_client = SRVClient(
            self, "/compute_cartesian_path", GetCartesianPath
        )
        self.compute_fk_client = SRVClient(self, "/compute_fk", GetPositionFK)
        self.compute_ik_client = SRVClient(self, "/compute_ik", GetPositionIK)
        self.plan_kinematic_path_client = SRVClient(
            self, "/plan_kinematic_path", GetMotionPlan
        )
        self.get_planning_scene_client = SRVClient(
            self, "/get_planning_scene", GetPlanningScene
        )
        # <<< Service Clients <<<

        # >>> Main Loop >>>
        while True:
            if self.joint_states is not None:
                break
            self.get_logger().warn("Waiting for joint_states...")
            rclpy.spin_once(self, timeout_sec=0.1)

        self.initialize_world()

        self.get_logger().info("Service is available.")

        self.timer = self.create_timer(1.0, self.run)
        # <<< Main Loop <<<

    # >>> Main Loop >>>
    def run(self):
        header = Header(
            stamp=self.get_clock().now().to_msg(),
            frame_id="base_link",
        )

        # collision_object: CollisionObject = MoveitClient.create_collision_object(
        #     id="box",
        #     header=header,
        #     pose=Pose(
        #         position=Point(x=0.5, y=0.5, z=0.5),
        #         orientation=Quaternion(x=0.0, y=0.0, z=0.0, w=1.0),
        #     ),
        #     scale=Vector3(x=0.1, y=0.1, z=0.1),
        #     operation=CollisionObject.ADD,
        # )

        # self.apply_planning_scene(collision_objects=[collision_object])

        orientation = Quaternion(
            x=2.00307048711234e-16,
            y=0.7071067812590626,
            z=0.7071067811140325,
            w=-4.329780280011333e-17,
        )

        waypoints = [
            Pose(
                position=Point(x=0.4, y=0.4, z=0.163),
                orientation=orientation,
            ),
        ]

        if self.compute_fk():
            pose_stamped = self.handle_compute_fk_response()

        if self.compute_cartesian_path(header=header, waypoints=waypoints):
            cartesian_path = self.handle_cartesian_path_response(header=header)

    # <<< Main Loop <<<

    # >>> Handle Responses >>>
    def handle_apply_planning_scene_response(self):
        pass

    def handle_cartesian_path_response(self, header: Header) -> Path:
        cartesian_path_response: GetCartesianPath.Response = (
            self.compute_cartesian_path_client.response
        )

        code = cartesian_path_response.error_code.val
        if code != MoveItErrorCodes.SUCCESS:
            code_type = self.compute_cartesian_path_client.get_error_code(code)
            self.get_logger().warn(
                f"Error code in compute_cartesian_path service: {code}/{code_type}"
            )
            return None

        trajectory: RobotTrajectory = cartesian_path_response.solution
        fraction = cartesian_path_response.fraction

        if fraction < 1.0:
            self.get_logger().warn(f"Fraction is not 1.0: {fraction}")
            return None

        eef_path: Path = self.fk.parse_robot_trajectory_to_path(
            header=header, joint_trajectory=trajectory
        )
        self.end_effector_path_publisher.publish(eef_path)

        return eef_path

    def handle_compute_fk_response(self) -> PoseStamped:
        response: GetPositionFK.Response = self.compute_fk_client.response

        code = response.error_code.val
        if code != MoveItErrorCodes.SUCCESS:
            code_type = self.compute_fk_client.get_error_code(code)
            self.get_logger().warn(
                f"Error code in compute_fk service: {code}/{code_type}"
            )
            return None

        pose_stamped: PoseStamped = response.pose_stamped[0]
        self.end_effector_pose_publisher.publish(pose_stamped)

        return pose_stamped

    def handle_compute_ik_response(self) -> RobotState:
        response: GetPositionIK.Response = self.compute_ik_client.response

        code = response.error_code.val
        if code != MoveItErrorCodes.SUCCESS:
            code_type = self.compute_ik_client.get_error_code(code)
            self.get_logger().warn(
                f"Error code in compute_ik service: {code}/{code_type}"
            )
            return None

        solution: RobotState = response.solution
        return solution

    def handle_plan_kinematic_path_response(self):
        response: GetMotionPlan.Response = self.plan_kinematic_path_client.response
        response_msg: MotionPlanResponse = response.motion_plan_response

        code = response_msg.error_code.val
        if code != MoveItErrorCodes.SUCCESS:
            code_type = self.plan_kinematic_path_client.get_error_code(code)
            self.get_logger().warn(
                f"Error code in plan_kinematic_path service: {code}/{code_type}"
            )
            return None

        trajectory: RobotTrajectory = response_msg.trajectory
        planning_time: float = response_msg.planning_time

        header = Header(
            stamp=self.get_clock().now().to_msg(),
            frame_id="base_link",
        )

        path: Path = self.fk.parse_robot_trajectory_to_path(
            header=header, joint_trajectory=trajectory
        )
        self.end_effector_path_publisher.publish(path)

        self.get_logger().info(
            f"Planning time: {planning_time}, Path length: {len(path.poses)}"
        )

        return path

    def handle_get_planning_scene_response(self):
        response: GetPlanningScene.Response = self.get_planning_scene_client.response

        scene = response.scene

        if scene.name != "":
            self.get_logger().info("Planning scene is updated.")
            return scene

        else:
            self.get_logger().warn("Planning scene is not updated.")
            return None

    # <<< Handle Responses >>><<<

    # >>> Service Clients >>>
    def apply_planning_scene(self, collision_objects: list):
        """
        Update the planning scene with the collision object.
        """
        if (
            self.joint_states is None
            or not self.apply_planning_scene_client.is_finished
        ):
            return False

        for collision_object in collision_objects:
            collision_object: CollisionObject

            is_duplicated = self.is_object_duplicated(collision_object)
            if not is_duplicated:
                self.planning_scene.world.collision_objects.append(collision_object)

        request = ApplyPlanningScene.Request(scene=self.planning_scene)
        self.apply_planning_scene_client.send_request(request)

        return True

    def compute_cartesian_path(self, header: Header, waypoints: list):
        # Exception handling
        if (
            self.joint_states is None
            or not self.compute_cartesian_path_client.is_finished
        ):
            return False

        request = GetCartesianPath.Request(
            header=header,
            start_state=RobotState(joint_state=self.joint_states),
            group_name="manipulator",
            link_name="wrist_3_link",
            waypoints=waypoints,
            max_step=0.05,
            jump_threshold=5.0,
            avoid_collisions=True,
        )

        self.compute_cartesian_path_client.send_request(request)

        return True

    def compute_fk(self):
        # Exception handling
        if self.joint_states is None or not self.compute_fk_client.is_finished:
            return False

        request = GetPositionFK.Request(
            header=Header(
                stamp=self.get_clock().now().to_msg(),
                frame_id="base_link",
            ),
            fk_link_names=["tool0"],
            robot_state=RobotState(joint_state=self.joint_states),
        )
        self.compute_fk_client.send_request(request)

        return True

    def compute_ik(self, pose_stamped: PoseStamped):
        # Exception handling
        if self.joint_states is None or not self.compute_ik_client.is_finished:
            return False

        request = GetPositionIK.Request(
            ik_request=PositionIKRequest(
                group_name="manipulator",
                robot_state=RobotState(joint_state=self.joint_states),
                constraints=Constraints(),
                avoid_collisions=True,
                pose_stamped=pose_stamped,
            )
        )

        self.compute_ik_client.send_request(request)

        return True

    def plan_kinematic_path(self, goal_constraints: list):
        """
        Unused Parameters:
        - workspace_parameters
        - path_constraints
        - trajectory_constraints
        - reference_trajectories
        - pipeline_id
        - planner_id
        """

        # Exception handling
        if self.joint_states is None or not self.plan_kinematic_path_client.is_finished:
            return False

        request = GetMotionPlan.Request(
            motion_plan_request=MotionPlanRequest(
                start_state=RobotState(joint_state=self.joint_states),
                goal_constraints=goal_constraints,
                group_name="manipulator",
                num_planning_attempts=5,
                allowed_planning_time=0.5,
                max_velocity_scaling_factor=1.0,
                max_acceleration_scaling_factor=1.0,
                # cartesian_speed_limited_link="wrist_3_link",
                # max_cartesian_speed=0.1,
            )
        )

        self.plan_kinematic_path_client.send_request(request)

        return True

    def get_planning_scene(self):
        # Exception handling
        if not self.get_planning_scene_client.is_finished:
            return False

        request = GetPlanningScene.Request()
        self.get_planning_scene_client.send_request(request)

        return True

    # <<< Service Clients <<<

    # >>> Subscriber Callbacks >>>
    def joint_states_callback(self, msg: JointState):
        self.joint_states = msg

    # <<< Subscriber Callbacks <<<

    # >>> Methods >>>
    def is_object_duplicated(self, new_object: CollisionObject):
        """
        Compare the id of the new object with the existing objects.
        If the id is the same, it is considered a duplicate.
        """
        current_world = self.planning_scene.world

        current_objects = current_world.collision_objects

        is_duplicated = False
        for i, obj in enumerate(current_objects):
            obj: CollisionObject

            if obj.id == new_object.id:
                new_object.operation = CollisionObject.MOVE
                is_duplicated = True
                break

        return is_duplicated

    def reset_world(self):
        # >>> STEP 1. Remove all collision objects >>>
        collision_objects = self.planning_scene.world.collision_objects

        for obj in collision_objects:
            obj: CollisionObject
            obj.operation = CollisionObject.REMOVE

        self.planning_scene.world.collision_objects = collision_objects
        # <<< STEP 1. Remove all collision objects <<<

        self.get_logger().info("Reset the planning scene.")

        request = ApplyPlanningScene.Request(scene=self.planning_scene)
        self.apply_planning_scene_client.send_request(request)

        return True

    def initialize_world(self):
        while True:
            self.get_logger().info("Initializing the planning scene...")

            if self.planning_scene is None:

                # Get the planning scene from the server
                is_get_planning_scene_success = self.get_planning_scene()

                if is_get_planning_scene_success:
                    scene = self.handle_get_planning_scene_response()

                    if scene is not None:
                        self.planning_scene = scene
                        # self.planning_scene.is_diff = True

                        self.reset_world()
                        return True

            # If the planning scene is already initialized
            else:
                self.get_logger().info("Planning scene is already initialized.")
                return True

            self.get_logger().warn(
                "Failed to initialize the planning scene... Waiting..."
            )
            rclpy.spin_once(self, timeout_sec=0.1)

    # <<< Methods <<<

    # >>> Static Methods >>>
    @staticmethod
    def parse_robot_state_to_constraints(robot_state: RobotState):
        joint_state = robot_state.joint_state

        joint_constraints = []
        for name, position in zip(joint_state.name, joint_state.position):
            joint_constraint = JointConstraint(
                joint_name=name,
                position=position,
                weight=1.0,
                # tolerance_above=0.1,
                # tolerance_below=0.1,
            )
            joint_constraints.append(joint_constraint)

        constraints = Constraints(
            joint_constraints=joint_constraints,
        )
        return constraints

    @staticmethod
    def create_collision_object(
        id: str, header: Header, pose: Pose, scale: Vector3, operation
    ):
        """
        Primitive Type:
        - 0: CollisionObject.ADD
        - 1: CollisionObject.REMOVE
        - 2: CollisionObject.APPEND
        - 3: CollisionObject.MOVE
        """
        if isinstance(operation, int):
            operation = int.to_bytes(operation, byteorder="big")
        elif isinstance(operation, bytes):
            operation = operation
        else:
            raise TypeError("operation must be int or bytes")

        # 장애물 객체 생성
        return CollisionObject(
            id=id,
            header=header,
            operation=operation,
            primitives=[
                SolidPrimitive(
                    type=SolidPrimitive.BOX, dimensions=[scale.x, scale.y, scale.z]
                ),
            ],
            primitive_poses=[pose],
        )

    # <<< Static Methods <<<


def main():
    rclpy.init(args=None)

    node = MoveitClient()

    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
