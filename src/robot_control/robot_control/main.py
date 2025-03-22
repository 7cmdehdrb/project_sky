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
from custom_msgs.msg import *
from moveit_msgs.msg import *
from trajectory_msgs.msg import *
from moveit_msgs.srv import *
from builtin_interfaces.msg import Duration as BuiltinDuration

# TF
from tf2_ros import *

# Python
import numpy as np
from enum import Enum

# custom
from base_package.header import QuaternionAngle
from robot_control.ur5e_control import MoveitClient
from object_tracker.object_pose_estimator_client import MegaPoseClient


class State(Enum):
    GRASPING_AIM = 0
    GRASPING = 1
    HOME_AIM = 2
    HOME = 3


class MainControlNode(object):
    def __init__(self, node: Node):
        self._node = node

        self._moveit_client = MoveitClient(node=self._node)
        # self._megapose_client = MegaPoseClient(node=self._node)
        self.state = State.GRASPING_AIM

        self.test = self._node.create_publisher(
            PoseStamped,
            "test",
            qos_profile=qos_profile_system_default,
        )

        self.home = None
        self.idx = 0

        self._node.get_logger().info("Home Pose: {}".format(self.home))

    def initialize(self):
        self._moveit_client.initialize_world()

    def get_test_data(self) -> BoundingBox3DMultiArray:
        result = BoundingBox3DMultiArray()

        box = BoundingBox3D()
        box.cls = "test1"
        box.pose = Pose(
            position=Point(x=1.0, y=0.0, z=0.2),
            orientation=Quaternion(x=0.0, y=0.0, z=0.0, w=1.0),
        )
        box.scale = Vector3(x=0.05, y=0.05, z=0.12)
        result.data.append(box)

        box = BoundingBox3D()
        box.cls = "test2"
        box.pose = Pose(
            position=Point(x=0.9, y=0.15, z=0.2),
            orientation=Quaternion(x=0.0, y=0.0, z=0.0, w=1.0),
        )
        box.scale = Vector3(x=0.05, y=0.05, z=0.12)
        result.data.append(box)

        box = BoundingBox3D()
        box.cls = "test3"
        box.pose = Pose(
            position=Point(x=0.9, y=-0.15, z=0.2),
            orientation=Quaternion(x=0.0, y=0.0, z=0.0, w=1.0),
        )
        box.scale = Vector3(x=0.05, y=0.05, z=0.12)
        result.data.append(box)

        return result

    def run(self):
        # Get All Object Poses
        # object_poses: BoundingBox3DMultiArray = (
        #     self._megapose_client.send_megapose_request()
        # )
        # if len(object_poses.data) == 0:
        #     self._node.get_logger().warn("No object detected.")
        #     return None

        if self.home is None:
            fk_response: GetPositionFK.Response = self._moveit_client.compute_fk()
            if fk_response is not None:
                self.home = self._moveit_client.handle_compute_fk_response(
                    response=fk_response
                ).pose
            self._node.get_logger().info("Home Pose: {}".format(self.home))
            return False

        self._node.get_logger().info(f"Running State: {self.state}")

        object_poses = self.get_test_data()

        # # Update Planning Scene
        collision_objects = self.collision_object_from_bbox_3d(bbox_3d=object_poses)

        self._moveit_client.apply_planning_scene(collision_objects=collision_objects)

        # Plan to Target
        target: BoundingBox3D = object_poses.data[self.idx]
        target_pose = Pose(
            position=Point(
                x=target.pose.position.x - (target.scale.x / 2.0) - 0.12,
                y=target.pose.position.y,
                z=target.pose.position.z,
            ),
            orientation=Quaternion(x=0.5, y=-0.5, z=0.5, w=-0.5),
        )

        header = Header(
            stamp=self._node.get_clock().now().to_msg(), frame_id="base_link"
        )

        self.test.publish(PoseStamped(header=header, pose=target_pose))

        # True: Cartesian Path, False: Kinematic Path
        planning_method = False

        if planning_method:
            self._node.get_logger().info("Cartesian Path")

            cartesian_path_response: GetCartesianPath.Response = (
                self._moveit_client.compute_cartesian_path(
                    header=header,
                    waypoints=[target_pose],
                )
            )

            if cartesian_path_response is not None:
                cartesian_path: RobotTrajectory = (
                    self._moveit_client.handle_cartesian_path_response(
                        header=header,
                        response=cartesian_path_response,
                        fraction_threshold=1.0,
                    )
                )

                if cartesian_path is not None:
                    scaled_trajectory = MainControlNode.scale_trajectory(
                        scale_factor=0.5, trajectory=cartesian_path
                    )
                    self._moveit_client.execute_trajectory(trajectory=scaled_trajectory)
                    print("Cartesian Path Executed.")
                    return False

        else:
            if self.state == State.GRASPING_AIM:
                temperal_target_pose = target_pose
                temperal_target_pose.position.x -= 0.1

            elif self.state == State.GRASPING:
                temperal_target_pose = target_pose

            elif self.state == State.HOME_AIM:
                temperal_target_pose = target_pose
                temperal_target_pose.position.x -= 0.2

            elif self.state == State.HOME:
                temperal_target_pose = self.home

            if self.state == State.GRASPING_AIM or self.state == State.HOME_AIM:
                tol = 0.05
            else:
                tol = 0.02

            goal_constrain = self._moveit_client.get_goal_constraint(
                pose_stamped=PoseStamped(header=header, pose=temperal_target_pose),
                orientation_constraints=None,
                tolerance=tol,
            )
            if goal_constrain is None:
                self._node.get_logger().warn("Goal Constrain is None.")
                return False

            if goal_constrain is not None:
                kinematic_path_response: GetMotionPlan.Response = (
                    self._moveit_client.plan_kinematic_path(
                        goal_constraints=[goal_constrain]
                    )
                )

                if kinematic_path_response is None:
                    self._node.get_logger().warn("Kinematic Path Response is None.")
                    return False

                if kinematic_path_response is not None:
                    kinematic_path: RobotTrajectory = (
                        self._moveit_client.handle_plan_kinematic_path_response(
                            header=header, response=kinematic_path_response
                        )
                    )

                    if kinematic_path is None:
                        self._node.get_logger().warn("Kinematic Path is None.")
                        return False

                    if kinematic_path is not None:
                        scaled_trajectory = MainControlNode.scale_trajectory(
                            scale_factor=0.5, trajectory=kinematic_path
                        )
                        self._moveit_client.execute_trajectory(
                            trajectory=scaled_trajectory
                        )
                        print("Kinematic Path Executed.")

                        if self.state == State.GRASPING_AIM:
                            self.state = State.GRASPING

                        elif self.state == State.GRASPING:
                            self.state = State.HOME_AIM

                        elif self.state == State.HOME_AIM:
                            self.state = State.HOME

                        elif self.state == State.HOME:
                            self.state = State.GRASPING_AIM
                            self.idx += 1
                            if self.idx >= 3:
                                self.idx = 0

                            self._moveit_client.initialize_world()

                        return False

        return False

    def collision_object_from_bbox_3d(self, bbox_3d: BoundingBox3DMultiArray) -> list:
        header = Header(
            stamp=self._node.get_clock().now().to_msg(),
            frame_id="base_link",
        )
        collision_objects = []

        for bbox in bbox_3d.data:
            bbox: BoundingBox3D

            collision_object = MoveitClient.create_collision_object(
                id=bbox.cls,
                header=header,
                pose=bbox.pose,
                scale=bbox.scale,
                operation=CollisionObject.ADD,
            )
            collision_objects.append(collision_object)

        return collision_objects

    @staticmethod
    def scale_trajectory(trajectory: RobotTrajectory, scale_factor: float):
        def scale_duration(duration: BuiltinDuration, factor: float) -> BuiltinDuration:
            # 전체 시간을 나노초로 환산
            total_nanosec = duration.sec * 1_000_000_000 + duration.nanosec
            # n배
            scaled_nanosec = int(total_nanosec * factor)

            # 다시 sec과 nanosec으로 분리
            new_sec = scaled_nanosec // 1_000_000_000
            new_nanosec = scaled_nanosec % 1_000_000_000

            return BuiltinDuration(sec=new_sec, nanosec=new_nanosec)

        new_points = []
        for point in trajectory.joint_trajectory.points:
            point: JointTrajectoryPoint

            new_point = JointTrajectoryPoint(
                positions=np.array(point.positions),
                velocities=np.array(point.velocities) * scale_factor,
                accelerations=np.array(point.accelerations) * scale_factor,
                time_from_start=scale_duration(
                    duration=point.time_from_start, factor=(1.0 / scale_factor)
                ),
            )
            new_points.append(new_point)

        new_joint_trajectory = JointTrajectory(
            header=trajectory.joint_trajectory.header,
            joint_names=trajectory.joint_trajectory.joint_names,
            points=new_points,
        )

        new_trajectory = RobotTrajectory(
            joint_trajectory=new_joint_trajectory,
            multi_dof_joint_trajectory=trajectory.multi_dof_joint_trajectory,
        )

        return new_trajectory


def main():
    rclpy.init(args=None)

    node = Node("main_control_node")
    main_node = MainControlNode(node=node)

    # Spin in a separate thread
    thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    thread.start()

    hz = 0.5
    rate = node.create_rate(hz)

    main_node.initialize()

    flag = False

    # main_node.run()

    try:
        while rclpy.ok():
            if not flag:
                flag = main_node.run()

            rate.sleep()
    except KeyboardInterrupt:
        pass

    node.destroy_node()

    rclpy.shutdown()
    thread.join()


if __name__ == "__main__":
    main()
