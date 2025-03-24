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
from shape_msgs.msg import *
from builtin_interfaces.msg import Duration as BuiltinDuration

# TF
from tf2_ros import *

# Python
import numpy as np
from enum import Enum

# custom
from base_package.header import QuaternionAngle
from robot_control.ur5e_control import MoveitClient
from robot_control.gripper_action import GripperActionClient
from object_tracker.object_pose_estimator_client import MegaPoseClient


class State(Enum):
    WAITING = -1
    FCN_SEARCHING = 0
    TARGET_AIMING = 1
    TARGET_POSITIONING = 2
    GRASPING = 3
    HOME_AIMING = 4
    HOME_POSITIONING = 5
    UNGRASPING = 6


class MainControlNode(object):
    def __init__(self, node: Node):
        self._node = node

        self._moveit_client = MoveitClient(node=self._node)
        self._gripper_client = GripperActionClient(node=self._node)
        self._megapose_client = MegaPoseClient(node=self._node)
        self.state = State.WAITING

        self.home_pose = None
        self.end_effector = "gripper_link"
        self.target_obects: BoundingBox3DMultiArray = None

        self.test = self._node.create_publisher(
            PoseStamped,
            "test",
            qos_profile=qos_profile_system_default,
        )

    def initialize(self):
        self._moveit_client.initialize_world()

    def get_test_data(self) -> BoundingBox3DMultiArray:
        result = BoundingBox3DMultiArray()

        box = BoundingBox3D()
        box.cls = "test1"
        box.pose = Pose(
            position=Point(x=0.85, y=0.0, z=0.2),
            orientation=Quaternion(x=0.0, y=0.0, z=0.0, w=1.0),
        )
        box.scale = Vector3(x=0.05, y=0.05, z=0.12)
        result.data.append(box)

        box = BoundingBox3D()
        box.cls = "test2"
        box.pose = Pose(
            position=Point(x=0.75, y=0.15, z=0.2),
            orientation=Quaternion(x=0.0, y=0.0, z=0.0, w=1.0),
        )
        box.scale = Vector3(x=0.05, y=0.05, z=0.12)
        # result.data.append(box)

        box = BoundingBox3D()
        box.cls = "test3"
        box.pose = Pose(
            position=Point(x=0.75, y=-0.15, z=0.2),
            orientation=Quaternion(x=0.0, y=0.0, z=0.0, w=1.0),
        )
        box.scale = Vector3(x=0.05, y=0.05, z=0.12)
        # result.data.append(box)

        return result

    def run(self):
        header = Header(
            stamp=self._node.get_clock().now().to_msg(), frame_id="base_link"
        )
        self._node.get_logger().info(f"Running State: {self.state}")

        if self.state == State.WAITING:
            if self.waiting():
                self.state = State.FCN_SEARCHING
                return True
            else:
                self._node.get_logger().warn("Error in Waiting. Home Pose Not Set.")
                return False

        elif self.state == State.FCN_SEARCHING:
            self.target_obects = self.fcn_searching()
            if self.target_obects is not None:
                self.state = State.TARGET_AIMING
            else:
                self._node.get_logger().warn("Error in FCN Searching. No Object Found.")
                return False

        elif (
            self.state == State.TARGET_AIMING
            or self.state == State.TARGET_POSITIONING
            or self.state == State.HOME_AIMING
            or self.state == State.HOME_POSITIONING
        ):
            # TODO: Replace with Real Data
            orientation = Quaternion(x=0.5, y=0.5, z=0.5, w=0.5)
            target: BoundingBox3D = self.target_obects.data[0]
            scale_factor = 0.5
            tolerance = 0.05

            if self.state == State.TARGET_AIMING:
                target_pose = Pose(
                    position=Point(
                        x=target.pose.position.x - 0.2,
                        y=target.pose.position.y,
                        z=target.pose.position.z,
                    ),
                    orientation=orientation,
                )
                tolerance = 0.05

            elif self.state == State.TARGET_POSITIONING:
                target_pose = Pose(
                    position=target.pose.position,
                    orientation=orientation,
                )
                tolerance = 0.01

            elif self.state == State.HOME_AIMING:
                target_pose = Pose(
                    position=Point(
                        x=target.pose.position.x - 0.2,
                        y=target.pose.position.y,
                        z=target.pose.position.z,
                    ),
                    orientation=orientation,
                )
                tolerance = 0.01

            elif self.state == State.HOME_POSITIONING:
                target_pose = Pose(
                    position=self.home_pose.position,
                    orientation=orientation,
                )
                tolerance = 0.05

            self.test.publish(PoseStamped(header=header, pose=target_pose))

            if self.control(
                header=header,
                target_pose=target_pose,
                scale_factor=scale_factor,
                tolerance=tolerance,
            ):
                self.state = State(self.state.value + 1)
            else:
                self._node.get_logger().warn("Error in Path Planning. Control Failed.")
                return False

        elif self.state == State.GRASPING:
            # self.grasping(open=True)
            self.state = State.HOME_AIMING

        elif self.state == State.UNGRASPING:
            # self.grasping(open=False)
            self.state = State.FCN_SEARCHING

        else:
            self._node.get_logger().warn("Invalid State.")
            return False

        return True

    # >>> Methods for Each State
    def waiting(self):
        """
        Initialize Home Pose. Return True if Home Pose is set.
        """
        if self.home_pose is None:
            fk_response: GetPositionFK.Response = self._moveit_client.compute_fk(
                end_effector=self.end_effector
            )
            if fk_response is not None:
                self.home_pose = self._moveit_client.handle_compute_fk_response(
                    response=fk_response
                ).pose

                self._node.get_logger().info(
                    "Setting Home Pose: {}".format(self.home_pose)
                )
                return True

        return False

    def fcn_searching(self):
        self.initialize()

        # Get All Object Poses
        bbox_3d: BoundingBox3DMultiArray = self._megapose_client.send_megapose_request()
        # bbox_3d = self.get_test_data()

        if len(bbox_3d.data) == 0:
            self._node.get_logger().warn("No object detected.")
            return None

        # # Update Planning Scene
        collision_objects = self.collision_object_from_bbox_3d(
            header=Header(
                stamp=self._node.get_clock().now().to_msg(), frame_id="base_link"
            ),
            bbox_3d=bbox_3d,
        )
        self._moveit_client.apply_planning_scene(collision_objects=collision_objects)

        return bbox_3d

    def grasping(self, open: bool = True):
        self._gripper_client.control_gripper(open=open)

    def control(
        self, header: Header, target_pose: Pose, tolerance: float, scale_factor: float
    ):
        goal_constraints = self._moveit_client.get_goal_constraint(
            pose_stamped=PoseStamped(header=header, pose=target_pose),
            orientation_constraints=None,
            tolerance=tolerance,
            end_effector=self.end_effector,
        )
        path_constraint = Constraints(
            name="path_constraint",
            # position_constraints=[
            #     PositionConstraint(
            #         header=header,
            #         link_name=self.end_effector,
            #         # target_point_offset=Vector3(x=0.0, y=0.0, z=0.0),
            #         constraint_region=BoundingVolume(
            #             primitives=[
            #                 SolidPrimitive(
            #                     type=SolidPrimitive.BOX,
            #                     dimensions=[2.0, 1.0, 1.0],
            #                 )
            #             ],
            #             primitive_poses=[
            #                 Pose(
            #                     position=Point(x=1.0, y=0.5, z=0.5),
            #                     orientation=Quaternion(x=0.0, y=0.0, z=0.0, w=1.0),
            #                 )
            #             ],
            #             # meshes=[Mesh()],
            #             # mesh_poses=[Pose()],
            #         ),
            #         weight=1.0,
            #     )
            # ],
            orientation_constraints=[
                OrientationConstraint(
                    header=header,
                    link_name=self.end_effector,
                    orientation=self.home_pose.orientation,
                    absolute_x_axis_tolerance=0.1,
                    absolute_y_axis_tolerance=0.1,
                    absolute_z_axis_tolerance=0.1,
                    weight=1.0,
                )
            ],
        )

        if goal_constraints is not None:
            kinematic_path_response: GetMotionPlan.Response = (
                self._moveit_client.plan_kinematic_path(
                    goal_constraints=[goal_constraints],
                    path_constraints=None,
                )
            )

            if kinematic_path_response is not None:
                kinematic_path: RobotTrajectory = (
                    self._moveit_client.handle_plan_kinematic_path_response(
                        header=header, response=kinematic_path_response
                    )
                )

                if kinematic_path is not None:
                    scaled_trajectory = MainControlNode.scale_trajectory(
                        scale_factor=scale_factor, trajectory=kinematic_path
                    )
                    self._moveit_client.execute_trajectory(trajectory=scaled_trajectory)
                    return True

        return False

    # <<< Methods for Each State

    @staticmethod
    def collision_object_from_bbox_3d(
        header: Header, bbox_3d: BoundingBox3DMultiArray
    ) -> list:
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

    hz = 1.0
    rate = node.create_rate(hz)

    main_node.initialize()

    try:
        while rclpy.ok():
            main_node.run()

            rate.sleep()
    except KeyboardInterrupt:
        pass

    node.destroy_node()

    rclpy.shutdown()
    thread.join()


if __name__ == "__main__":
    main()
