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
from custom_msgs.srv import *
from moveit_msgs.msg import *
from trajectory_msgs.msg import *
from moveit_msgs.srv import *
from shape_msgs.msg import *
from builtin_interfaces.msg import Duration as BuiltinDuration
from tf2_geometry_msgs.tf2_geometry_msgs import PoseStamped as TF2PoseStamped

# TF
from tf2_ros import *

# Python
import numpy as np
from enum import Enum
import json

# custom
from base_package.manager import ObjectManager, TransformManager
from fcn_network.fcn_integration_server import FCN_Integration_Client_Manager
from object_tracker.megapose_client import MegaPoseClient
from robot_control.control_manager import (
    GripperActionManager,
    FK_ServiceManager,
    IK_ServiceManager,
    GetPlanningScene_ServiceManager,
    ApplyPlanningScene_ServiceManager,
    CartesianPath_ServiceManager,
    KinematicPath_ServiceManager,
    ExecuteTrajectory_ServiceManager,
    JointStatesManager,
    ObjectSelectionManager,
)


class State(Enum):
    WAITING = -1
    MEGAPOSE_SEARCHING = 0
    FCN_SEARCHING = 1
    GRASPING_POSITIONING = 2
    TARGET_AIMING = 3
    TARGET_POSITIONING = 4
    GRASPING = 5
    HOME_AIMING = 6
    HOME_POSITIONING = 7
    DROP_POSITIONING = 8
    UNGRASPING = 9
    FCN_POSITIONING = 10


class MainControlNode(object):
    def __init__(self, node: Node, *args, **kwargs):
        self._node = node

        # >>> Managers >>>
        self._transform_manager = TransformManager(node=self._node, *args, **kwargs)
        self._joint_states_manager = JointStatesManager(
            node=self._node, *args, **kwargs
        )

        self._fcn_integration_client_manager = FCN_Integration_Client_Manager(
            node=self._node, *args, **kwargs
        )

        self._object_manager = ObjectManager(node=self._node, *args, **kwargs)
        self._object_selection_manager = ObjectSelectionManager(node=self._node)

        self._gripper_action_manager = GripperActionManager(
            node=self._node, *args, **kwargs
        )

        self._fk_service_manager = FK_ServiceManager(node=self._node, *args, **kwargs)
        self._ik_service_manager = IK_ServiceManager(node=self._node, *args, **kwargs)
        self._get_planning_scene_service_manager = GetPlanningScene_ServiceManager(
            node=self._node, *args, **kwargs
        )
        self._apply_planning_scene_service_manager = ApplyPlanningScene_ServiceManager(
            node=self._node, *args, **kwargs
        )
        self._cartesian_path_service_manager = CartesianPath_ServiceManager(
            node=self._node, *args, **kwargs
        )
        self._kinematic_path_service_manager = KinematicPath_ServiceManager(
            node=self._node, *args, **kwargs
        )
        self._execute_trajectory_service_manager = ExecuteTrajectory_ServiceManager(
            node=self._node, *args, **kwargs
        )

        self._megapose_client = MegaPoseClient(node=self._node, *args, **kwargs)
        # <<< Managers <<<

        self.target_cls = kwargs["target_cls"]
        self.state = State.WAITING

        # >>> Unique Joint States >>>

        self.home_joints = JointState(
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
                4.695071220397949,
                3.1019468307495117,
                -1.616389576588766,
            ],
        )
        self.dropping_joints = JointState(
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
                4.695071220397949,
                3.1019468307495117,
                1.1181697845458984,
            ],
        )
        self.waiting_joints = JointState(
            name=[
                "shoulder_lift_joint",
                "elbow_joint",
                "wrist_1_joint",
                "wrist_2_joint",
                "wrist_3_joint",
                "shoulder_pan_joint",
            ],
            position=[
                -0.21154792726550298,
                -1.1941368579864502,
                -3.0269323788084925,
                4.695095539093018,
                3.101895332336426,
                -0.006377998982564748,
            ],
        )

        # <<< Unique Joint States <<<

    def run(self):
        header = Header(stamp=self._node.get_clock().now().to_msg(), frame_id="world")
        self._node.get_logger().info(f"Running State: {self.state}")

        # >>> LEVEL 0. Waiting for initialization
        if self.state == State.WAITING:
            if self.waiting():
                self.state = State.MEGAPOSE_SEARCHING
                return True
            else:
                self._node.get_logger().warn("Error in Waiting. Home Pose Not Set.")
                return False

        elif self.state == State.MEGAPOSE_SEARCHING:
            self.fcn_result = [None, None]
            self.target_objects = self.fcn_searching()
            if self.target_objects is not None:
                self.state = State.FCN_SEARCHING
                return True
            else:
                self._node.get_logger().warn(
                    "Error in Megapose Searching. No Object Found."
                )
                return False

        elif self.state == State.FCN_SEARCHING:
            self._node.get_logger().info(f"Requesting Target Class: {self.target_cls}")
            self.test_pub.publish(String(data=self.target_cls))

            row, col = self.fcn_result
            if row is not None and col is not None:
                self.target_object = self._object_selector.get_target_object(
                    row=row, col=int(col), target_objects=self.target_objects
                )

                if self.target_object is not None:
                    self.state = State.GRASPING_POSITIONING
                    self.fcn_result = [None, None]
                    return True

                else:
                    self._node.get_logger().warn(
                        "Error in FCN Searching. No Target Object Found."
                    )
                    return False

            else:
                self._node.get_logger().warn("Error in FCN Searching. No FCN data.")

            return False

        elif (
            self.state == State.GRASPING_POSITIONING
            or self.state == State.FCN_POSITIONING
            or self.state == State.DROP_POSITIONING
        ):
            if self.state == State.GRASPING_POSITIONING:
                if self.control(
                    header=header,
                    target_pose=Pose(),  # None
                    scale_factor=0.4,
                    tolerance=0.01,
                    joint_states=self.home_joints,
                ):
                    self.state = State.TARGET_AIMING
                    return True
                else:
                    self._node.get_logger().warn(
                        "Error in GRASPING_POSITIONING. Control Failed."
                    )
                    return False

            elif self.state == State.FCN_POSITIONING:
                if self.control(
                    header=header,
                    target_pose=Pose(),  # None
                    scale_factor=0.4,
                    tolerance=0.01,
                    joint_states=self.waiting_joints,
                ):
                    self.state = State.MEGAPOSE_SEARCHING
                    return True
                else:
                    self._node.get_logger().warn(
                        "Error in FCN_POSITIONING. Control Failed."
                    )
                    return False

            elif self.state == State.DROP_POSITIONING:
                if self.control(
                    header=header,
                    target_pose=Pose(),  # None
                    scale_factor=0.4,
                    tolerance=0.01,
                    joint_states=self.dropping_joints,
                ):
                    self.state = State.UNGRASPING
                    return True
                else:
                    self._node.get_logger().warn(
                        "Error in DROP_POSITIONING. Control Failed."
                    )
                    return False

        elif (
            self.state == State.GRASPING_POSITIONING
            or self.state == State.FCN_POSITIONING
        ):
            if self.state == State.GRASPING_POSITIONING:
                if self.control(
                    header=header,
                    target_pose=Pose(),
                    scale_factor=0.2,
                    tolerance=0.02,
                    joint_states=self.home_joints,
                ):
                    self.state = State.TARGET_AIMING

            elif self.state == State.FCN_POSITIONING:
                if self.control(
                    header=header,
                    target_pose=Pose(),
                    scale_factor=0.2,
                    tolerance=0.02,
                    joint_states=self.waiting_joints,
                ):
                    self.state = State.FCN_SEARCHING

        elif (
            self.state == State.TARGET_AIMING
            or self.state == State.TARGET_POSITIONING
            or self.state == State.HOME_AIMING
            or self.state == State.HOME_POSITIONING
        ):
            # Target which is transformed to world frame
            target: BoundingBox3D = self.target_object
            scale_factor = 0.5
            tolerance = 0.05

            if self.state == State.TARGET_AIMING:
                self.grasping(open=True)
                target_pose = Pose(
                    position=Point(
                        x=target.pose.position.x,
                        y=target.pose.position.y - 0.2,
                        z=target.pose.position.z,
                    ),
                    orientation=self.home_pose.orientation,
                )
                scale_factor = 0.5
                tolerance = 0.04

            elif self.state == State.TARGET_POSITIONING:
                # self.temperal_reset()
                target_pose = Pose(
                    position=Point(
                        x=target.pose.position.x,
                        y=target.pose.position.y,
                        z=target.pose.position.z,  # Offset 2 cm below
                    ),
                    orientation=self.home_pose.orientation,
                )
                scale_factor = 0.2
                tolerance = 0.02

            elif self.state == State.HOME_AIMING:
                # self.temperal_reset()
                target_pose = Pose(
                    position=Point(
                        x=target.pose.position.x,
                        y=target.pose.position.y - 0.2,
                        z=target.pose.position.z + 0.05,  # Offset 5 cm above
                    ),
                    orientation=self.home_pose.orientation,
                )
                scale_factor = 0.5
                tolerance = 0.04

            elif self.state == State.HOME_POSITIONING:
                # self.temperal_reset()
                target_pose = Pose(
                    position=Point(
                        x=self.home_pose.position.x,
                        y=self.home_pose.position.y,
                        z=self.home_pose.position.z + 0.05,  # Offset 5 cm above
                    ),
                    orientation=self.home_pose.orientation,
                )
                scale_factor = 0.5
                tolerance = 0.04

            self.target_pose_publisher.publish(
                PoseStamped(header=header, pose=target_pose)
            )

            if self.control(
                header=header,
                target_pose=target_pose,
                scale_factor=scale_factor,
                tolerance=tolerance,
            ):
                self.state = State(self.state.value + 1)
                return True
            else:
                self._node.get_logger().warn("Error in Path Planning. Control Failed.")
                return False

        elif self.state == State.GRASPING:
            self.grasping(open=False)
            if self.gripper_joint_states is not None:
                if self.gripper_joint_states.position[0] > 0.05:
                    self.state = State.HOME_AIMING
                    return True

        elif self.state == State.UNGRASPING:
            self.grasping(open=True)
            if self.gripper_joint_states is not None:
                if self.gripper_joint_states.position[0] < 0.05:
                    self.state = State.FCN_POSITIONING

        else:
            self._node.get_logger().warn("Invalid State.")
            return False

        return True

    # >>> Methods for Each State
    def waiting(self):
        """
        Initialize Home Pose. Return True if Home Pose is set.
        """
        success1, success2 = False, False

        if self.home_pose is None:
            fk_response: GetPositionFK.Response = self._moveit_client.compute_fk(
                end_effector=self.end_effector, joint_states=self.home_joints
            )
            if fk_response is not None:
                self.home_pose = self._moveit_client.handle_compute_fk_response(
                    response=fk_response
                ).pose

                self._node.get_logger().info(
                    "Setting Home Pose: {}".format(self.home_pose)
                )
                success1 = True

        if self.drop_pose is None:
            fk_response: GetPositionFK.Response = self._moveit_client.compute_fk(
                end_effector=self.end_effector, joint_states=self.dropping_joints
            )
            if fk_response is not None:
                self.drop_pose = self._moveit_client.handle_compute_fk_response(
                    response=fk_response
                ).pose

                self._node.get_logger().info(
                    "Setting Home Pose: {}".format(self.drop_pose)
                )
                success2 = True

        return success1 and success2

    def fcn_searching(self):
        self.initialize()

        # Get All Object Poses
        bbox_3d: BoundingBox3DMultiArray = self._megapose_client.send_megapose_request()
        self._megapose_client.post_process_response(
            response=bbox_3d,
            header=Header(
                frame_id="camera1_link", stamp=self._node.get_clock().now().to_msg()
            ),
        )

        if len(bbox_3d.data) == 0:
            self._node.get_logger().warn("No object detected.")
            return None

        # Update Planning Scene
        transformed_bbox_3d = self.transform_bbox3d(
            bbox_3d=bbox_3d, target_frame="world"
        )
        if transformed_bbox_3d is None:
            return None

        transformed_bbox_3d = self.append_default_collision_objects(transformed_bbox_3d)

        collision_objects = self.collision_object_from_bbox_3d(
            header=Header(
                stamp=self._node.get_clock().now().to_msg(), frame_id="world"
            ),
            bbox_3d=transformed_bbox_3d,
        )
        self._moveit_client.apply_planning_scene(collision_objects=collision_objects)

        return bbox_3d

    def append_default_collision_objects(
        self, transformed_bbox_3d: BoundingBox3DMultiArray
    ):
        transformed_bbox_3d.data.append(
            BoundingBox3D(
                id=999,
                cls="camera_box",
                pose=Pose(
                    position=Point(x=-0.04, y=-0.39, z=0.3),
                    orientation=Quaternion(x=0.0, y=0.0, z=0.0, w=1.0),
                ),
                scale=Vector3(x=0.15, y=0.06, z=0.6),
            )
        )

        # Add Plane Box
        transformed_bbox_3d.data.append(
            BoundingBox3D(
                id=998,
                cls="plane_box",
                pose=Pose(
                    position=Point(x=0.0, y=0.0, z=-0.5 - 0.01),
                    orientation=Quaternion(x=0.0, y=0.0, z=0.0, w=1.0),
                ),
                scale=Vector3(x=0.8, y=0.44, z=1.0),
            )
        )

        # Add Shelf Box
        transformed_bbox_3d.data.append(
            BoundingBox3D(
                id=997,
                cls="shelf_box1",
                pose=Pose(
                    position=Point(x=0.0, y=0.6, z=0.23),
                    orientation=Quaternion(x=0.0, y=0.0, z=0.0, w=1.0),
                ),
                scale=Vector3(x=0.8, y=0.44, z=0.04),
            )
        )

        transformed_bbox_3d.data.append(
            BoundingBox3D(
                id=996,
                cls="shelf_box2",
                pose=Pose(
                    position=Point(x=0.0, y=0.6, z=0.7),
                    orientation=Quaternion(x=0.0, y=0.0, z=0.0, w=1.0),
                ),
                scale=Vector3(x=0.8, y=0.44, z=0.04),
            )
        )

        return transformed_bbox_3d

    def grasping(self, open: bool = True):
        self._gripper_client.control_gripper(open=open)

    def control(
        self,
        header: Header,
        target_pose: Pose,
        tolerance: float,
        scale_factor: float,
        joint_states: JointState = None,
    ):
        goal_constraints = self._moveit_client.get_goal_constraint(
            pose_stamped=PoseStamped(header=header, pose=target_pose),
            orientation_constraints=None,
            tolerance=tolerance,
            end_effector=self.end_effector,
            joint_states=joint_states,
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

    def transform_bbox3d(
        self, bbox_3d: BoundingBox3DMultiArray, target_frame: str = "world"
    ) -> BoundingBox3DMultiArray:
        if self.tf_buffer.can_transform(
            target_frame,
            "camera1_link",
            time=self._node.get_clock().now(),
            timeout=Duration(seconds=1),
        ):
            for bbox in bbox_3d.data:
                bbox: BoundingBox3D

                pose = TF2PoseStamped(
                    header=Header(
                        stamp=self._node.get_clock().now().to_msg(),
                        frame_id="camera1_link",
                    ),
                    pose=bbox.pose,
                )
                pose = self.tf_buffer.transform(
                    object_stamped=pose,
                    target_frame=target_frame,
                    timeout=Duration(seconds=1),
                )
                bbox.pose = pose.pose
        else:
            self._node.get_logger().warn(
                f"Cannot lookup transform between {target_frame} and camera1_link."
            )
            return None

        return bbox_3d

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
