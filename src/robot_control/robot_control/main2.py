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
from object_tracker.object_pose_estimation_server import ObjectPoseEstimationManager
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

        self._object_pose_estimation_manager = ObjectPoseEstimationManager(
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

        # >>> Parameters >>>
        self._end_effector = "gripper_link"
        self._target_cls = kwargs["target_cls"]
        self._state = State.WAITING
        self._operations = {
            State.WAITING.value: self.waiting,
            State.MEGAPOSE_SEARCHING.value: self.megapose_searching,
            State.FCN_SEARCHING.value: self.fcn_searching,
            State.GRASPING_POSITIONING.value: self.grasping_positioning,
            State.TARGET_AIMING.value: self.target_aiming,
            State.TARGET_POSITIONING.value: self.target_positioning,
            State.GRASPING.value: self.grasping,
            State.HOME_AIMING.value: self.home_aiming,
            State.HOME_POSITIONING.value: self.home_positioning,
            State.DROP_POSITIONING.value: self.drop_positioning,
            State.UNGRASPING.value: self.ungrasping,
            State.FCN_POSITIONING.value: self.fcn_positioning,
        }
        # <<< Parameters <<<

        # >>> Data >>>
        self._target_objects: BoundingBox3DMultiArray = None
        self._target_object: BoundingBox3D = None
        # <<< Data <<<

        # >>> Unique Joint States >>>
        self._home_joints = JointState(
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
        self._dropping_joints = JointState(
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
        self._waiting_joints = JointState(
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

        self._home_pose: PoseStamped = None
        self._drop_pose: PoseStamped = None
        # <<< Unique Joint States <<<

    # >>> Main Control Method >>>

    def run(self):
        header = Header(
            stamp=self._node.get_clock().now().to_msg(),
            frame_id="world",
        )
        self._operations[self.state.value](header=header)

    # <<< Main Control Method <<<

    # >>> Operation Methods >>>
    def waiting(self, header: Header):
        """
        Initialize home pose and drop pose
        """
        try:
            self._home_pose = self._fk_service_manager.run(
                joint_states=self._home_joints,
                end_effector=self._end_effector,
            )
            self._drop_pose = self._fk_service_manager.run(
                joint_states=self._dropping_joints,
                end_effector=self._end_effector,
            )

            self._node.get_logger().info("Waiting Success")
            self._state = State(self.state.value + 1)

            return True

        except ValueError as ve:
            self._node.get_logger().warn(f"Value Error: {ve}")

        except Exception as e:
            self._node.get_logger().error(f"Unexpected Error: {e}")
            self._node.get_logger().error("Waiting Failed")

        return False

    def megapose_searching(self, header: Header):
        """
        Run megapose client to get all the objects' pose
        Return the object pose estimation in camera frame
        """
        try:
            # >>> STEP 1. Get the current planning scene and reset it >>>
            current_scene = self._get_planning_scene_service_manager.run()
            is_reset_success = (
                self._apply_planning_scene_service_manager.reset_planning_scene(
                    scene=current_scene
                )
            )

            # >>> STEP 2. Get the object pose estimation >>>
            if is_reset_success:
                bbox_3d = self._object_pose_estimation_manager.send_request()

                # >>> STEP 3. Transform the bounding box to the world frame >>>
                transformed_bbox_3d = self._transform_manager.transform_bbox_3d(
                    bbox_3d=bbox_3d,
                    target_frame="world",
                    source_frame="camera1_link",
                )

                # >>> STEP 4. Apply the transformed bounding box to the planning scene >>>
                if transformed_bbox_3d is not None:
                    collision_objects = self._apply_planning_scene_service_manager.collision_object_from_bbox_3d(
                        header=header,
                        bbox_3d=transformed_bbox_3d,
                    )
                    is_applying_success = self._apply_planning_scene_service_manager.add_collistion_objects(
                        collision_objects=collision_objects
                    )
                    if is_applying_success:
                        self._target_objects = bbox_3d
                        self._node.get_logger().info("Apply Planning Scene Success")
                        self.state = State(self.state.value + 1)
                        return True

        except ValueError as ve:
            self._node.get_logger().warn(f"Value Error: {ve}")

        except Exception as e:
            self._node.get_logger().error(f"Unexpected Error: {e}")
            self._node.get_logger().error("Megapose Searching Failed")

        return False

    def fcn_searching(self, header: Header):
        """
        Run FCN Integration Client to get the target object
        """
        try:
            self._fcn_integration_client_manager.send_fcn_integration_request(
                target_cls=self._target_cls
            )

            row, col = self._fcn_integration_client_manager.fcn_result

            if row is not None and col is not None:
                center_coord = self._object_selection_manager.get_center_coord(
                    row=row, col=int(col)
                )
                target_object: BoundingBox3D = (
                    self._object_selection_manager.get_target_object(
                        center_coord=center_coord, target_objects=self._target_objects
                    )
                )

                if target_object is not None:
                    self._target_object = target_object
                    self._state = State(self.state.value + 1)
                    self._node.get_logger().info(f"FCN Searching Success")
                    return True

        except ValueError as ex:
            self._node.get_logger().warn(f"Value Error: {ex}")

        except Exception as e:
            self._node.get_logger().error(f"Unexpected Error: {e}")
            self._node.get_logger().error("FCN Searching Failed")

        return False

    def grasping_positioning(self, header: Header):
        """
        Run kinematic path service to get the target object pose.
        Target pose is home joints
        """
        try:
            self.control(
                header=header,
                target_pose=Pose(),  # To ignore the target pose
                joint_states=self._home_joints,
                tolerance=0.01,
                scale_factor=0.5,
                use_path_contraint=False,
            )

            self._state = State(self.state.value + 1)

            return True

        except ValueError as ve:
            self._node.get_logger().warn(f"Value Error: {ve}")

        except Exception as e:
            self._node.get_logger().error(f"Unexpected Error: {e}")
            self._node.get_logger().error("Grasping Positioning Failed")

        return False

    def target_aiming(self, header: Header):
        """
        Run kinematic path service to get the target object pose.
        Target pose is the pose which is located in front of the target object
        """
        try:
            target_pose: Pose = self._target_object.pose

            self.control(
                header=header,
                target_pose=Pose(
                    position=Point(
                        x=target_pose.position.x,
                        y=target_pose.position.y - 0.2,
                        z=target_pose.position.z,
                    ),
                    orientation=self._home_pose.pose.orientation,
                ),
                joint_states=None,
                tolerance=0.01,
                scale_factor=0.5,
                use_path_contraint=True,
            )

            self._state = State(self.state.value + 1)

            return True

        except ValueError as ve:
            self._node.get_logger().warn(f"Value Error: {ve}")

        except Exception as e:
            self._node.get_logger().error(f"Unexpected Error: {e}")
            self._node.get_logger().error("Target Aiming Failed")

        return False

    def target_positioning(self, header: Header):
        """
        Run kinematic path service to get the target object pose.
        Target pose is the pose of the target object.
        """
        try:
            target_pose: Pose = self._target_object.pose

            self.control(
                header=header,
                target_pose=Pose(
                    position=target_pose.position,
                    orientation=self._home_pose.pose.orientation,
                ),
                joint_states=None,
                tolerance=0.01,
                scale_factor=0.2,
                use_path_contraint=True,
            )

        except ValueError as ve:
            self._node.get_logger().warn(f"Value Error: {ve}")

        except Exception as e:
            self._node.get_logger().error(f"Unexpected Error: {e}")
            self._node.get_logger().error("Target Positioning Failed")

        return False

    def grasping(self, header: Header):
        """
        Run gripper action to grasp the target object
        """
        self._gripper_action_manager.control_gripper(open=False)
        if self._gripper_action_manager.is_finished is True:
            self._state = State(self.state.value + 1)
            self._node.get_logger().info("Grasping Success")
            return True

        return False

    def ungrasping(self, header: Header):
        """
        Run gripper action to ungrasp the target object
        """
        try:
            self._gripper_action_manager.control_gripper(open=True)
            if self._gripper_action_manager.is_finished is True:
                self._state = State(self.state.value + 1)
                self._node.get_logger().info("Ungrasing Success")
                return True

        except ValueError as ve:
            self._node.get_logger().warn(f"Value Error: {ve}")

        except Exception as e:
            self._node.get_logger().error(f"Unexpected Error: {e}")
            self._node.get_logger().error("Ungrasing Failed")

        return False

    def home_aiming(self, header: Header):
        """
        Run kinematic path service to get the target object pose.
        Target pose is the pose which is located the front and above the target object.
        """
        try:
            target_pose: Pose = self._target_object.pose

            self.control(
                header=header,
                target_pose=Pose(
                    position=Point(
                        x=target_pose.position.x,
                        y=target_pose.position.y - 0.2,
                        z=target_pose.position.z + 0.05,
                    ),
                    orientation=self._home_pose.pose.orientation,
                ),
                joint_states=None,
                tolerance=0.01,
                scale_factor=0.5,
                use_path_contraint=True,
            )

            self._state = State(self.state.value + 1)

            return True

        except ValueError as ve:
            self._node.get_logger().warn(f"Value Error: {ve}")

        except Exception as e:
            self._node.get_logger().error(f"Unexpected Error: {e}")
            self._node.get_logger().error("Home Aiming Failed")

        return False

    def home_positioning(self, header: Header):
        """
        Run kinematic path service to get the target object pose.
        Target pose is the pose which is located above the target object.
        """
        try:
            self.control(
                header=header,
                target_pose=Pose(
                    position=Point(
                        x=self._home_pose.pose.position.x,
                        y=self._home_pose.pose.position.y,
                        z=self._home_pose.pose.position.z + 0.05,
                    ),
                    orientation=self._home_pose.pose.orientation,
                ),
                joint_states=None,
                tolerance=0.01,
                scale_factor=0.5,
                use_path_contraint=True,
            )

            self._state = State(self.state.value + 1)

            return True

        except ValueError as ve:
            self._node.get_logger().warn(f"Value Error: {ve}")

        except Exception as e:
            self._node.get_logger().error(f"Unexpected Error: {e}")
            self._node.get_logger().error("Home Positioning Failed")

        return False

    def drop_positioning(self, header: Header):
        """
        Run kinematic path service to get the target object pose.
        Target pose is dropping joints.
        """

        try:
            self.control(
                header=header,
                target_pose=Pose(),  # To ignore the target pose
                joint_states=self._dropping_joints,
                tolerance=0.01,
                scale_factor=0.5,
                use_path_contraint=False,
            )

            self._state = State(self.state.value + 1)

            return True

        except ValueError as ve:
            self._node.get_logger().warn(f"Value Error: {ve}")

        except Exception as e:
            self._node.get_logger().error(f"Unexpected Error: {e}")
            self._node.get_logger().error("Drop Positioning Failed")

        return False

    def fcn_positioning(self, header: Header):
        """
        Run kinematic path service to get the target object pose.
        Target pose is waiting joints.
        """

        try:
            self.control(
                header=header,
                target_pose=Pose(),  # To ignore the target pose
                joint_states=self._waiting_joints,
                tolerance=0.01,
                scale_factor=0.5,
                use_path_contraint=False,
            )

            self._state = State(self.state.value + 1)

            return True

        except ValueError as ve:
            self._node.get_logger().warn(f"Value Error: {ve}")

        except Exception as e:
            self._node.get_logger().error(f"Unexpected Error: {e}")
            self._node.get_logger().error("FCN Positioning Failed")

        return False

    def control(
        self,
        header: Header,
        target_pose: Pose,
        joint_states: JointState,
        tolerance: float,
        scale_factor: float,
        use_path_contraint: bool = False,
    ):
        """
        Control the robot to the target pose.
        :param target_pose: The target pose to control the robot to. If None, use the current pose.
        :param joint_states: The joint states to control the robot to. If None, use the current joint states.
        """
        # >>> STEP 0. Exception handling >>>
        if target_pose is None and joint_states is None:
            raise ValueError("Either target_pose or joint_states must be provided.")

        if target_pose is not None and joint_states is not None:
            raise ValueError(
                "Either target_pose or joint_states must be provided, not both."
            )

        try:
            # >>>> STEP 1. Set path constraint. Limit the end-effector's orientation >>>
            path_constraint = Constraints(
                name="path_constraint",
                orientation_constraints=[
                    OrientationConstraint(
                        header=header,
                        link_name=self._end_effector,
                        orientation=self._home_pose.pose.orientation,
                        absolute_x_axis_tolerance=0.1,
                        absolute_y_axis_tolerance=0.1,
                        absolute_z_axis_tolerance=0.1,
                        weight=1.0,
                    )
                ],
            )

            # >>> STEP 2-1. Case 1. If target_pose is given >>>
            if target_pose is not None:
                ik_robot_state = self._ik_service_manager.run(
                    pose_stamped=target_pose,
                    joint_states=self._joint_states_manager.joint_states,
                    end_effector=self._end_effector,
                )
                ik_joint_states = ik_robot_state.joint_state

            # >>> STEP 2-2. Case 2. If joint_states is given >>>
            if joint_states is not None:
                ik_joint_states = joint_states

            # >>> STEP 3. Get the goal constraint >>>
            goal_constraints = self._kinematic_path_service_manager.get_goal_constraint(
                goal_joint_states=ik_joint_states,
                tolerance=tolerance,
                tolerance=tolerance,
            )

            # >>> STEP 4. Get the current planning scene >>>
            trajectory: RobotTrajectory = self._kinematic_path_service_manager.run(
                goal_constraints=goal_constraints,
                path_constraint=path_constraint if use_path_contraint else None,
                joint_states=self._joint_states_manager.joint_states,
            )

            # >>> STEP 5. Scale the trajectory >>>
            scaled_trajectory: RobotTrajectory = (
                self._execute_trajectory_service_manager.scale_trajectory(
                    trajectory=trajectory,
                    scale_factor=scale_factor,
                )
            )

            # >>> STEP 6. Get the current planning scene >>>
            self._execute_trajectory_service_manager.run(trajectory=scaled_trajectory)

            return True

        except Exception as e:
            self._node.get_logger().error(f"Unexpected Error: {e}")
            self._node.get_logger().error("Control Failed")
            return False

    # <<< Operation Methods <<<


def main():
    rclpy.init(args=None)

    node = Node("main_control_node")
    main_node = MainControlNode(node=node)

    # Spin in a separate thread
    thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    thread.start()

    hz = 1.0
    rate = node.create_rate(hz)

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
