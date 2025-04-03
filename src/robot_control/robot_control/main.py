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
import argparse

# custom
from base_package.manager import ObjectManager, TransformManager
from fcn_network.fcn_manager import FCN_Integration_Manager
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
    ControlAction,
)


class State(Enum):
    ACTION_SELECTING = -2
    WAITING = -1  # Reset the planning scene, add home and drop pose

    FCN_POSITIONING = 0  # Move to the waiting pose
    MEGAPOSE_SEARCHING = 1  #  Get the object pose estimation
    FCN_SEARCHING = 2  # Send FCN request to get the target grid

    GRASPING_HOMING1 = 10
    GARSPING_TARGET_AIMING = 11  # Move to the front of the target object
    GARSPING_TARGET_POSITIONING = 12  # Move to the target object pose
    GARSPING_GRASPING = 13
    GARSPING_HOME_AIMING = 14  # Move to the front of the target object
    GARSPING_HOMING2 = 15
    GARSPING_DROP_POSITIONING = 16  # Move to the drop pose
    GARSPING_UNGRASPING = 17

    SWEEPING_HOMING1 = 20  # Move to the home pose
    SWEEPING_TARGET_AIMING = 21  # Move to the front of the target object
    SWEEPING_TARGET_POSITIONING = 22  # Move to the side of the target object
    SWEEPING_SWEEPING = 23  # Sweep the target object
    SWEEPING_HOME_AIMING = 24  # Move to the front of the target object
    SWEEPING_HOMING2 = 25  # Move to the home pose


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

        self._fcn_integration_manager = FCN_Integration_Manager(
            node=self._node, *args, **kwargs
        )

        self._object_manager = ObjectManager(node=self._node, *args, **kwargs)
        self._object_selection_manager = ObjectSelectionManager(
            node=self._node, *args, **kwargs
        )

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

        # <<< Managers <<<

        # >>> Parameters >>>
        self._end_effector = "gripper_link"
        self._target_cls = kwargs["target_cls"]
        self._state = State.WAITING
        self._operations = {
            # LEVEL 0
            State.ACTION_SELECTING.value: self.action_selecting,
            State.WAITING.value: self.waiting,
            # LEVEL 1
            State.FCN_POSITIONING.value: self.fcn_positioning,
            State.MEGAPOSE_SEARCHING.value: self.megapose_searching,
            State.FCN_SEARCHING.value: self.fcn_searching,
            # LEVEL 3
            State.GRASPING_HOMING1.value: self.home_positioning,
            State.GARSPING_TARGET_AIMING.value: self.target_aiming,
            State.GARSPING_TARGET_POSITIONING.value: self.target_positioning,
            State.GARSPING_GRASPING.value: self.grasping,
            State.GARSPING_HOME_AIMING.value: self.home_aiming,
            State.GARSPING_HOMING2.value: self.home_positioning,
            State.GARSPING_DROP_POSITIONING.value: self.drop_positioning,
            State.GARSPING_UNGRASPING.value: self.ungrasping,
            # LEVEL 4
            State.SWEEPING_HOMING1.value: self.home_positioning,
            State.SWEEPING_TARGET_AIMING.value: self.sweep_target_aiming,
            State.SWEEPING_TARGET_POSITIONING.value: self.sweep_target_positioning,
            State.SWEEPING_SWEEPING.value: self.sweep,
            State.SWEEPING_HOME_AIMING.value: self.home_aiming,
            State.SWEEPING_HOMING2.value: self.home_positioning,
        }
        # <<< Parameters <<<

        # >>> Data >>>
        self._target_objects: BoundingBox3DMultiArray = None
        self._control_action: ControlAction = None
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
        self._node.get_logger().info(f"State: {self._state.name}")

        is_success = self._operations[self._state.value](
            header=Header(
                stamp=self._node.get_clock().now().to_msg(),
                frame_id="world",
            )
        )
        if is_success:
            self._state = State.ACTION_SELECTING

    # <<< Main Control Method <<<

    # >>> Operation Methods >>>

    # >>> LEVEL 0 >>>
    def action_selecting(self, header: Header):
        # CASE 0. Before action selecting
        if self._control_action is None:
            if self._state == State.WAITING:
                self._state = State.MEGAPOSE_SEARCHING
            elif self._state == State.MEGAPOSE_SEARCHING:
                self._state = (
                    State.FCN_SEARCHING
                )  # After FCN_SEARCHING, control action will be defined

        # CASE 1. Grasping
        elif self._control_action.action:
            if self._state == State.FCN_SEARCHING:
                self._state = State.GRASPING_HOMING1
            elif self._state == State.GARSPING_UNGRASPING:
                self._state = State.FCN_POSITIONING
            else:
                self._state = State(self._state.value + 1)

        # CASE 2. Sweeping
        else:
            if self._state == State.FCN_SEARCHING:
                self._state = State.SWEEPING_HOMING1
            elif self._state == State.SWEEPING_HOMING2:
                self._state = State.FCN_POSITIONING
            else:
                self._state = State(self._state.value + 1)

    def waiting(self, header: Header):
        """
        Initialize home pose and drop pose
        """
        try:
            if self._home_pose is None:
                self._home_pose = self._fk_service_manager.run(
                    joint_states=self._home_joints,
                    end_effector=self._end_effector,
                )
            if self._drop_pose is None:
                self._drop_pose = self._fk_service_manager.run(
                    joint_states=self._dropping_joints,
                    end_effector=self._end_effector,
                )
            return True

        except ValueError as ve:
            self._node.get_logger().warn(f"Value Error: {ve}")

        except Exception as e:
            self._node.get_logger().error(f"Unexpected Error: {e}")
            self._node.get_logger().error("Waiting Failed")

        return False

    # >>> LEVEL 1 >>>
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
                        collision_objects=collision_objects, scene=current_scene
                    )
                    is_applying_default_success = self._apply_planning_scene_service_manager.append_default_collision_objects(
                        header=header,
                        scene=current_scene,
                    )

                    if is_applying_success and is_applying_default_success:
                        self._target_objects = bbox_3d
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
            fcn_response, fcn_occupied_response = self._fcn_integration_manager.run(
                target_cls=self._target_cls,
            )

            if fcn_response is None or fcn_occupied_response is None:
                raise ValueError("FCN response or FCN occupied response is None")

            target_id = f"{fcn_occupied_response.moving_row}{fcn_response.target_col}"  # e.g. 'A1'
            goal_ids = [
                f"{fcn_occupied_response.moving_row}{col}"
                for col in fcn_occupied_response.moving_cols
            ]  # e.g. ['A0', 'A2']
            action = fcn_occupied_response.action  # True for sweep, False for grasp

            # center_coord = self._object_selection_manager.get_center_coord(
            #     row=row, col=int(col)
            # )
            # target_object: BoundingBox3D = (
            #     self._object_selection_manager.get_target_object(
            #         center_coord=center_coord, target_objects=self._target_objects
            #     )
            # )

            target_object: BoundingBox3D = (
                self._object_selection_manager.get_target_object_with_grid_id(
                    target_objects=self._target_objects, grid_id=target_id
                )
            )

            if target_object is not None:
                self._control_action = ControlAction(
                    target_id=target_id,
                    goal_ids=goal_ids,
                    action=action,
                    target_object=target_object,
                )

                self._node.get_logger().info(
                    f"FCN Searching Success: {target_id} / {self._control_action.target_object.cls}"
                )
                return True

        except ValueError as ex:
            self._node.get_logger().warn(f"Value Error: {ex}")

        except Exception as e:
            self._node.get_logger().error(f"Unexpected Error: {e}")
            self._node.get_logger().error("FCN Searching Failed")

        return False

    def fcn_positioning(self, header: Header):
        """
        Run kinematic path service to get the target object pose.
        Target pose is waiting joints.
        """

        try:
            control_success = self.control(
                header=header,
                target_pose=None,  # To ignore the target pose
                joint_states=self._waiting_joints,
                tolerance=0.01,
                scale_factor=0.5,
                use_path_contraint=False,
            )

            return control_success

        except ValueError as ve:
            self._node.get_logger().warn(f"Value Error: {ve}")

        except Exception as e:
            self._node.get_logger().error(f"Unexpected Error: {e}")
            self._node.get_logger().error("FCN Positioning Failed")

        return False

    def home_positioning(self, header: Header):
        """
        Run kinematic path service to get the target object pose.
        Target pose is the pose which is located above the target object.
        """
        try:
            control_success = self.control(
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
                use_path_contraint=False,
            )

            return control_success

        except ValueError as ve:
            self._node.get_logger().warn(f"Value Error: {ve}")

        except Exception as e:
            self._node.get_logger().error(f"Unexpected Error: {e}")
            self._node.get_logger().error("Home Positioning Failed")

        return False

    # >>> LEVEL 3 >>>
    def target_aiming(self, header: Header):
        """
        Run kinematic path service to get the target object pose.
        Target pose is the pose which is located in front of the target object
        """
        try:
            target_pose: Pose = self._control_action.target_object.pose

            control_success = self.control(
                header=header,
                target_pose=Pose(
                    position=Point(
                        x=target_pose.position.x,
                        y=target_pose.position.y - 0.1,
                        z=target_pose.position.z,
                    ),
                    orientation=self._home_pose.pose.orientation,
                ),
                joint_states=None,
                tolerance=0.05,
                scale_factor=0.5,
                use_path_contraint=False,
            )

            return control_success

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
            target_pose: Pose = self._control_action.target_object.pose

            control_success = self.control(
                header=header,
                target_pose=Pose(
                    position=Point(
                        x=target_pose.position.x,
                        y=target_pose.position.y,
                        z=target_pose.position.z - 0.02,
                    ),
                    orientation=self._home_pose.pose.orientation,
                ),
                joint_states=None,
                tolerance=0.02,
                scale_factor=0.2,
                use_path_contraint=False,
            )

            return control_success

        except ValueError as ve:
            self._node.get_logger().warn(f"Value Error: {ve}")

        except Exception as e:
            self._node.get_logger().error(f"Unexpected Error: {e}")
            self._node.get_logger().error("Target Positioning Failed")

        return False

    def home_aiming(self, header: Header):
        """
        Run kinematic path service to get the target object pose.
        Target pose is the pose which is located the front and above the target object.
        """
        try:
            target_pose: Pose = self._control_action.target_object.pose
            home_pose: Pose = self._home_pose.pose

            control_success = self.control(
                header=header,
                target_pose=Pose(
                    position=Point(
                        x=target_pose.position.x,
                        y=home_pose.position.y + 0.05,
                        z=target_pose.position.z + 0.05,
                    ),
                    orientation=self._home_pose.pose.orientation,
                ),
                joint_states=None,
                tolerance=0.01,
                scale_factor=0.5,
                use_path_contraint=False,
            )

            return control_success

        except ValueError as ve:
            self._node.get_logger().warn(f"Value Error: {ve}")

        except Exception as e:
            self._node.get_logger().error(f"Unexpected Error: {e}")
            self._node.get_logger().error("Home Aiming Failed")

        return False

    def drop_positioning(self, header: Header):
        """
        Run kinematic path service to get the target object pose.
        Target pose is dropping joints.
        """

        try:
            control_success = self.control(
                header=header,
                target_pose=None,  # To ignore the target pose
                joint_states=self._dropping_joints,
                tolerance=0.01,
                scale_factor=0.5,
                use_path_contraint=False,
            )

            return control_success

        except ValueError as ve:
            self._node.get_logger().warn(f"Value Error: {ve}")

        except Exception as e:
            self._node.get_logger().error(f"Unexpected Error: {e}")
            self._node.get_logger().error("Drop Positioning Failed")

        return False

    # >>> LEVEL 4 >>>
    def sweep_target_aiming(self, header: Header):
        """
        Run kinematic path service to get the target object pose.
        Target pose is the pose which is the front/side of the target object
        """
        try:
            target_pose: Pose = self._control_action.target_object.pose

            target_row = int(self._control_action.target_id[1])  # e.g. 'A1' -> 1
            moving_rows = max(
                [int(id[1]) for id in self._control_action.goal_ids]
            )  # e.g. ['A0', 'A2'] -> [0, 2]
            direction = target_row < moving_rows  # True for right, False for left
            offset = -0.1 if direction else 0.1

            control_success = self.control(
                header=header,
                target_pose=Pose(
                    position=Point(
                        x=target_pose.position.x + offset,
                        y=target_pose.position.y - 0.1,
                        z=target_pose.position.z,
                    ),
                    orientation=self._home_pose.pose.orientation,
                ),
                joint_states=None,
                tolerance=0.05,
                scale_factor=0.5,
                use_path_contraint=False,
            )

            return control_success

        except ValueError as ve:
            self._node.get_logger().warn(f"Value Error: {ve}")

        except Exception as e:
            self._node.get_logger().error(f"Unexpected Error: {e}")
            self._node.get_logger().error("Target Aiming Failed")

        return False

    def sweep_target_positioning(self, header: Header):
        """
        Run kinematic path service to get the target object pose.
        Target pose is the side pose of the target object.
        """
        try:
            target_pose: Pose = self._control_action.target_object.pose

            target_row = int(self._control_action.target_id[1])  # e.g. 'A1' -> 1
            moving_rows = max(
                [int(id[1]) for id in self._control_action.goal_ids]
            )  # e.g. ['A0', 'A2'] -> [0, 2]
            direction = target_row < moving_rows  # True for right, False for left
            offset = -0.1 if direction else 0.1

            control_success = self.control(
                header=header,
                target_pose=Pose(
                    position=Point(
                        x=target_pose.position.x + offset,
                        y=target_pose.position.y,
                        z=target_pose.position.z - 0.02,
                    ),
                    orientation=self._home_pose.pose.orientation,
                ),
                joint_states=None,
                tolerance=0.02,
                scale_factor=0.2,
                use_path_contraint=False,
            )

            return control_success

        except ValueError as ve:
            self._node.get_logger().warn(f"Value Error: {ve}")

        except Exception as e:
            self._node.get_logger().error(f"Unexpected Error: {e}")
            self._node.get_logger().error("Target Positioning Failed")

        return False

    def sweep(self, header: Header):
        """
        Run kinematic path service to get the target object pose.
        Target pose is the side pose of the target object.
        """
        try:
            if self._apply_planning_scene_service_manager.reset_planning_scene():
                if (
                    self._apply_planning_scene_service_manager.append_default_collision_objects()
                ):

                    target_pose: Pose = self._control_action.target_object.pose

                    target_row = int(
                        self._control_action.target_id[1]
                    )  # e.g. 'A1' -> 1
                    moving_rows = max(
                        [int(id[1]) for id in self._control_action.goal_ids]
                    )  # e.g. ['A0', 'A2'] -> [0, 2]
                    direction = (
                        target_row < moving_rows
                    )  # True for right, False for left
                    offset = 0.15 if direction else -0.15

                    control_success = self.control(
                        header=header,
                        target_pose=Pose(
                            position=Point(
                                x=target_pose.position.x + offset,
                                y=target_pose.position.y,
                                z=target_pose.position.z - 0.02,
                            ),
                            orientation=self._home_pose.pose.orientation,
                        ),
                        joint_states=None,
                        tolerance=0.02,
                        scale_factor=0.2,
                        use_path_contraint=False,
                    )

                    return control_success

        except ValueError as ve:
            self._node.get_logger().warn(f"Value Error: {ve}")

        except Exception as e:
            self._node.get_logger().error(f"Unexpected Error: {e}")
            self._node.get_logger().error("Target Positioning Failed")

        return False

    # >>> LEVEL 5 >>>
    def grasping(self, header: Header):
        """
        Run gripper action to grasp the target object
        """
        self._gripper_action_manager.control_gripper(open=False)
        if self._gripper_action_manager.is_finished is True:
            return True

        return False

    def ungrasping(self, header: Header):
        """
        Run gripper action to ungrasp the target object
        """
        try:
            self._gripper_action_manager.control_gripper(open=True)
            if self._gripper_action_manager.is_finished is True:
                return True

        except ValueError as ve:
            self._node.get_logger().warn(f"Value Error: {ve}")

        except Exception as e:
            self._node.get_logger().error(f"Unexpected Error: {e}")
            self._node.get_logger().error("Ungrasing Failed")

        return False

    # >>> ETC >>>
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
                        absolute_x_axis_tolerance=0.2,
                        absolute_y_axis_tolerance=0.2,
                        absolute_z_axis_tolerance=0.2,
                        weight=1.0,
                    )
                ],
            )

            # >>> STEP 2-1. Case 1. If target_pose is given >>>
            if target_pose is not None:
                ik_robot_state = self._ik_service_manager.run(
                    pose_stamped=PoseStamped(header=header, pose=target_pose),
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
            )

            # >>> STEP 4. Get the current planning scene >>>
            trajectory: RobotTrajectory = self._kinematic_path_service_manager.run(
                goal_constraints=[goal_constraints],
                path_constraints=path_constraint if use_path_contraint else None,
                joint_states=self._joint_states_manager.joint_states,
            )

            if trajectory is None:
                raise ValueError("Trajectory is None")

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

    parser = argparse.ArgumentParser(description="FCN Server Node")

    parser.add_argument(
        "--grid_data_file",
        type=str,
        required=False,
        default="grid_data.json",
        help="Path or file name of object bounds. If input is a file name, the file should be located in the 'resource' directory. Required",
    )
    parser.add_argument(
        "--target_cls",
        type=str,
        required=True,
        help="Target class to search. Required",
    )

    args = parser.parse_args()
    kagrs = vars(args)

    node = Node("main_control_node")
    main_node = MainControlNode(node=node, **kagrs)

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
