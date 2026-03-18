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
import sys
import os
import numpy as np
from enum import Enum
import json
import argparse
import time

# custom
from base_package.manager import ObjectManager, TransformManager
from fcn_network.fcn_manager import (
    FCN_Integration_Manager,
    GridManager,
)
from fcn_network.direct_fcn_server import DirectFCNServer
from object_tracker.closest_object_classifier import ClosestObjectClassifierNode
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
    DropGridManager,
    RandomSearchManager,
)
from robot_control.log_server import LogManager


class State(Enum):
    ACTION_SELECTING = -2
    WAITING = -1  # Reset the planning scene, add home and drop pose

    FCN_POSITIONING = 0  # Move to the waiting pose
    MEGAPOSE_SEARCHING = 1  #  Get the object pose estimation
    FCN_SEARCHING = 2  # Send FCN request to get the target grid

    GRASPING_HOMING1 = 10
    GARSPING_TARGET_AIMING = 11  # Move to the front of the target object
    GARSPING_TARGET_AIMING2 = 12
    GARSPING_TARGET_POSITIONING = 13  # Move to the target object pose
    GARSPING_GRASPING = 14
    GARSPING_HOME_AIMING = 15  # Move to the front of the target object
    GARSPING_HOMING2 = 16
    GARSPING_DROP_POSITIONING = 17  # Move to the drop pose
    GRASPING_DROP_TABLE_POSITIONING = 18  # Move to the drop pose
    GARSPING_UNGRASPING = 19
    GARSPING_DROP_POSITIONING2 = 20

    SWEEPING_HOMING1 = 30  # Move to the home pose
    SWEEPING_TARGET_AIMING = 31  # Move to the front of the target object
    SWEEPING_TARGET_AIMING2 = 32
    SWEEPING_TARGET_POSITIONING = 33  # Move to the side of the target object
    SWEEPING_SWEEPING = 34  # Sweep the target object
    SWEEPING_HOME_AIMING = 35  # Move to the front of the target object
    SWEEPING_HOMING2 = 36  # Move to the home pose

    FINISHED = 99


class MainControlNode(object):
    def __init__(self, node: Node, *args, **kwargs):
        self._node = node

        self._debug = kwargs["debug"]
        self._mode = kwargs["mode"]
        self._node.get_logger().info(f"MODE: {self._mode}")
        self._node.get_logger().info(f"DEBUG MODE: {self._debug}")

        # >>> Managers >>>
        self._transform_manager = TransformManager(node=self._node, *args, **kwargs)
        self._joint_states_manager = JointStatesManager(
            node=self._node, *args, **kwargs
        )

        self._log_manager = LogManager(
            node=self._node,
            *args,
            **kwargs,
        )

        drop_grid_manager_kwargs = dict(kwargs)
        drop_grid_manager_kwargs["grid_data_file"] = kwargs["drop_grid_data_file"]
        self._drop_grid_manager = DropGridManager(
            node=self._node, *args, **drop_grid_manager_kwargs
        )
        self._object_pose_estimation_manager = ObjectPoseEstimationManager(
            node=self._node, *args, **kwargs
        )

        self._object_manager = ObjectManager(node=self._node, *args, **kwargs)
        self._object_selection_manager = ObjectSelectionManager(
            node=self._node, *args, **kwargs
        )

        if self._mode == 1 or self._mode == 0:
            self._fcn_integration_manager = FCN_Integration_Manager(
                node=self._node, *args, **kwargs
            )
        elif self._mode == 2:
            self._fcn_direct_mananger = DirectFCNServer(
                node=self._node, *args, **kwargs
            )
        elif self._mode == 3:
            self._random_search_manager = RandomSearchManager(
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
            node=self._node, fraction_threshold=0.8, *args, **kwargs
        )
        self._kinematic_path_service_manager = KinematicPath_ServiceManager(
            node=self._node, *args, **kwargs
        )
        self._execute_trajectory_service_manager = ExecuteTrajectory_ServiceManager(
            node=self._node, *args, **kwargs
        )

        self._closest_object_classifier = ClosestObjectClassifierNode(
            node=self._node,
            *args,
            **kwargs,
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
            State.FCN_SEARCHING.value: (
                self.direct_fcn_searching
                if self._mode == 2
                else self.random_searching if self._mode == 3 else self.fcn_searching
            ),
            # LEVEL 3
            State.GRASPING_HOMING1.value: self.home_positioning,  # self.home_positioning,
            State.GARSPING_TARGET_AIMING.value: self.target_aiming,
            State.GARSPING_TARGET_AIMING2.value: self.target_aiming2,
            State.GARSPING_TARGET_POSITIONING.value: self.target_positioning,
            State.GARSPING_GRASPING.value: self.grasping,
            State.GARSPING_HOME_AIMING.value: self.home_aiming,
            State.GARSPING_HOMING2.value: self.home_up_positioning,
            State.GARSPING_DROP_POSITIONING.value: self.drop_positioning,
            State.GRASPING_DROP_TABLE_POSITIONING.value: self.drop_table_positioning,
            State.GARSPING_UNGRASPING.value: self.ungrasping,
            State.GARSPING_DROP_POSITIONING2.value: self.drop_rollback,
            # LEVEL 4
            State.SWEEPING_HOMING1.value: self.home_positioning,
            State.SWEEPING_TARGET_AIMING.value: self.sweep_target_aiming,
            State.SWEEPING_TARGET_AIMING2.value: self.sweep_target_aiming2,
            State.SWEEPING_TARGET_POSITIONING.value: self.sweep_target_positioning,
            State.SWEEPING_SWEEPING.value: self.sweep,
            State.SWEEPING_HOME_AIMING.value: self.home_aiming,
            State.SWEEPING_HOMING2.value: self.home_positioning,
            # FINISHED
            State.FINISHED.value: self.finished,
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
                -np.pi / 2.0,
                np.pi,
                -1.616389576588766,
            ],
        )
        self._home_up_joints = JointState(
            name=[
                "shoulder_lift_joint",
                "elbow_joint",
                "wrist_1_joint",
                "wrist_2_joint",
                "wrist_3_joint",
                "shoulder_pan_joint",
            ],
            position=[
                -0.8533289450337911,
                -2.2019173476285356,
                -3.243817707337641,
                -np.pi / 2.0,
                np.pi,
                -1.6112989998426164,
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
                -np.pi / 2.0,
                np.pi,
                0.0,
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
                -np.pi / 18.0,
                -np.pi * (7.0 / 18.0),
                -np.pi,
                -np.pi / 2.0,
                np.pi,
                0.0,
            ],
        )
        self._sweeping_to_right_joints = JointState(
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
                np.pi * 1.5,
                -1.616389576588766,
            ],
        )

        self._sweeping_to_left_joints = JointState(
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
                np.pi * 0.5,
                -1.616389576588766,
            ],
        )

        self._home_pose: PoseStamped = None
        self._home_up_pose: PoseStamped = None
        self._drop_pose: PoseStamped = None
        self._drop_rollback_pose: Pose = None
        self._sweeping_to_right_pose: PoseStamped = None
        self._sweeping_to_left_pose: PoseStamped = None
        # <<< Unique Joint States <<<

        # >>> TEST >>>
        self._step = 0
        self._target_z = 0.30
        self._is_finished = False
        self._planning_attempt = 0
        self._moving_col: int = -1
        self._target_pose_pub = self._node.create_publisher(
            PoseStamped,
            self._node.get_name() + "/target_pose",
            qos_profile_system_default,
        )

    # >>> Main Control Method >>>

    def run(self):
        self._drop_grid_manager.publish_grid_marker()

        self._node.get_logger().info(f"State: {self._state.name}")

        header = Header(
            stamp=self._node.get_clock().now().to_msg(),
            frame_id="world",
        )

        # self._node.get_logger().error("")
        # if len(self._drop_grid_manager.collision_objects) == 0:
        #     self._node.get_logger().error("No Drop Grid")

        # for g in self._drop_grid_manager.collision_objects:
        #     self._node.get_logger().error(
        #         f"{g.id} {g.primitive_poses[0].position.x:.3f}"
        #     )
        # self._node.get_logger().error("")

        success = self._operations[self._state.value](header=header)

        if success:
            self.action_selecting(header)

    # <<< Main Control Method <<<

    # >>> Operation Methods >>>

    def finished(self, header: Header):
        return True

    def logging(self, header: Header):
        fcn_data = (
            self._fcn_direct_mananger.fcn_result_data.tolist()
            if self._mode == 2
            else None
        )
        action = -1

        if not self._control_action.action:
            action = 0

        else:
            target_col = int(self._control_action.target_id[1])
            moving_col = int(self._control_action.goal_ids[-1][1])

            if target_col < moving_col:
                action = 1
            elif target_col > moving_col:
                action = 2
        # 0: Grasp, 1: Sweep Right 2: Sweep Left

        is_success = self._log_manager.log(
            fcn_data=fcn_data,
            action=action,
            column=int(self._control_action.target_id[1]),
            step=self._step,
        )
        return is_success

    # >>> LEVEL 0 >>>
    def action_selecting(self, header: Header):
        # CASE 0. Before action selecting
        if self._state == State.FINISHED:
            return True

        elif self._control_action is None:
            if self._state == State.WAITING:
                self._state = State.MEGAPOSE_SEARCHING

            elif self._state == State.MEGAPOSE_SEARCHING:
                self._state = (
                    State.FCN_SEARCHING
                )  # After FCN_SEARCHING, control action will be defined

            elif self._state == State.FCN_POSITIONING:
                if self._is_finished:
                    self._state = State.FINISHED

                else:
                    self._state = State.WAITING

        # CASE 1. Grasping
        elif self._control_action.action:

            if self._state == State.FCN_SEARCHING:
                self._state = State.SWEEPING_HOMING1

            elif self._state == State.SWEEPING_HOMING2:
                self._state = State.FCN_POSITIONING
                self.logging(header=header)
                self._control_action = None
                self._step += 1

            else:
                self._state = State(self._state.value + 1)

        # CASE 2. Sweeping
        else:
            if self._state == State.FCN_SEARCHING:
                self._state = State.GRASPING_HOMING1

            elif self._state == State.GARSPING_DROP_POSITIONING2:
                self._state = State.FCN_POSITIONING
                self.logging(header=header)
                self._control_action = None
                self._step += 1

            else:
                self._state = State(self._state.value + 1)

        return True

    def waiting(self, header: Header):
        """
        Initialize home pose and drop pose
        """
        try:
            self.ungrasping(header=header)

            if self._home_pose is None:
                self._home_pose = self._fk_service_manager.run(
                    joint_states=self._home_joints,
                    end_effector=self._end_effector,
                )
            if self._home_up_pose is None:
                self._home_up_pose = self._fk_service_manager.run(
                    joint_states=self._home_up_joints,
                    end_effector=self._end_effector,
                )
            if self._drop_pose is None:
                self._drop_pose = self._fk_service_manager.run(
                    joint_states=self._dropping_joints,
                    end_effector=self._end_effector,
                )
            if self._sweeping_to_left_pose is None:
                self._sweeping_to_left_pose = self._fk_service_manager.run(
                    joint_states=self._sweeping_to_left_joints,
                    end_effector=self._end_effector,
                )
            if self._sweeping_to_right_pose is None:
                self._sweeping_to_right_pose = self._fk_service_manager.run(
                    joint_states=self._sweeping_to_right_joints,
                    end_effector=self._end_effector,
                )
            # self.action_selecting(header=header)
            return True

        except ValueError as ve:
            self._node.get_logger().warn(f"Value Error: {ve}")

        except Exception as e:
            self._node.get_logger().error(f"Unexpected Error: {e}")
            self._node.get_logger().error("Waiting Failed")

        return False

    def check_mission_finished(self, header: Header):
        # >>> STEP 0. Check Target is exist >>>
        try:
            front_objects: dict = self._closest_object_classifier.get_closest_object()

            for key, values in front_objects.items():
                if values["class_id"] == -1:
                    continue

                cls = self._object_manager.reverse_indexs[
                    values["class_id"]
                ]  # e.g. bottle_1

                if cls == self._target_cls:
                    self._node.get_logger().info(
                        f"*** END FLAG ***\nTarget Object Detected: {cls}"
                    )

                    for obj in self._target_objects.data:
                        obj: BoundingBox3D

                    target_object: BoundingBox3D = None
                    for obj in self._target_objects.data:
                        obj: BoundingBox3D

                        if int(obj.cls[1]) == key:
                            target_object = obj
                            break

                    if target_object is not None:
                        target_id = target_object.cls

                        target_object: BoundingBox3D = (
                            self._object_selection_manager.get_target_object_with_grid_id(
                                target_objects=self._target_objects, grid_id=target_id
                            )
                        )

                        self._control_action = ControlAction(
                            target_id=target_id,
                            goal_ids=[],
                            action=False,
                            target_object=target_object,
                        )

                        self._is_finished = True

                        # self.action_selecting(header=header)

                        return True

        except ValueError as ve:
            self._node.get_logger().warn(f"Value Error: {ve}")
        except Exception as e:
            self._node.get_logger().error(f"Unexpected Error: {e}")
            self._node.get_logger().error("Check Mission Finished Failed")

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

                    default_collision_objects_bbox = (
                        self._apply_planning_scene_service_manager.get_default_collision_objects()
                    )
                    default_collision_objects = self._apply_planning_scene_service_manager.collision_object_from_bbox_3d(
                        header=header,
                        bbox_3d=default_collision_objects_bbox,
                    )

                    is_applying_success = self._apply_planning_scene_service_manager.add_collistion_objects(
                        collision_objects=(
                            # collision_objects
                            self._drop_grid_manager.collision_objects
                            + default_collision_objects
                        ),
                        scene=current_scene,
                    )

                    if is_applying_success:
                        self._target_objects = bbox_3d
                        # self.action_selecting(header=header)

                        return True

        except ValueError as ve:
            self._node.get_logger().warn(f"Value Error: {ve}")

        except Exception as e:
            self._node.get_logger().error(f"Unexpected Error: {e}")
            self._node.get_logger().error("Megapose Searching Failed")

        return False

    def random_searching(self, header: Header):
        try:
            self._node.get_logger().info("Random Searching...")

            if self.check_mission_finished(header=header):
                return True

            target_object = self._random_search_manager.get_random_target(
                target_objects=self._target_objects
            )

            if target_object is not None:

                action = ControlAction(
                    target_id=target_object.cls,
                    goal_ids=[],
                    action=False,
                    target_object=target_object,
                )

                self._control_action = action

                action_str = "Sweep" if self._control_action.action else "Grasp"

                self._node.get_logger().info(
                    f"FCN Searching Success: {action_str} {self._control_action.target_id} -> {self._control_action.goal_ids}"
                )

                return True

        except ValueError as ve:
            self._node.get_logger().warn(f"Value Error: {ve}")

        except Exception as ex:
            self._node.get_logger().error(f"Unexpected Error: {ex}")
            self._node.get_logger().error("Random Searching Failed")

    def direct_fcn_searching(self, header: Header):
        """
        Run FCN Integration Client to get the target object
        """
        try:
            self._node.get_logger().info("FCN Direct Searching...")
            if self.check_mission_finished(header=header):
                return True

            target_id = self._object_manager.indexs[self._target_cls]
            control_action: ControlAction = self._fcn_direct_mananger.run(
                target_id=target_id
            )

            target_object: BoundingBox3D = (
                self._object_selection_manager.get_target_object_with_grid_id(
                    target_objects=self._target_objects,
                    grid_id=control_action.target_id,
                )
            )

            new_control_action = ControlAction(
                target_id=control_action.target_id,
                goal_ids=control_action.goal_ids,
                action=control_action.action,
                target_object=target_object,
            )

            if control_action is not None:
                self._control_action = new_control_action

                action_str = "Sweep" if self._control_action.action else "Grasp"

                self._node.get_logger().info(
                    f"FCN Searching Success: {action_str} {self._control_action.target_id} -> {self._control_action.goal_ids}"
                )

                # self.action_selecting(header=header)
                return True

        except ValueError as ex:
            self._node.get_logger().warn(f"Value Error: {ex}")

        except Exception as e:
            self._node.get_logger().error(f"Unexpected Error: {e}")
            self._node.get_logger().error("FCN Searching Failed")

        return False

    def fcn_searching(self, header: Header):
        """
        Run FCN Integration Client to get the target object
        """
        try:
            self._node.get_logger().info("FCN Searching...")
            if self.check_mission_finished(header=header):
                return True

            fcn_response, fcn_occupied_response = self._fcn_integration_manager.run(
                target_cls=self._target_cls, last_target_col=self._moving_col
            )

            if fcn_response is None or fcn_occupied_response is None:
                raise ValueError("FCN response or FCN occupied response is None")

            target_id = f"{fcn_occupied_response.moving_row}{fcn_response.target_col}"  # e.g. 'A1'
            goal_ids = [
                f"{fcn_occupied_response.moving_row}{col}"
                for col in fcn_occupied_response.moving_cols
            ]  # e.g. ['A0', 'A2']
            action = fcn_occupied_response.action  # True for sweep, False for grasp

            if self._mode == 0:
                action = False

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

                action_str = "Sweep" if self._control_action.action else "Grasp"

                self._node.get_logger().info(
                    f"FCN Searching Success: {action_str} {self._control_action.target_id} -> {self._control_action.goal_ids}"
                )

                # self.action_selecting(header=header)
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
                scale_factor=1.0,
                use_path_contraint=False,
            )

            if control_success:
                # self.action_selecting(header=header)
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
            if self._state == State.GRASPING_HOMING1:
                if (
                    int(self._control_action.target_id[1]) == 0
                    or self._control_action.target_id[1] == 1
                ):
                    return self.home_up_positioning(header=header)

            if self._state == State.GARSPING_HOMING2:
                target_pose = self._home_up_pose.pose

            else:
                target_pose = self._home_pose.pose

            control_success = self.control(
                header=header,
                target_pose=None,  # target_pose,
                joint_states=self._home_joints,
                tolerance=0.01,
                scale_factor=1.0,
                use_path_contraint=False,
            )

            self._target_pose_pub.publish(PoseStamped(header=header, pose=target_pose))

            if control_success:
                # self.action_selecting(header=header)
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
            # current_scene = self._get_planning_scene_service_manager.run()
            # self._apply_planning_scene_service_manager.reset_planning_scene(
            #     current_scene
            # )

            target_pose: Pose = self._control_action.target_object.pose
            target_pose = Pose(
                position=Point(
                    x=target_pose.position.x,
                    y=self._home_pose.pose.position.y + 0.03,
                    z=self._target_z,
                ),
                orientation=self._home_pose.pose.orientation,
            )

            # control_success = self.control_caterian_path(
            #     header=header,
            #     target_pose=target_pose,
            #     joint_states=None,
            #     tolerance=None,
            #     scale_factor=1.0,
            #     use_path_contraint=None,
            # )

            print(target_pose)

            control_success = self.control(
                header=header,
                target_pose=target_pose,
                joint_states=None,
                tolerance=0.01,
                scale_factor=1.0,
                use_path_contraint=False,
            )

            self._target_pose_pub.publish(PoseStamped(header=header, pose=target_pose))

            if control_success:
                # self.action_selecting(header=header)
                return control_success

        except ValueError as ve:
            self._node.get_logger().warn(f"Value Error: {ve}")

        except Exception as e:
            self._node.get_logger().error(f"Unexpected Error: {e}")
            self._node.get_logger().error("Target Aiming Failed")

        return False

    def target_aiming2(self, header: Header):
        """
        Run kinematic path service to get the target object pose.
        Target pose is the pose which is located in front of the target object
        """
        try:
            target_pose: Pose = self._control_action.target_object.pose
            target_pose = Pose(
                position=Point(
                    x=target_pose.position.x,
                    y=target_pose.position.y - 0.1,
                    z=self._target_z,
                ),
                orientation=self._home_pose.pose.orientation,
            )

            control_success = self.control_caterian_path(
                header=header,
                target_pose=target_pose,
                joint_states=None,
                tolerance=None,
                scale_factor=0.5,
                use_path_contraint=None,
            )

            # control_success = self.control(
            #     header=header,
            #     target_pose=target_pose,
            #     joint_states=None,
            #     tolerance=0.01,
            #     scale_factor=1.0,
            #     use_path_contraint=False,
            # )

            self._target_pose_pub.publish(PoseStamped(header=header, pose=target_pose))

            if control_success:
                # self.action_selecting(header=header)
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
            target_pose = Pose(
                position=Point(
                    x=target_pose.position.x,
                    y=target_pose.position.y,
                    z=self._target_z,
                ),
                orientation=self._home_pose.pose.orientation,
            )

            control_success = self.control_caterian_path(
                header=header,
                target_pose=target_pose,
                joint_states=None,
                tolerance=None,
                scale_factor=0.2,
                use_path_contraint=None,
            )

            # control_success = self.control(
            #     header=header,
            #     target_pose=target_pose,
            #     joint_states=None,
            #     tolerance=0.01,
            #     scale_factor=0.2,
            #     use_path_contraint=False,
            # )

            self._target_pose_pub.publish(PoseStamped(header=header, pose=target_pose))

            if control_success:
                # self.action_selecting(header=header)
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

            target_pose = Pose(
                position=Point(
                    x=target_pose.position.x + 0.03,
                    y=home_pose.position.y + 0.03,
                    z=target_pose.position.z + 0.03,
                ),
                orientation=self._home_pose.pose.orientation,
            )

            control_success = self.control_caterian_path(
                header=header,
                target_pose=target_pose,
                joint_states=None,
                tolerance=None,
                scale_factor=1.0,
                use_path_contraint=None,
            )

            # control_success = self.control(
            #     header=header,
            #     target_pose=target_pose,
            #     joint_states=None,
            #     tolerance=0.01,
            #     scale_factor=1.0,
            #     use_path_contraint=False,
            # )

            self._target_pose_pub.publish(PoseStamped(header=header, pose=target_pose))

            if control_success:
                # self.action_selecting(header=header)
                return control_success

        except ValueError as ve:
            self._node.get_logger().warn(f"Value Error: {ve}")

        except Exception as e:
            self._node.get_logger().error(f"Unexpected Error: {e}")
            self._node.get_logger().error("Home Aiming Failed")

        return False

    def home_up_positioning(self, header: Header):
        """
        Run kinematic path service to get the target object pose.
        Target pose is the pose which is located the front and above the target object.
        """
        try:
            if self._state == State.GARSPING_HOMING2:
                if int(self._control_action.target_id[1]) == 0:
                    return True

            target_pose: Pose = self._control_action.target_object.pose
            home_pose: Pose = self._home_pose.pose

            target_pose = Pose(
                position=Point(
                    x=home_pose.position.x,
                    y=home_pose.position.y,
                    z=home_pose.position.z + 0.03,
                ),
                orientation=self._home_pose.pose.orientation,
            )

            control_success = self.control(
                header=header,
                target_pose=target_pose,
                joint_states=None,
                tolerance=0.01,
                scale_factor=1.0,
                use_path_contraint=False,
            )

            self._target_pose_pub.publish(PoseStamped(header=header, pose=target_pose))

            if control_success:
                # self.action_selecting(header=header)
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
            self._drop_grid_manager.publish_grid_marker()

            control_success = self.control(
                header=header,
                target_pose=None,  # To ignore the target pose
                joint_states=self._dropping_joints,
                tolerance=0.01,
                scale_factor=1.0,
                use_path_contraint=False,
            )

            if control_success:
                # self.action_selecting(header=header)
                return control_success

        except ValueError as ve:
            self._node.get_logger().warn(f"Value Error: {ve}")

        except Exception as e:
            self._node.get_logger().error(f"Unexpected Error: {e}")
            self._node.get_logger().error("Drop Positioning Failed")

        return False

    def drop_table_positioning(self, header: Header):
        try:
            self._drop_grid_manager.publish_grid_marker()

            # >>> STEP 1. Get Empty Drop Grid >>>
            empty_grid: GridManager.Grid = self._drop_grid_manager.get_target_grid()

            # >>> STEP 2. Transform the drop pose >>>
            target_position: Point = empty_grid.center_coord
            target_pose = Pose(
                position=target_position,
                orientation=Quaternion(x=0.0, y=0.0, z=0.0, w=1.0),
            )
            transformed_target_pose: PoseStamped = (
                self._transform_manager.transform_pose(
                    pose=target_pose,
                    source_frame="camera1_link",
                    target_frame="world",
                )
            )
            final_target_pose = Pose(
                position=transformed_target_pose.pose.position,
                orientation=self._drop_pose.pose.orientation,  # TODO: Check the orientation
            )

            self._drop_rollback_pose = final_target_pose

            # >>> STEP 3. Plan & Execute Trajectory >>>
            is_success = self.control(
                header=header,
                target_pose=final_target_pose,
                joint_states=None,
                tolerance=0.001,
                scale_factor=1.0,
                use_path_contraint=False,
            )

            if is_success:
                # >>> STEP 4. Update the drop grid >>>
                self._drop_grid_manager.set_grid_dropped(
                    col=empty_grid.col, row=empty_grid.row
                )

                # >>> STEP 5. Get collision object >>>
                collision_object = (
                    self._apply_planning_scene_service_manager.create_collision_object(
                        id=f"drop{empty_grid.row}{empty_grid.col}",
                        header=header,
                        pose=final_target_pose,
                        scale=Vector3(x=0.03, y=0.50, z=0.03),
                        operation=CollisionObject.ADD,
                    )
                )
                collision_objects = self._drop_grid_manager.append_collision_object(
                    collision_object=collision_object,
                )

                for co in collision_objects:
                    self._node.get_logger().warn(
                        f"Collision Object: {co.id}, {co.primitive_poses[0].position.x:.3f}"
                    )

                # >>> STEP 6. Update the planning scene >>>
                current_scene = self._get_planning_scene_service_manager.run()
                self._apply_planning_scene_service_manager.add_collistion_objects(
                    collision_objects=collision_objects,
                    scene=current_scene,
                )

                self._node.get_logger().info(
                    f"Drop -> {empty_grid.row}{empty_grid.col}"
                )
                # self.action_selecting(header=header)
                return True

        except ValueError as ve:
            self._node.get_logger().warn(f"Value Error: {ve}")

        except Exception as e:
            self._node.get_logger().error(f"Unexpected Error: {e}")
            self._node.get_logger().error("Drop Table Positioning Failed")

    def drop_rollback(self, header: Header):
        try:
            drop_pose: Pose = self._drop_rollback_pose

            control_success = self.control(
                header=header,
                target_pose=Pose(
                    position=Point(
                        x=drop_pose.position.x + 0.1,
                        y=drop_pose.position.y,
                        z=drop_pose.position.z,
                    ),
                    orientation=drop_pose.orientation,
                ),
                joint_states=None,
                tolerance=0.01,
                scale_factor=1.0,
                use_path_contraint=False,
            )

            if control_success:
                # self.action_selecting(header=header)
                return True

        except ValueError as ve:
            self._node.get_logger().warn(f"Value Error: {ve}")
        except Exception as ex:
            self._node.get_logger().error(f"Unexpected Error: {ex}")

        return False

    # >>> LEVEL 4 >>>
    def sweep_target_aiming(self, header: Header):
        """
        Run kinematic path service to get the target object pose.
        Target pose is the pose which is the front/side of the target object
        """
        try:
            target_pose: Pose = self._control_action.target_object.pose

            target_col = int(self._control_action.target_id[1])  # e.g. 'A1' -> 1

            moving_grid_id = self._control_action.goal_ids[-1]
            moving_col = int(moving_grid_id[1])  # e.g. 'A2' -> 2

            direction = target_col < moving_col  # True for right, False for left

            offset = -0.07 if direction else 0.07

            self._moving_col = moving_col

            target_pose = Pose(
                position=Point(
                    x=target_pose.position.x + offset,
                    y=self._home_pose.pose.position.y + 0.04,
                    z=self._target_z + 0.05,
                ),
                orientation=self._home_pose.pose.orientation,
            )

            self._target_pose_pub.publish(PoseStamped(header=header, pose=target_pose))

            control_success = self.control(
                header=header,
                target_pose=target_pose,
                joint_states=None,
                tolerance=0.01,
                scale_factor=1.0,
                use_path_contraint=False,
            )

            if control_success:
                # self.action_selecting(header=header)
                self._planning_attempt = 0
                return control_success

        except ValueError as ve:
            self._node.get_logger().warn(f"Value Error: {ve}")
            self._planning_attempt += 1

        except Exception as e:
            self._node.get_logger().error(f"Unexpected Error: {e}")
            self._node.get_logger().error("Target Aiming Failed")
            self._planning_attempt += 1

        return False

    def sweep_target_aiming2(self, header: Header):
        """
        Run kinematic path service to get the target object pose.
        Target pose is the pose which is the front/side of the target object
        """
        try:
            target_pose: Pose = self._control_action.target_object.pose

            target_col = int(self._control_action.target_id[1])  # e.g. 'A1' -> 1

            moving_grid_id = self._control_action.goal_ids[-1]
            moving_col = int(moving_grid_id[1])  # e.g. 'A2' -> 2

            # if self._planning_attempt // 2 == 0:
            #     moving_col = max(
            #         [int(id[1]) for id in self._control_action.goal_ids]
            #     )  # e.g. ['A0', 'A2'] -> [0, 2]
            # else:
            #     moving_col = min([int(id[1]) for id in self._control_action.goal_ids])

            direction = target_col < moving_col  # True for right, False for left

            offset = -0.1 if direction else 0.1

            self._moving_col = moving_col

            target_pose = Pose(
                position=Point(
                    x=target_pose.position.x + offset,
                    y=target_pose.position.y - 0.1,
                    z=self._target_z + 0.05,
                ),
                orientation=self._home_pose.pose.orientation,
            )

            self._target_pose_pub.publish(PoseStamped(header=header, pose=target_pose))

            control_success = self.control(
                header=header,
                target_pose=target_pose,
                joint_states=None,
                tolerance=0.01,
                scale_factor=1.0,
                use_path_contraint=False,
            )

            if control_success:
                # self.action_selecting(header=header)
                self._planning_attempt = 0
                return control_success

        except ValueError as ve:
            self._node.get_logger().warn(f"Value Error: {ve}")
            self._planning_attempt += 1

        except Exception as e:
            self._node.get_logger().error(f"Unexpected Error: {e}")
            self._node.get_logger().error("Target Aiming Failed")
            self._planning_attempt += 1

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

            target_pose = Pose(
                position=Point(
                    x=target_pose.position.x + offset,
                    y=target_pose.position.y,
                    z=target_pose.position.z,
                ),
                orientation=(
                    self._sweeping_to_right_pose.pose.orientation
                    if direction
                    else self._sweeping_to_left_pose.pose.orientation
                ),
            )

            self._target_pose_pub.publish(PoseStamped(header=header, pose=target_pose))

            control_success = self.control(
                header=header,
                target_pose=target_pose,
                joint_states=None,
                tolerance=0.01,
                scale_factor=1.0,
                use_path_contraint=False,
            )

            if control_success:
                # self.action_selecting(header=header)
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
            current_scene = self._get_planning_scene_service_manager.run()
            if self._apply_planning_scene_service_manager.reset_planning_scene(
                scene=current_scene
            ):
                current_scene = self._get_planning_scene_service_manager.run()
                if self._apply_planning_scene_service_manager.append_default_collision_objects(
                    header=header,
                    scene=current_scene,
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
                    sweep_offset = -0.05  # if direction else -0.05
                    sweep_distance = (
                        self._object_selection_manager.get_grid_data()[
                            "grid_identifier"
                        ]["grid_size"]["y"]
                        + sweep_offset
                    )

                    offset = sweep_distance if direction else -sweep_distance

                    self._node.get_logger().info(f"Sweep Offset: {offset}")

                    target_pose = Pose(
                        position=Point(
                            x=target_pose.position.x + offset,
                            y=target_pose.position.y,
                            z=target_pose.position.z,
                        ),
                        orientation=(
                            self._sweeping_to_right_pose.pose.orientation
                            if direction
                            else self._sweeping_to_left_pose.pose.orientation
                        ),
                    )

                    control_success = self.control(
                        header=header,
                        target_pose=target_pose,
                        joint_states=None,
                        tolerance=0.01,
                        scale_factor=0.2,
                        use_path_contraint=False,
                    )

                    if control_success:
                        # self.action_selecting(header=header)
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
        if self._debug:
            # self.action_selecting(header=header)
            return True

        self._gripper_action_manager.control_gripper(open=False)
        if self._gripper_action_manager.is_finished is True:
            # self.action_selecting(header=header)
            time.sleep(3.0)
            return True

        return False

    def ungrasping(self, header: Header):
        """
        Run gripper action to ungrasp the target object
        """
        if self._debug:
            # self.action_selecting(header=header)
            return True

        try:
            self._gripper_action_manager.control_gripper(open=True)
            if self._gripper_action_manager.is_finished is True:
                # self.action_selecting(header=header)
                return True

        except ValueError as ve:
            self._node.get_logger().warn(f"Value Error: {ve}")

        except Exception as e:
            self._node.get_logger().error(f"Unexpected Error: {e}")
            self._node.get_logger().error("Ungrasing Failed")

        return False

    # >>> ETC >>>
    def control_caterian_path(
        self,
        header: Header,
        target_pose: Pose,
        joint_states: JointState,
        tolerance: float,
        scale_factor: float,
        use_path_contraint: bool = False,
    ):
        # >>> STEP 0. Exception handling >>>
        if target_pose is None:
            raise ValueError("Target_pose must be provided.")

        try:
            # >>> STEP 4. Get the current planning scene >>>
            trajectory: RobotTrajectory = self._cartesian_path_service_manager.run(
                header=header,
                waypoints=[target_pose],
                joint_states=self._joint_states_manager.joint_states,
                end_effector=self._end_effector,
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
            path_constraint = Constraints(
                name="path_constraint",
                orientation_constraints=[
                    OrientationConstraint(
                        header=header,
                        link_name=self._end_effector,
                        orientation=self._home_pose.pose.orientation,
                        absolute_x_axis_tolerance=0.3,
                        absolute_y_axis_tolerance=0.3,
                        absolute_z_axis_tolerance=0.3,
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

    from rclpy.utilities import remove_ros_args
    from base_package.header import str2bool

    # Remove ROS2 arguments
    argv = remove_ros_args(sys.argv)

    parser = argparse.ArgumentParser(description="FCN Server Node")

    parser.add_argument(
        "--model_file",
        type=str,
        required=True,
        help="Path or file name of the trained FCN model. If input is a file name, the file should be located in the 'resource' directory. Required",
    )
    parser.add_argument(
        "--grid_data_file",
        type=str,
        required=False,
        default="grid_data.json",
        help="Path or file name of object bounds. If input is a file name, the file should be located in the 'resource' directory. Required",
    )
    parser.add_argument(
        "--drop_grid_data_file",
        type=str,
        required=False,
        default="drop_grid_data.json",
        help="Path or file name of object bounds. If input is a file name, the file should be located in the 'resource' directory. Required",
    )

    parser.add_argument(
        "--target_cls",
        type=str,
        required=True,
        help="Target class to search. Required",
    )

    # >>> Closest Object Classifier >>>
    parser.add_argument(
        "--debug",
        type=str2bool,
        required=False,
        default=False,
    )
    parser.add_argument(
        "--mode",
        type=int,
        required=True,
        default=0,
        help="0: FCN(Grasp only) -> grasp only model 1: FCN(Rule-base) -> 0408 model, 2: DRL -> 0408, 3: Random",
    )

    parser.add_argument(
        "--threshold",
        type=int,
        required=False,
        default=50,
    )
    parser.add_argument(
        "--fcn_image_transform", type=bool, required=False, default=True
    )
    parser.add_argument(
        "--fcn_gain",
        type=float,
        required=False,
        default=2.0,
    )
    parser.add_argument(
        "--fcn_gamma",
        type=float,
        required=False,
        default=0.7,
    )

    args = parser.parse_args(argv[1:])
    kagrs = vars(args)

    print(kagrs)

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
