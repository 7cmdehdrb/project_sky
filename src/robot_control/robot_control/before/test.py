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
)


class Test(object):
    def __init__(self, node: Node, *args, **kwargs):
        self._node = node

        self._fk_manager = FK_ServiceManager(self._node, *args, **kwargs)

        self._joint_state_sub = self._node.create_subscription(
            JointState,
            "/joint_states",
            self.joint_state_callback,
            qos_profile_system_default,
        )
        self._joint_states: JointState = None

    def run(self):
        if self._joint_states is None:
            return

        # Example of using FK_ServiceManager
        result = self._fk_manager.run(
            joint_states=self._joint_states,
            end_effector="tool0",
        )

        if result:
            self._node.get_logger().info("FK calculation successful.")
            print(result)
        else:
            self._node.get_logger().error("FK calculation failed.")

    def joint_state_callback(self, msg):
        # Process the joint state message
        self._joint_states = msg


def main(args=None):
    rclpy.init(args=args)

    node = Node("main_control_node")
    main_node = Test(node=node)

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


if __name__ == "__main__":
    main()
