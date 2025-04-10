# Python
import os
import sys
import json
import numpy as np
import argparse
import array

# ROS2
import rclpy
from rclpy.node import Node
from rclpy.time import Time
from rclpy.duration import Duration
from rclpy.qos import QoSProfile, qos_profile_system_default
import time

# Message
from std_msgs.msg import *
from geometry_msgs.msg import *
from sensor_msgs.msg import *
from nav_msgs.msg import *
from visualization_msgs.msg import *

# TF
from tf2_ros import *


class IntegratedJointStateBroadcaster(object):
    def __init__(self, node: Node, *args, **kwargs):
        self._node = node

        # >>> Data >>>
        self._ur5e_joint_state: JointState = None
        self._gripper_joint_state: JointState = None
        # <<< Data <<<

        # >>> Subscriber & Publisher >>>
        self._ur5e_joint_state_subscriber = self._node.create_subscription(
            JointState,
            "/ur5e/joint_states",
            self._ur_joint_state_callback,
            qos_profile=qos_profile_system_default,
        )
        self._gripper_joint_state_subscriber = self._node.create_subscription(
            JointState,
            "/gripper/joint_states",
            self._robotiq_joint_state_callback,
            qos_profile=qos_profile_system_default,
        )

        self._joint_state_publisher = self._node.create_publisher(
            JointState, "/joint_states", qos_profile=qos_profile_system_default
        )
        # <<< Subscriber & Publisher <<<

        self._timer = self._node.create_timer(0.01, self.run)

    def run(self):
        if self._ur5e_joint_state is None:
            # self._node.get_logger().warn("Waiting for UR5e joint state...")
            # time.sleep(0.1)
            return None

        if self._gripper_joint_state is None:
            # self._node.get_logger().warn("Waiting for Gripper joint state...")
            # time.sleep(0.1)
            return None

        # >>> Joint State >>>
        """
        std_msgs/Header header
        string[] name
        float64[] position
        float64[] velocity
        float64[] effort
        """
        integrated_joint_states = JointState(
            header=Header(
                stamp=self._node.get_clock().now().to_msg(),
            ),
            name=self._ur5e_joint_state.name + self._gripper_joint_state.name,
            position=self._ur5e_joint_state.position
            + self._gripper_joint_state.position,
            velocity=self._ur5e_joint_state.velocity
            + self._gripper_joint_state.velocity,
            effort=self._ur5e_joint_state.effort + self._gripper_joint_state.effort,
        )

        self._joint_state_publisher.publish(integrated_joint_states)

    def _ur_joint_state_callback(self, msg: JointState):
        self._ur5e_joint_state = msg

    def _robotiq_joint_state_callback(self, msg: JointState):
        """
        <joint name="robotiq_85_left_knuckle_joint" type="revolute">
        <joint name="robotiq_85_right_knuckle_joint" type="revolute">
        <joint name="robotiq_85_left_inner_knuckle_joint" type="continuous">
        <joint name="robotiq_85_right_inner_knuckle_joint" type="continuous">
        <joint name="robotiq_85_left_finger_tip_joint" type="continuous">
        <joint name="robotiq_85_right_finger_tip_joint" type="continuous">
        """
        gripper_names = [
            "robotiq_85_left_knuckle_joint",
            # "robotiq_85_right_knuckle_joint",
            # "robotiq_85_left_inner_knuckle_joint",
            # "robotiq_85_right_inner_knuckle_joint",
            # "robotiq_85_left_finger_tip_joint",
            # "robotiq_85_right_finger_tip_joint",
        ]
        # gripper_positions = [msg.position[0] for _ in gripper_names]
        gripper_positions = [0.0 for _ in gripper_names]
        gripper_velocities = [0.0 for _ in gripper_names]
        gripper_efforts = [0.0 for _ in gripper_names]

        # Update the gripper joint state with the new values
        gripper_msg = JointState(
            header=msg.header,
            name=gripper_names,
            position=gripper_positions,
            velocity=gripper_velocities,
            effort=gripper_efforts,
        )

        # Publish the gripper joint state
        self._gripper_joint_state = gripper_msg

        # self._gripper_joint_state = msg


def main():
    rclpy.init()

    node = rclpy.create_node("integrated_joint_states_broadcaster")
    integrated_joint_state_broadcaster = IntegratedJointStateBroadcaster(node)

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
        sys.exit(0)


if __name__ == "__main__":
    main()
