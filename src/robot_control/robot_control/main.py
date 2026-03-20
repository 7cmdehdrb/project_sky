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
from abc import ABC, abstractmethod

# Custom
from rotutils import *
from robot_control.action_sequence import (
    ActionSequence,
    GraspActionSequence,
    SweepLeftActionSequence,
    SweepRightActionSequence,
)
from robot_control.controller import UR5eController


class MainControlNode(Node):
    def __init__(self):
        super().__init__("main_control_node")

        # UR5eController 인스턴스
        self._ur5e_controller = UR5eController(node=self)

        # ActionSequence 인스턴스
        self._grasp_action_sequence = GraspActionSequence(
            node=self, controller=self._ur5e_controller, target_point=None
        )
        self._sweep_left_action_sequence = SweepLeftActionSequence(
            node=self, controller=self._ur5e_controller, target_point=None
        )
        self._sweep_right_action_sequence = SweepRightActionSequence(
            node=self, controller=self._ur5e_controller, target_point=None
        )


def main(args=None):
    rclpy.init(args=args)

    node = MainControlNode()

    th = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    th.start()

    hz = 30.0
    r = node.create_rate(hz)
    try:
        while rclpy.ok():
            r.sleep()
    except KeyboardInterrupt:
        node.get_logger().info("KeyboardInterrupt received, shutting down.")
    except Exception as e:
        node.get_logger().error(f"Exception in main loop: {e}")
    finally:
        th.join(timeout=1.0)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
