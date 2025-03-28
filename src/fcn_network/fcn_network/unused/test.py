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

# TF
from tf2_ros import *

# Python
import os
import sys
import time

import argparse
import numpy as np
from ament_index_python.packages import get_package_share_directory
from types import SimpleNamespace as NameSpace


class Test(Node):
    def __init__(self, *args, **kwargs):
        super().__init__("test")

        self.model_path = kwargs["model_path"]
        print(self.model_path)

        self.create_timer(1.0, self.timer_callback)

    def timer_callback(self):
        try:
            package_path = get_package_share_directory("fcn_network")
            self.get_logger().info(f"Absolute path of 'fcn_network': {package_path}")
        except Exception as e:
            self.get_logger().error(f"Error finding package 'fcn_network': {str(e)}")


def main():
    rclpy.init(args=None)

    parser = argparse.ArgumentParser(description="FCN Server Node")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the trained FCN model file",
    )

    args: argparse.Namespace = parser.parse_args()
    kargs = vars(args)

    node = Test(**kargs)

    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
