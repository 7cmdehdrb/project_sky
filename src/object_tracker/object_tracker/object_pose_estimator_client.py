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
from custom_msgs.msg import (
    BoundingBox,
    BoundingBoxMultiArray,
    BoundingBox3D,
    BoundingBox3DMultiArray,
)
from custom_msgs.srv import MegaposeRequest

# TF
from tf2_ros import *

# Megapose Server
import socket
import struct
import json
import io
import time
from cv_bridge import CvBridge
import cv2

# Python
import numpy as np
import time

# Custom
from base_package.header import QuaternionAngle


class MegaPoseClient(object):
    def __init__(self, node: Node):
        self._node = node

        self.megapose_client = self._node.create_client(
            MegaposeRequest, "/megapose_request", qos_profile=qos_profile_system_default
        )

        while not self.megapose_client.wait_for_service(timeout_sec=1.0):
            self._node.get_logger().warn(
                f"Service called megapose_request not available, waiting again..."
            )

        self._node.get_logger().info("Service is available.")

    def send_megapose_request(self) -> BoundingBox3DMultiArray:
        request = MegaposeRequest.Request()
        response: MegaposeRequest.Response = self.megapose_client.call(request)
        return response.response
