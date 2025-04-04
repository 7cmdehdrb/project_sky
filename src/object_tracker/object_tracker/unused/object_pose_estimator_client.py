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
from builtin_interfaces.msg import Duration as ROS2Duration
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
        self.marker_array_pub = self._node.create_publisher(
            MarkerArray,
            self._node.get_name() + "/megapose/marker_array",
            qos_profile=qos_profile_system_default,
        )
        self._unique_key = {
            "cup_1": 0,
            "cup_2": 1,
            "cup_3": 2,
            "mug_1": 3,
            "mug_2": 4,
            "mug_3": 5,
            "bottle_1": 6,
            "bottle_2": 7,
            "bottle_3": 8,
            "can_1": 9,
            "can_2": 10,
            "can_3": 11,
        }

        while not self.megapose_client.wait_for_service(timeout_sec=1.0):
            self._node.get_logger().warn(
                f"Service called megapose_request not available, waiting again..."
            )

        self._node.get_logger().info("Service is available.")

    def send_megapose_request(self) -> BoundingBox3DMultiArray:
        request = MegaposeRequest.Request()
        response: MegaposeRequest.Response = self.megapose_client.call(request)
        return response.response

    def post_process_response(
        self, response: BoundingBox3DMultiArray, header: Header
    ) -> bool:
        marker_array = self.parse_resonse_to_marker_array(
            response,
            header,
        )
        self.marker_array_pub.publish(marker_array)

    def parse_resonse_to_marker_array(
        self, response: BoundingBox3DMultiArray, header: Header
    ) -> MarkerArray:
        marker_array = MarkerArray()

        for id, bbox3d in enumerate(response.data):
            bbox3d: BoundingBox3D

            marker = Marker()
            marker.ns = bbox3d.cls
            marker.id = int(self._unique_key[bbox3d.cls])
            marker.header = header
            marker.type = Marker.CUBE
            marker.action = Marker.ADD
            marker.pose = bbox3d.pose
            marker.scale = bbox3d.scale
            marker.color = ColorRGBA(r=0.0, g=1.0, b=0.0, a=1.0)
            # marker.lifetime = ROS2Duration(sec=10, nanosec=0)
            marker_array.markers.append(marker)

        return marker_array
