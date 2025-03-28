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
from builtin_interfaces.msg import Duration as BuiltinDuration

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
import os
import sys
import numpy as np
import time
import tqdm

# Custom
from object_tracker.real_time_tracking_client import MegaPoseClient
from base_package.header import QuaternionAngle
from object_tracker.real_time_segmentation import RealTimeSegmentationNode


class MegaposeRequestNode(Node):
    def __init__(self):
        super().__init__("megapose_request_node")
        self.cli = self.create_client(MegaposeRequest, "/megapose_request")

        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info("Service not available, waiting...")
        self.req = MegaposeRequest.Request()

        self.pub = self.create_publisher(
            MarkerArray, "/megapose_markers", qos_profile=qos_profile_system_default
        )
        self.pub2 = self.create_publisher(
            PoseArray, "/megapose_poses", qos_profile=qos_profile_system_default
        )
        self.response: BoundingBox3DMultiArray = None

        self.send_request()
        self.timer = self.create_timer(1.0, self.publish_markers)

    def publish_markers(self):
        markers = MarkerArray()
        pose_array = PoseArray()
        pose_array.header.frame_id = "camera1_link"
        pose_array.header.stamp = self.get_clock().now().to_msg()

        if self.response is None:
            self.get_logger().warn("No response received yet")
            return

        for i, bb in enumerate(self.response.data):
            bb: BoundingBox3D

            marker = Marker()
            marker.header.frame_id = "camera1_link"
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.id = i
            marker.ns = bb.cls
            marker.type = Marker.CUBE
            marker.action = Marker.ADD
            marker.pose.position = Point(
                x=bb.pose.position.x,
                y=bb.pose.position.y,
                z=bb.pose.position.z,
            )
            marker.pose.orientation = bb.pose.orientation
            marker.scale = Vector3(x=bb.scale.x, y=bb.scale.y, z=bb.scale.z)
            marker.color = ColorRGBA(r=1.0, g=0.0, b=0.0, a=0.5)
            marker.lifetime = BuiltinDuration(sec=1, nanosec=0)
            markers.markers.append(marker)

            pose_array.poses.append(bb.pose)

        self.pub.publish(markers)
        self.pub2.publish(pose_array)

    def send_request(self):
        # Populate the request with appropriate data
        self.future = self.cli.call_async(self.req)
        self.future.add_done_callback(self.response_callback)

    def response_callback(self, future: Future):
        try:
            response: MegaposeRequest.Response = future.result()
            self.response = response.response
            self.get_logger().info(f"Received response: {response.response}")
        except Exception as e:
            self.get_logger().error(f"Service call failed: {e}")


def main(args=None):
    rclpy.init(args=args)
    node = MegaposeRequestNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
