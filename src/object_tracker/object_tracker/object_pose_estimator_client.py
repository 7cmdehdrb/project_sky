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
            self._node.get_name() + "/megapose_markers",
            qos_profile=qos_profile_system_default,
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

    @staticmethod
    def parse_resonse_to_marker_array(
        response: BoundingBox3DMultiArray, header: Header
    ) -> MarkerArray:
        marker_array = MarkerArray()

        for id, bbox3d in enumerate(response.data):
            bbox3d: BoundingBox3D

            marker = Marker()
            marker.ns = bbox3d.cls
            marker.id = id
            marker.header = header
            marker.type = Marker.CUBE
            marker.action = Marker.ADD
            marker.pose = bbox3d.pose
            marker.scale = bbox3d.scale
            marker.color = ColorRGBA(r=0.0, g=1.0, b=0.0, a=1.0)
            marker.lifetime = ROS2Duration(sec=5, nanosec=0)
            marker_array.markers.append(marker)

        return marker_array


def main(args=None):
    rclpy.init(args=args)

    node = Node("object_pose_estimator_client")

    megapose_client = MegaPoseClient(node)

    # Spin in a separate thread
    thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    thread.start()

    hz = 0.2
    rate = node.create_rate(hz)

    try:
        while rclpy.ok():
            response = megapose_client.send_megapose_request()
            header = Header(
                stamp=node.get_clock().now().to_msg(), frame_id="camera1_link"
            )

            for _ in range(10):
                megapose_client.marker_array_pub.publish(
                    megapose_client.parse_resonse_to_marker_array(
                        response,
                        header,
                    )
                )

            rate.sleep()
    except KeyboardInterrupt:
        pass

    node.destroy_node()

    rclpy.shutdown()
    thread.join()


if __name__ == "__main__":
    main()
