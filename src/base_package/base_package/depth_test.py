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
import numpy as np
import cv2
import cv_bridge
import pandas as pd


class DepthTestNode(Node):
    def __init__(self):
        super().__init__("depth_test_node")

        self.bridge = cv_bridge.CvBridge()

        topic = "/camera/camera1/depth/image_raw"
        topic = "/camera/camera1/depth/image_rect_raw"
        self.depth_subscriber = self.create_subscription(
            Image,
            topic,
            self.depth_callback,
            qos_profile=qos_profile_system_default,
        )

    def depth_callback(self, msg: Image):
        depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        print(f"Depth image dtype: {depth_image.dtype}")


def main():
    rclpy.init(args=None)

    node = DepthTestNode()

    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
