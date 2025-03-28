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
from custom_msgs.msg import BoundingBox, BoundingBoxMultiArray


# TF
from tf2_ros import *

# Python
import numpy as np
from PIL import ImageEnhance
from PIL import Image as PILImage

# OpenCV
import cv2
from cv_bridge import CvBridge


# Megapose Server
import socket
import struct
import json
import io
import time

from base_package.header import QuaternionAngle


class RealTimeTrackingClientNode(Node):
    def __init__(self):
        super().__init__("real_time_tracking_client_node")

        self.megapose_client = MegaPoseClient(node=self)
        self.bridge = CvBridge()

        self.do_publish_image = True
        self.width, self.height = 640, 480

        # ROS
        self.image_subscriber = self.create_subscription(
            Image,
            "/camera/camera1/color/image_raw",
            self.image_callback,
            qos_profile=qos_profile_system_default,
        )
        self.camera_info_subscriber = self.create_subscription(
            CameraInfo,
            "/camera/camera1/color/camera_info",
            self.camera_info_callback,
            qos_profile=qos_profile_system_default,
        )
        self.pose_publisher = self.create_publisher(
            PoseStamped,
            self.get_name() + "/megapose",
            qos_profile=qos_profile_system_default,
        )
        self.image_publisher = self.create_publisher(
            Image,
            self.get_name() + "/megapose_image",
            qos_profile=qos_profile_system_default,
        )

        self.frame = None

        self.timer = self.create_timer(0.05, self.run)

    def image_callback(self, msg: Image):
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")

        width = frame.shape[1]
        height = frame.shape[0]
        print(f"Image shape: {frame.shape[:2]}")

        if width == 1280 and height == 720:
            self.frame = self.crop_and_resize_image(frame)

        elif width == self.width and height == self.height:
            self.frame = frame

        else:
            self.get_logger().warn("Invalid image size. Cannot set frame.")

    def camera_info_callback(self, msg: CameraInfo):
        K = np.array(msg.k).reshape(3, 3)
        image_size = (msg.height, msg.width)

        if msg.height == 720 and msg.width == 1280:
            image_size = (self.height, self.width)

            offset = int((msg.width - self.width) // 2)

            K[0, 0] = K[0, 0] * (self.width / msg.width)
            K[1, 1] = K[1, 1] * (self.height / msg.height)
            K[0, 2] = (K[0, 2] - offset) * (self.width / msg.width)
            K[1, 2] = K[1, 2] * (self.height / msg.height)

        elif msg.height == self.height and msg.width == self.width:
            # Do nothing
            pass

        else:
            # Prevent setting intrinsics
            self.get_logger().warn("Invalid image size. Cannot set intrinsics.")
            return None

        self.megapose_client.set_intrinsics(
            K=K,
            image_size=image_size,
        )

    def run(self):
        # while True:
        if self.frame is None:
            self.get_logger().warn("Frame does not exist.")
            return None

        pose_msg, bbox = self.megapose_client.run(frame=self.frame)

        if pose_msg is not None:
            self.pose_publisher.publish(pose_msg)

        if bbox is not None and self.do_publish_image:
            cv2.rectangle(
                self.frame,
                (int(bbox[0]), int(bbox[1])),
                (int(bbox[2]), int(bbox[3])),
                (0, 255, 0),
                2,
            )

            image_msg = self.bridge.cv2_to_imgmsg(self.frame, encoding="bgr8")
            self.image_publisher.publish(image_msg)

    def crop_and_resize_image(self, image: np.array):
        """Crop 1280x720 image to 640x480."""
        image_height, image_width = image.shape[:2]

        assert image_height == 720 and image_width == 1280

        crop_ratio = self.width / self.height

        new_width = int(image_height * crop_ratio)
        offset = (image_width - new_width) // 2
        cropped_image = image[:, offset : offset + new_width]

        return cv2.resize(cropped_image, (self.width, self.height))


def main(args=None):
    rclpy.init(args=args)

    node = RealTimeTrackingClientNode()

    rclpy.spin(node=node)

    cv2.destroyAllWindows()

    node.destroy_node()


if __name__ == "__main__":
    main()
