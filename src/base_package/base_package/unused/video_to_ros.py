import cv2

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
import sys
import os
import numpy as np
from enum import Enum
import json
import argparse
import time
import subprocess

# custom
from base_package.manager import ImageManager


class ActionCameraNode(object):
    def __init__(self, node: Node, *args, **kwargs):
        self._node = node

        self._topic = kwargs.get("topic", "/action_camera/color/image_raw")
        self._video = kwargs.get("video", "/dev/video8")

        self._image_manager = ImageManager(
            self._node,
            subscribed_topics=[],
            published_topics=[
                {"topic_name": self._topic},
            ],
            *args,
            **kwargs,
        )

        self._last_time = self._node.get_clock().now()

        self.configure_camera_format(self._video, 1280, 720)

        self._cap = cv2.VideoCapture(self._video, cv2.CAP_V4L2)

        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self._cap.set(cv2.CAP_PROP_FPS, 30)

        self._timer = self._node.create_timer(0.01, self.run)

    def configure_camera_format(self, device_path, width, height):
        cmd = [
            "v4l2-ctl",
            f"--device={device_path}",
            f"--set-fmt-video=width={width},height={height},pixelformat=MJPG",
        ]
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if result.returncode != 0:
            self._node.get_logger().warn(
                "Failed to set MJPEG format. You may get low FPS."
            )
        else:
            self._node.get_logger().info("Camera format set to MJPEG.")

    def shutdown(self):
        self._cap.release()

    def run(self):
        if not self._cap.isOpened():
            self._node.get_logger().warn("Cannot open camera stream!")
            return None

        current_time = self._node.get_clock().now()

        dt = (current_time - self._last_time).nanoseconds / 1e9

        self._last_time = current_time

        ret, frame = self._cap.read()

        print(f"Frame shape: {frame.shape}")

        if not ret:
            self._node.get_logger().warn("Cannot read frame!")
            return None

        print(f"FPS: {1/dt:.2f}")

        image_msg = self._image_manager.encode_message(frame, encoding="bgr8")

        self._image_manager.publish(self._topic, image_msg)

        # cv2.imshow("frame", frame)
        # cv2.waitKey(1)


def main(args=None):
    rclpy.init(args=args)

    from rclpy.utilities import remove_ros_args
    from base_package.header import str2bool

    # Remove ROS2 arguments
    argv = remove_ros_args(sys.argv)

    parser = argparse.ArgumentParser(description="FCN Server Node")

    parser.add_argument(
        "--video",
        type=str,
        required=True,
        help="Video file path to publish the image. Required",
    )

    parser.add_argument(
        "--topic",
        type=str,
        required=True,
        help="Topic name to publish the image. Required",
    )

    args = parser.parse_args(argv[1:])
    kagrs = vars(args)

    node = rclpy.create_node("action_camera_node")

    action_camera_node = ActionCameraNode(node, **kagrs)

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        action_camera_node.shutdown()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
