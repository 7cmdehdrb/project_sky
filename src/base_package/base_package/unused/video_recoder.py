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
import cv2

from base_package.manager import ImageManager, Manager, ObjectManager


class VideoRecoderNode(object):
    def __init__(self, node: Node, *args, **kwargs):
        self._node = node

        self._attempt = kwargs.get("exp_attempt", -1)
        self._record = kwargs.get("record", True)

        self._image: Image = None
        self._image2: Image = None
        self._action_image: Image = None

        self._image_manager = ImageManager(
            self._node,
            subscribed_topics=[
                {
                    "topic_name": "/camera/camera1/color/image_raw",
                    "callback": self.image_callback,
                },
                {
                    "topic_name": "/action_camera/color/image_raw",
                    "callback": self.action_image_callback,
                },
                {
                    "topic_name": "/camera/camera2/color/image_raw",
                    "callback": self.image_callback2,
                },
            ],
            published_topics=[
                {
                    "topic_name": "/video_recoder/concatenate_image",
                }
            ],
        )

        fps = 30.0  # 프레임 속도

        if self._record:
            self._fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # (*'MP42')
            self._out = cv2.VideoWriter(
                f"/home/irol/workspace/project_sky/src/robot_control/resource/exp_result/{self._attempt}.mp4",
                self._fourcc,
                fps,
                (int(1280 * 3), 720),
            )

        self._timer = self._node.create_timer((1.0 / fps), self.run)

    def shutdown(self):
        """
        Shutdown function to release the video writer.
        """
        try:
            if self._out.isOpened():
                self._out.release()
                self._node.get_logger().info("Video writer released.")
            else:
                self._node.get_logger().warn("Video writer is not opened.")
        except Exception as e:
            self._node.get_logger().error(f"Failed to release video writer: {e}")

    def image_callback(self, msg: Image):
        """
        Callback function for the image.
        """
        self._image = msg

    def image_callback2(self, msg: Image):
        """
        Callback function for the second camera image.
        """
        self._image2 = msg

    def action_image_callback(self, msg: Image):
        """
        Callback function for the action camera image.
        """
        self._action_image = msg

    def run(self):

        if self._image is None or self._action_image is None or self._image2 is None:
            self._node.get_logger().warn("Image not received yet!")
            return None

        np_image = self._image_manager.decode_message(
            self._image,
            desired_encoding="bgr8",
        )

        np_image2 = self._image_manager.decode_message(
            self._image2,
            desired_encoding="bgr8",
        )

        np_action_image = self._image_manager.decode_message(
            self._action_image,
            desired_encoding="bgr8",
        )

        np_concatenate_image = np.concatenate(
            (np_image, np_image2, np_action_image),
            axis=1,
        )

        concatenated_image_msg = self._image_manager.encode_message(
            np_concatenate_image,
            encoding="bgr8",
        )

        frame = cv2.resize(np_concatenate_image, (int(1280 * 3), 720))

        # Write the frame to the video f
        if self._record:
            if self._out.isOpened():
                self._out.write(frame)
                self._node.get_logger().info(f"{frame.shape}")

        self._image_manager.publish(
            "/video_recoder/concatenate_image",
            concatenated_image_msg,
        )


def main(args=None):
    rclpy.init(args=args)

    import argparse
    from rclpy.utilities import remove_ros_args
    from base_package.header import str2bool

    # Remove ROS2 arguments
    argv = remove_ros_args(sys.argv)

    parser = argparse.ArgumentParser(description="FCN Server Node")

    parser.add_argument(
        "--exp_attempt",
        type=int,
        required=True,
    )
    parser.add_argument(
        "--record",
        type=str2bool,
        default=True,
    )

    args = parser.parse_args(argv[1:])
    kagrs = vars(args)

    node = rclpy.create_node("video_recoder_node")

    video_recoder_node = VideoRecoderNode(node, **kagrs)

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        video_recoder_node.shutdown()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
