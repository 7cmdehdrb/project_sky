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
import cv2

# Custom
from base_package.image_manager import ImageManager


class IntegrationImageNode(Node):
    def __init__(self):
        super().__init__("integration_image_node")

        self._raw_image: Image = None
        self._closest_image: Image = None
        self._segmentation_image: Image = None
        self._1d_fcn_processed_image: Image = None
        self._2d_fcn_processed_image: Image = None

        response = __import__("urllib.request", fromlist=["urlopen"]).urlopen(
            "https://images.velog.io/images/717lumos/post/8715706d-f855-4211-8bba-3ddeede780c4/%EA%B7%B8%EB%A6%BC1.png"
        )
        img_array = np.asarray(bytearray(response.read()), dtype=np.uint8)
        self._ros_image: np.ndarray = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        self._image_manager = ImageManager(
            node=self,
            subscribed_topics=[
                {
                    "topic_name": "/camera/camera1/color/image_raw",
                    "callback": self._callback_raw_image,
                },
                {
                    "topic_name": "/closest_object_classifier/closest_object_overlay",
                    "callback": self._callback_closest_image,
                },
                {
                    "topic_name": "/real_time_segmentation_node/segmented_image",
                    "callback": self._callback_segmentation_image,
                },
                {
                    "topic_name": "/fcn_service_node/pdm_visualization",
                    "callback": self._callback_1d_fcn_processed_image,
                },
                {
                    "topic_name": "/fcn_service_node/target_map_visualization",
                    "callback": self._callback_2d_fcn_processed_image,
                },
            ],
            published_topics=[
                {
                    "topic_name": f"{self.get_name()}/integrated_image",
                },
            ],
        )

        HZ = 10.0
        self._timer = self.create_timer(1.0 / HZ, self._publish_integrated_image)

    def _callback_raw_image(self, msg: Image):
        self._raw_image = msg

    def _callback_closest_image(self, msg: Image):
        self._closest_image = msg

    def _callback_segmentation_image(self, msg: Image):
        self._segmentation_image = msg

    def _callback_1d_fcn_processed_image(self, msg: Image):
        self._1d_fcn_processed_image = msg

    def _callback_2d_fcn_processed_image(self, msg: Image):
        self._2d_fcn_processed_image = msg

    def _post_process_png(self, msg: np.ndarray):
        if msg is None:
            return np.zeros((480, 640, 3), dtype=np.uint8)

        h, w = msg.shape[:2]
        target_ratio = 640 / 480

        current_ratio = w / h

        if current_ratio > target_ratio:
            # wider than target: add padding to top and bottom
            new_h = int(w / target_ratio)
            pad_top = (new_h - h) // 2
            pad_bottom = new_h - h - pad_top
            msg = cv2.copyMakeBorder(
                msg,
                pad_top,
                pad_bottom,
                0,
                0,
                cv2.BORDER_CONSTANT,
                value=(255, 255, 255),
            )
        elif current_ratio < target_ratio:
            # taller than target: add padding to left and right
            new_w = int(h * target_ratio)
            pad_left = (new_w - w) // 2
            pad_right = new_w - w - pad_left
            msg = cv2.copyMakeBorder(
                msg,
                0,
                0,
                pad_left,
                pad_right,
                cv2.BORDER_CONSTANT,
                value=(255, 255, 255),
            )

        return cv2.resize(msg, (640, 480))

    def _post_process_images(self, msg: Image):
        if msg is None:
            return np.zeros((480, 640, 3), dtype=np.uint8)

        np_image = self._image_manager.decode_message(
            image_msg=msg, desired_encoding="bgr8"
        )
        if np_image.shape[0] != 480 or np_image.shape[1] != 640:
            np_image = cv2.resize(np_image, (640, 480))

        return np_image

    def _get_integrated_image(self) -> Image:
        # Integrate the images here

        np_raw_image = self._post_process_images(self._raw_image)

        np_closest_image = self._post_process_images(self._closest_image)

        np_segmentation_image = self._post_process_images(self._segmentation_image)

        np_1d_fcn_processed_image = self._post_process_images(
            self._1d_fcn_processed_image
        )

        np_2d_fcn_processed_image = self._post_process_images(
            self._2d_fcn_processed_image
        )

        np_ros_image = self._post_process_png(self._ros_image)

        top_integrated_image = np.hstack(
            [
                np_raw_image,
                np_segmentation_image,
                np_closest_image,
            ]
        )

        bottom_integrated_image = np.hstack(
            [
                np_1d_fcn_processed_image,
                np_2d_fcn_processed_image,
                np_ros_image,
            ]
        )

        np_integrated_image = np.vstack(
            [
                top_integrated_image,
                bottom_integrated_image,
            ]
        )

        integrated_image_msg = self._image_manager.encode_message(
            image=np_integrated_image, encoding="bgr8"
        )

        return integrated_image_msg

    def _publish_integrated_image(self):
        integrated_image_msg: Image = self._get_integrated_image()

        publisher = self._image_manager.get_publisher(
            f"{self.get_name()}/integrated_image"
        )

        if publisher is not None:
            publisher.publish(integrated_image_msg)

        else:
            self.get_logger().error("통합 이미지 퍼블리셔를 찾을 수 없습니다.")


def main(args=None):
    rclpy.init(args=args)

    node = IntegrationImageNode()

    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
