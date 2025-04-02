# Python
import io
import json
import socket
import struct
import time

# OpenCV
import cv2

# NumPy
import numpy as np

# ROS2
import rclpy
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import QoSProfile, qos_profile_system_default
from rclpy.time import Time

# ROS2 Messages
from custom_msgs.msg import BoundingBox, BoundingBoxMultiArray
from geometry_msgs.msg import *
from nav_msgs.msg import *
from sensor_msgs.msg import *
from std_msgs.msg import *
from visualization_msgs.msg import *

# TF
from tf2_ros import *

# Custom Packages
from base_package.manager import ObjectManager, Manager


class SegmentationManager(Manager):
    def __init__(self, node: Node, *args, **kwargs):
        super().__init__(node, *args, **kwargs)

        self._object_manager = ObjectManager(node=self._node, *args, **kwargs)

        # >>> ROS2 >>>
        self._segmentation_subscriber = self._node.create_subscription(
            BoundingBoxMultiArray,
            "/real_time_segmentation_node/segmented_bbox",
            self.segmentation_callback,
            qos_profile=qos_profile_system_default,
        )
        # <<< ROS2 <<<

        # >>> Parameters >>>
        self._score_threshold = kwargs.get("score_threshold", 0.8)
        # <<< Parameters <<<

        # >>> Data >>>
        self._segmentation_data = []
        self._available_objects = kwargs.get("available_objects", [])
        # <<< Data <<<

    @property
    def segmentation_data(self):
        """
        return:
            [
                {
                    "label": [str],
                    "bbox": [[float, float, float, float]],
                    "conf": float
                }
            ]
        """
        return self._segmentation_data

    def segmentation_callback(self, msg: BoundingBoxMultiArray):
        detections = []

        for bbox in msg.data:
            bbox: BoundingBox

            if (
                (bbox.conf < self._score_threshold)
                or not (bbox.cls in self._object_manager.names.keys())
                or not (self._object_manager.names[bbox.cls] in self._available_objects)
            ):
                """
                Skip the bounding box if:
                    - The confidence is below the threshold
                    - The class is not in the object manager's names (e.g., cup_1)
                    - The object is not in the available objects list
                """
                pass

            else:
                detections.append(
                    {
                        "label": [bbox.cls],
                        "bbox": [bbox.bbox],
                        "conf": bbox.conf,
                    }
                )

        self._segmentation_data = detections
