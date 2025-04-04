# Python
import argparse
import json
import os
import sys
from PIL import Image as PILImage
from PIL import ImageEnhance

# OpenCV
import cv2
from cv_bridge import CvBridge

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

# YOLO
from ultralytics import YOLO
from ultralytics.engine.results import Boxes, Masks, Results

# Custom Packages
from ament_index_python.packages import get_package_share_directory
from base_package.manager import ImageManager, Manager, ObjectManager


class ClosestObjectClassifierNode(Node):
    def __init__(self, *args, **kwargs):
        super().__init__("closest_object_classifier_node")
        self._image_raw = None
        self._depth_raw = None
        self._masks = dict()

        self._threshold = kwargs["threshold"]

        self._image_manager = ImageManager(
            self,
            subscribed_topics=[
                {
                    "topic_name": "/camera/camera1/color/image_raw",
                    "callback": self.image_callback,
                },
                {
                    "topic_name": "/camera/camera1/depth/image_rect_raw",
                    "callback": self.depth_callback,
                },
            ],
            published_topics=[],
            *args,
            **kwargs,
        )

        self._object_manager = ObjectManager(self, *args, **kwargs)

        self.create_subscription(
            BoundingBoxMultiArray,
            "/real_time_segmentation_node/segmented_bbox",
            self.bbox_callback,
            qos_profile=qos_profile_system_default,
        )

    def image_callback(self, msg: Image):
        self._image_raw = self._image_manager.decode_message(
            msg, desired_encoding="rgb8"
        )

    def depth_callback(self, msg: Image):
        self._depth_raw = self._image_manager.decode_message(
            msg, desired_encoding="16UC1"
        )

    def bbox_callback(self, msg: BoundingBoxMultiArray):
        masks = dict()
        for bbox in msg.data:
            bbox: BoundingBox
            class_id = bbox.cls
            mask = np.reshape(np.array(bbox.mask_data), (bbox.mask_row, bbox.mask_col))
            masks[class_id] = mask
        self._masks = masks
        self.get_closest_object()

    def mask_to_image(self, mask: np.ndarray):
        image = np.zeros((480, 640), dtype=np.uint8)  # Adjusted to height x width
        for pixel in mask:
            if pixel[0] >= 480 or pixel[1] >= 640:  # Ensure bounds are correct
                return None
            image[pixel[0]][pixel[1]] = 1
        return image

    def masks_image(self, masks: dict):
        image = np.full((480, 640), "", dtype=object)  # Adjusted to height x width
        for class_id, mask in masks.items():
            mask_image = self.mask_to_image(mask)
            if mask_image is None:
                continue
            image[mask_image > 0] = class_id
        return image

    def get_closest_object(self):
        if self._depth_raw is None:
            return None

        # Crop the depth image to the same size as the RGB image
        depth_image = self._image_manager.crop_image(self._depth_raw)
        if depth_image.shape != (480, 640):  # Ensure depth image shape matches
            print(f"Depth image shape mismatch: {depth_image.shape}")
            return None

        distance = dict()
        center = dict()

        # Calculate the average distance and center point for each object
        for class_id, mask in self._masks.items():
            mask_image = self.mask_to_image(mask)
            if mask_image is None:
                continue
            if mask_image.shape != (480, 640):  # Ensure mask image shape matches
                print(
                    f"Mask image shape mismatch for class {class_id}: {mask_image.shape}"
                )
                continue
            mask_depth = depth_image[mask_image > 0]
            center_x = np.mean(mask[:, 0])
            center_y = np.mean(mask[:, 1])
            distance[class_id] = np.mean(mask_depth)
            center[class_id] = (center_x, center_y)

        # Grouping objects which are in the same column based on their center points x coordinate
        # If the difference between the x coordinates of two objects is less than threshold, they are considered in the same column
        grouped_objects = [[]]
        for class_id, center_point in center.items():
            if class_id not in distance:
                continue
            if grouped_objects[0] == []:
                grouped_objects[0].append(class_id)
                continue
            for group in grouped_objects:
                for obj in group:
                    if abs(center_point[0] - center[obj][0]) < self._threshold:
                        group.append(class_id)
                        break
            else:
                grouped_objects.append([class_id])

        try:
            # Find the closest object in each group
            closest_object = []
            for group in grouped_objects:
                min_distance = min(
                    [distance[obj] for obj in group if obj in distance]
                )  # Find the minimum distance in the group
                min_class_id = [obj for obj in group if distance[obj] == min_distance]
                closest_object.append(min_class_id)
                print("Closest Object:", closest_object)

        except Exception as e:
            print(f"Error finding closest object: {e}")
            return None

        return closest_object


def main(args=None):
    rclpy.init(args=args)

    parser = argparse.ArgumentParser(description="Closest Object Classifier Node")
    parser.add_argument(
        "--threshold",
        type=int,
        required=False,
        default=50,
        help="Threshold for object classification",
    )

    args = parser.parse_args()
    kagrs = vars(args)

    node = ClosestObjectClassifierNode(**kagrs)

    rclpy.spin(node=node)

    node.destroy_node()


if __name__ == "__main__":
    main()
