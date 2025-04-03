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
        super().__init__("closest_object_classifier_node", *args, **kwargs)
        self._image_raw = None
        self._depth_raw = None
        self._masks = dict()

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

        # Calculate the average distance for each object
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
            distance[class_id] = np.mean(mask_depth)

        # Create a list of objects in each column
        mask_image = self.masks_image(self._masks)
        if mask_image.shape != (480, 640):  # Ensure masks image shape matches
            print(f"Masks image shape mismatch: {mask_image.shape}")
            return None

        column_objects_info = []
        for i in range(640):
            objects = []
            for j in range(480):
                if (
                    not mask_image[j][i] in objects
                ):  # Adjusted indexing to height x width
                    objects.append(mask_image[j][i])
            column_objects_info.append(objects)

        # Find the closest object
        entire_objects = list(self._masks.keys())
        closest_object = None
        for column in column_objects_info:
            dist_column = [distance[i] for i in column if i in distance]
            if not dist_column:
                continue
            min_distance = min(dist_column)
            min_index = dist_column.index(min_distance)
            occluded_object = column.pop(min_index)
            if occluded_object in entire_objects:
                entire_objects.remove(occluded_object)
        closest_object = entire_objects

        print("Closest Object:", closest_object)

        return closest_object


def main(args=None):
    rclpy.init(args=args)

    parser = argparse.ArgumentParser(description="Closest Object Classifier Node")

    args = parser.parse_args()
    kagrs = vars(args)

    node = ClosestObjectClassifierNode(**kagrs)

    rclpy.spin(node=node)

    node.destroy_node()


if __name__ == "__main__":
    main()
