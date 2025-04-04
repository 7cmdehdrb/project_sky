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
            if pixel[0] >= 640 or pixel[1] >= 480:  # Ensure bounds are correct
                self.get_logger().warn(f"ERROR HERE: {pixel}")
                return None

            image[pixel[1]][pixel[0]] = 1
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
            mask_image = self.mask_to_image(mask).astype(bool)  # 0, 1 (640, 480)
            if mask_image is None:
                self.get_logger().warn(f"MASK: {class_id}")
                continue
            if mask_image.shape != (480, 640):  # Ensure mask image shape matches
                print(
                    f"Mask image shape mismatch for class {class_id}: {mask_image.shape}"
                )
                continue
            mask_depth = depth_image[mask_image]  # (640, 480)
            mask_depth = mask_depth[mask_depth > 0]  # Remove zero values
            center_x = np.mean(mask[:, 0])
            center_y = np.mean(mask[:, 1])
            print(
                f"CLS: {class_id}, MEAN: {np.mean(mask_depth)}, STD: {np.std(mask_depth)}, MAX: {np.max(mask_depth)}, MIN: {np.min(mask_depth)}, 50%: {np.percentile(mask_depth, 50)}"
            )

            distance[class_id] = np.mean(mask_depth)
            center[class_id] = (center_x, center_y)

        # Grouping objects which are in the same column based on their center points x coordinate
        # If the difference between the x coordinates of two objects is less than threshold, they are considered in the same column
        grouped_objects = {
            0: [],
            1: [],
            2: [],
            3: [],
        }

        def get_empty_group():
            for key, value in grouped_objects.items():
                if len(value) == 0:
                    return key
            return None

        for class_id, center_point in center.items():
            class_id: str  # e.g. "cup_1"
            center_point: tuple  # e.g. (x, y) pixel

            if class_id not in distance:
                continue

            # Main Loop
            flag = False
            for key, value in grouped_objects.items():
                if len(value) == 0:
                    continue
                pixel_distance = abs(center[value[0]][0] - center_point[0])

                if pixel_distance < self._threshold:
                    value.append(class_id)
                    flag = True
                    break

            if not flag:
                empty_group_key = get_empty_group()
                if empty_group_key is not None:
                    grouped_objects[empty_group_key].append(class_id)
                else:
                    self.get_logger().warn(
                        f"All groups are full. Unable to classify object {class_id}."
                    )
                    continue

        temp = []
        for key, value in grouped_objects.items():
            if len(value) > 0:
                temp.append(value)
        grouped_objects = temp

        # grouped_objects = [[]]
        # for class_id, center_point in center.items():
        #     if class_id not in distance:
        #         continue
        #     if grouped_objects[0] == []:
        #         grouped_objects[0].append(class_id)
        #         continue
        #     for group in grouped_objects:
        #         for obj in group:
        #             if abs(center_point[0] - center[obj][0]) < self._threshold:
        #                 group.append(class_id)
        #                 break
        #         else:
        #             grouped_objects.append([class_id])

        print("Grouped Objects:", grouped_objects)

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
