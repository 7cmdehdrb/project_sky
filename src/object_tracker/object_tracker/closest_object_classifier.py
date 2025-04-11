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


class ClosestObjectClassifierNode(object):
    def __init__(self, node: Node, *args, **kwargs):
        self._node = node

        self._masks = dict()

        self._boundary = [170, 300, 460]

        self._threshold = kwargs.get("threshold", 50)
        self._debug = kwargs.get("debug", False)

        self._depth_raw = None
        self._image_manager = ImageManager(
            self._node,
            subscribed_topics=[
                {
                    "topic_name": "/camera/camera1/depth/image_rect_raw",
                    "callback": self.depth_callback,
                },
            ],
            published_topics=[],
            *args,
            **kwargs,
        )

        self._object_manager = ObjectManager(self._node, *args, **kwargs)

        self._node.create_subscription(
            BoundingBoxMultiArray,
            "/real_time_segmentation_node/segmented_bbox",
            self.bbox_callback,
            qos_profile=qos_profile_system_default,
        )

        self._node.create_timer(
            0.1,
            self.get_closest_object,
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

    def remove_outliers_iqr(self, depth_image: np.ndarray):
        try:
            # depth_image = np.log1p(depth_image)
            # q1 = np.percentile(depth_image, 25)
            # q3 = np.percentile(depth_image, 75)
            # iqr = q3 - q1

            # lower_bound = q1 - 1.5 * iqr
            # upper_bound = q3 + 1.5 * iqr

            # result = depth_image[
            #     (depth_image >= lower_bound) & (depth_image <= upper_bound)
            # ]
            # result = np.expm1(result)
            result = depth_image[depth_image < 1240]
            return result

        except IndexError as e:
            self._node.get_logger().warn("Fail to remove outliers")

        except Exception as e:
            self._node.get_logger().error(f"Error removing outliers: {e}")

        return depth_image

    def get_closest_object(self):
        if self._depth_raw is None:
            return None

        # Crop the depth image to the same size as the RGB image
        depth_image = self._image_manager.crop_image(self._depth_raw)

        # Translate the depth image for difference between RGB and depth
        zero_pixel = np.zeros((480, 40), dtype=np.uint16)

        depth_image = np.hstack([depth_image, zero_pixel])
        depth_image = depth_image[:, 40:]  # Ensure depth image is 640x480

        # Initialize the grouped objects
        grouped_objects = {
            0: [],
            1: [],
            2: [],
            3: [],
        }

        # Calculate the average distance and center point for each object
        for class_id, mask in self._masks.items():
            mask = mask.astype(bool)  # Convert mask to uint8
            mask_depth = depth_image[mask]  # (640, 480)
            mask_depth = mask_depth[mask_depth > 0]  # Remove zero values
            mask_depth = self.remove_outliers_iqr(mask_depth)  # Remove outliers

            mask_x = np.where(mask)[1]  # Get x coordinates of the mask
            center_x = np.mean(mask_x)  # Calculate the center x coordinate

            idx = (
                0
                if center_x < self._boundary[0]
                else (
                    1
                    if center_x < self._boundary[1]
                    else 2 if center_x < self._boundary[2] else 3
                )
            )
            grouped_objects[idx].append(
                {
                    "class_id": self._object_manager.indexs[class_id],
                    "distance": np.mean(mask_depth),
                }
            )

        # Initialize the result dictionary
        result = {
            0: {"class_id": -1, "distance": None},
            1: {"class_id": -1, "distance": None},
            2: {"class_id": -1, "distance": None},
            3: {"class_id": -1, "distance": None},
        }

        # Find the closest object in each group
        for key, value in grouped_objects.items():
            if len(value) > 0:
                value.sort(key=lambda x: x["distance"])
                result[key] = value[0]

            # if self._debug:
            #     for key, value in result.items():
            #         print(
            #             f"Group {key}: {self._object_manager.reverse_indexs[value['class_id']] if value['class_id'] != -1 else 'None'}",
            #             end=", ",
            #         )
            # print("")

        return result


def main(args=None):
    rclpy.init(args=args)

    from base_package.header import str2bool
    from rclpy.utilities import remove_ros_args

    # Remove ROS2 arguments
    argv = remove_ros_args(sys.argv)

    parser = argparse.ArgumentParser(description="Closest Object Classifier Node")

    parser.add_argument(
        "--threshold",
        type=int,
        required=False,
        default=50,
        help="Threshold for object classification",
    )
    parser.add_argument(
        "--debug",
        type=str2bool,
        required=False,
        default=False,
        help="Enable debug mode",
    )

    args = parser.parse_args(argv[1:])
    kagrs = vars(args)

    node = Node("closest_object_classifier_node")

    main_node = ClosestObjectClassifierNode(node, **kagrs)

    rclpy.spin(node=node)

    node.destroy_node()


if __name__ == "__main__":
    main()
