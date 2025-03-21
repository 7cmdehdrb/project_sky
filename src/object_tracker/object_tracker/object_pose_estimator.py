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
from custom_msgs.msg import (
    BoundingBox,
    BoundingBoxMultiArray,
    BoundingBox3D,
    BoundingBox3DMultiArray,
)
from custom_msgs.srv import MegaposeRequest

# TF
from tf2_ros import *

# Megapose Server
import socket
import struct
import json
import io
import time
from cv_bridge import CvBridge
import cv2

# Python
import os
import sys
import numpy as np
import time

# Custom
from object_tracker.real_time_tracking_client import MegaPoseClient
from base_package.header import QuaternionAngle
from object_tracker.real_time_segmentation import RealTimeSegmentationNode


class MegaPoseEstimator(MegaPoseClient):
    def __init__(self, node: Node):
        self.SERVER_HOST = "127.0.0.1"
        self.SERVER_PORT = 5555
        self.SERVER_OPERATION_CODE_LENGTH = 4

        self.height, self.width = 480, 640

        self.node = node
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.connect((self.SERVER_HOST, self.SERVER_PORT))

        self.segmentation_subscriber = self.node.create_subscription(
            BoundingBoxMultiArray,
            "/real_time_segmentation_node/segmented_bbox",
            self.segmentation_callback,
            qos_profile=qos_profile_system_default,
        )
        self.camera_info_subscriber = self.node.create_subscription(
            CameraInfo,
            "/camera/camera1/color/camera_info",
            self.camera_info_callback,
            qos_profile=qos_profile_system_default,
        )

        self.segmentation_data = BoundingBoxMultiArray()

        # Parameters
        self.score_threshold = 0.7
        self.intrinsics_flag = False

        self.names = {
            # "cup_1": "sky",
            # "cup_2": "white",
            # "cup_3": "blue",
            # "mug_1": "black",
            # "mug_2": "gray",
            # "mug_3": "yello",
            "bottle_1": "alive",
            "bottle_2": "green_tea",
            "bottle_3": "yello_smoothie",
            "can_1": "coca_cola",
            "can_2": "cyder",
            "can_3": "yello_peach",
        }
        self.clss = {v: k for k, v in self.names.items()}

        self.detected_data = {
            # "labels": ["alive", "green_tea"],
            # "detections": [
            #     list(map(int, [352.0, 261.0, 404.0, 392.0])),
            #     list(map(int, [475.0, 234.0, 514.0, 359.0])),
            # ],
            # "labels": ["alive"],
            # "detections": [list(map(int, [352.0, 261.0, 404.0, 392.0]))],
            "labels": [],
            "detections": [],
            "use_depth": False,
        }
        self.enable_objects = []

        while True:
            response = self.send_list_objects_request(self.socket)
            if response is not None:
                self.enable_objects = response
                self.node.get_logger().info(f"Enable Objects: {self.enable_objects}")
                break

    def set_intrinsics(self, K: np.ndarray, image_size: tuple):
        is_success = self.send_intrinsics_request(
            self.socket,
            K=K,
            image_size=image_size,
        )

        if is_success:
            self.node.get_logger().info("Intrinsics set successfully.")
        else:
            self.node.get_logger().warn("Failed to set intrinsics.")

    def segmentation_callback(self, msg: BoundingBoxMultiArray):
        self.segmentation_data = msg

        data = {"detections": [], "labels": [], "use_depth": False}

        detections = []
        labels = []
        for bbox in msg.data:
            bbox: BoundingBox

            if bbox.conf < self.score_threshold:
                continue

            # if 'type_n' is in (...)
            if not (bbox.cls in self.names.keys()):
                continue

            # if "name" in (...)
            if not (self.names[bbox.cls] in self.enable_objects):
                continue

            detections.append(list(map(int, bbox.bbox.tolist())))
            labels.append(self.names[bbox.cls])

        data["detections"] = detections
        data["labels"] = labels

        self.detected_data = data

    def camera_info_callback(self, msg: CameraInfo):
        if self.intrinsics_flag:
            return None

        K = np.array(msg.k).reshape(3, 3)
        image_size = (msg.height, msg.width)

        if msg.height == self.height and msg.width == self.width:
            # Do nothing
            pass

        elif msg.height == 720 and msg.width == 1280:
            image_size = (self.height, self.width)

            h, w = 720, 1280
            crop_w, crop_h = 640, 480
            start_x = int((w - crop_w) // 2.05)
            start_y = int((h - crop_h) // 2.7)

            # offset = int((msg.width - self.width) // 2)

            # K[0, 0] = K[0, 0] * (self.width / msg.width)
            # K[1, 1] = K[1, 1] * (self.height / msg.height)
            # K[0, 2] = (K[0, 2] - offset) * (self.width / msg.width)
            # K[1, 2] = K[1, 2] * (self.height / msg.height)

            K = K.copy()
            K[0, 2] -= start_x
            K[1, 2] -= start_y

        else:
            # Prevent setting intrinsics
            self.node.get_logger().warn("Invalid image size. Cannot set intrinsics.")
            return None

        self.set_intrinsics(
            K=K,
            image_size=image_size,
        )

        print(image_size)
        self.intrinsics_flag = True


class ObjectPoseEstimator(Node):
    def __init__(self):
        super().__init__("object_pose_estimator")

        # >>> Instance Variables
        self.tracking_client = MegaPoseEstimator(node=self)
        self.bridge = CvBridge()
        # <<< Instance Variables

        # >>> Data
        self.image = None

        resource_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "..", "resource"
        )
        with open(os.path.join(resource_dir, "obj_bounds.json"), "r") as f:
            self.obj_bounds = json.load(f)

        # <<< Data

        # >>> Subscribers
        self.image_subscriber = self.create_subscription(
            Image,
            "/camera/camera1/color/image_raw",
            self.image_callback,
            qos_profile=qos_profile_system_default,
        )
        self.megapose_srv = self.create_service(
            MegaposeRequest,
            "/megapose_request",
            self.megapose_request_callback,
            qos_profile=qos_profile_system_default,
        )
        # <<< Subscribers

    # >>> Callbacks
    def image_callback(self, msg: Image):
        self.image = msg

    def megapose_request_callback(
        self, request: MegaposeRequest.Request, response: MegaposeRequest.Response
    ):
        self.get_logger().info(f"Get Megapose {len(request.request.data)} Requests")

        if self.image is None:
            self.get_logger().warn("No image received.")
            return response

        np_image = self.bridge.imgmsg_to_cv2(self.image, desired_encoding="bgr8")
        np_image = RealTimeSegmentationNode.crop_image(np_image)

        # TODO

        """
            self.detected_data = {
            # "labels": ["alive", "green_tea"],
            # "detections": [
            #     list(map(int, [352.0, 261.0, 404.0, 392.0])),
            #     list(map(int, [475.0, 234.0, 514.0, 359.0])),
            # ],
            # "labels": ["alive"],
            # "detections": [list(map(int, [352.0, 261.0, 404.0, 392.0]))],
            "labels": [],
            "detections": [],
            "use_depth": False,
        }
        """
        labels = self.tracking_client.detected_data["labels"]
        detections = self.tracking_client.detected_data["detections"]

        response_msg = BoundingBox3DMultiArray()
        for label, detection in zip(labels, detections):
            data = {
                "detections": [detection],
                "labels": [label],
                "use_depth": False,
            }
            results = self.tracking_client.send_pose_request(
                image=np_image, json_data=data
            )

            if results is None:
                self.get_logger().warn(f"Failed to get results for {label}")
                continue

            bbox_3d = self.post_process_result(results[0], label)
            if bbox_3d is not None:
                response_msg.data.append(bbox_3d)

        response.response = response_msg

        return response

    # <<< Callbacks

    def post_process_result(self, result: dict, label: str) -> BoundingBox3D:
        score = result["score"]
        cTo_matrix = np.array(result["cTo"]).reshape(4, 4)

        cls = self.tracking_client.clss[label]

        if score < self.tracking_client.score_threshold:
            return None

        offset_matrix = np.zeros((4, 4))
        offset_matrix[0, 3] = 0.06
        cTo_matrix += offset_matrix

        cTo_matrix_ros = QuaternionAngle.transform_realsense_to_ros(cTo_matrix)
        translation_matrix = cTo_matrix_ros[:3, 3] + np.array(
            [0, 0, self.obj_bounds[label]["z"] / 2.0]
        )

        rotation_matrix = cTo_matrix_ros[:3, :3]
        rpy_matrix = QuaternionAngle.euler_from_rotation_matrix(rotation_matrix)
        quaternion_matrix = QuaternionAngle.quaternion_from_euler(*rpy_matrix)

        bbox_3d = BoundingBox3D(
            # id=object_id,
            cls=cls,  # black
            conf=result["score"],
            pose=Pose(
                position=Point(**dict(zip(["x", "y", "z"], translation_matrix))),
                orientation=Quaternion(
                    **dict(zip(["x", "y", "z", "w"], quaternion_matrix))
                ),
            ),
            scale=Vector3(
                x=self.obj_bounds[label]["x"],
                y=self.obj_bounds[label]["z"],
                z=self.obj_bounds[label]["y"],
            ),
        )

        return bbox_3d


def main():
    rclpy.init(args=None)

    node = ObjectPoseEstimator()

    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
