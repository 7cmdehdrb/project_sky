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
import tqdm
from scipy.spatial.transform import Rotation as R

# Custom
from object_tracker.real_time_tracking_client import MegaPoseClient
from base_package.header import QuaternionAngle
from object_tracker.real_time_segmentation import RealTimeSegmentationNode


class Queue(object):
    def __init__(self, max_size=10):
        self.max_size = max_size
        self.queue = []

    def push(self, item):
        if len(self.queue) >= self.max_size:
            self.queue.pop(0)
        self.queue.append(item)

    def get(self):
        return self.queue

    def get_average(self):
        if len(self.queue) != self.max_size:
            return 0.0

        return np.mean(self.queue, axis=0)


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
            "cup_1": "cup_sky",
            "cup_2": "cup_white",
            "cup_3": "cup_blue",
            "mug_1": "mug_black",
            "mug_2": "mug_gray",
            "mug_3": "mug_yello",
            "bottle_1": "alive",
            "bottle_2": "green_tea",
            "bottle_3": "yello_smoothie",
            "can_1": "coca_cola",
            "can_2": "cyder",
            "can_3": "yello_peach",
        }
        self.clss = {v: k for k, v in self.names.items()}

        self.detected_data = {
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

        self.node.get_logger().info(f"\nOriginal K: {K}")

        image_size = (msg.height, msg.width)

        if msg.height == self.height and msg.width == self.width:
            # Do nothing
            pass

        elif msg.height == 720 and msg.width == 1280:
            image_size = (self.height, self.width)

            K = K.copy()

            # offset = int((msg.width - self.width) // 2)

            # K[0, 0] = K[0, 0] * (self.width / msg.width)
            # K[1, 1] = K[1, 1] * (self.height / msg.height)
            # K[0, 2] = (K[0, 2] - offset) * (self.width / msg.width)
            # K[1, 2] = K[1, 2] * (self.height / msg.height)

            h, w = 720, 1280
            crop_w, crop_h = 640, 480
            start_x = int((w - crop_w) // 2.05)
            start_y = int((h - crop_h) // 2.7)

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

        self.node.get_logger().info("Set intrinsics successfully.")
        self.node.get_logger().info(f"\nK: {K}")
        self.node.get_logger().info(f"\nImage Size: {msg.height}, {msg.width}")

        self.intrinsics_flag = True


class ObjectPoseEstimator(Node):
    def __init__(self):
        super().__init__("object_pose_estimator")

        # >>> Instance Variables
        self.tracking_client = MegaPoseEstimator(node=self)
        self.bridge = CvBridge()
        # <<< Instance Variables

        # >>> Data
        self.image: Image = None
        self.depth_image: Image = None

        resource_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "..", "resource"
        )
        with open(os.path.join(resource_dir, "obj_bounds.json"), "r") as f:
            self.obj_bounds = json.load(f)

        # <<< Data

        # >>> Subscribers

        self.depth_image_subscriber = self.create_subscription(
            Image,
            "/camera/camera1/depth/image_rect_raw",
            self.depth_image_callback,
            qos_profile=qos_profile_system_default,
        )
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

        self.test = self.create_publisher(
            Image,
            "/test",
            qos_profile=qos_profile_system_default,
        )

    # >>> Callbacks
    def image_callback(self, msg: Image):
        self.image = msg

    def depth_image_callback(self, msg: Image):
        self.depth_image = msg

    def megapose_request_callback(
        self, request: MegaposeRequest.Request, response: MegaposeRequest.Response
    ):
        self.get_logger().info(f"Get Megapose {len(request.request.data)} Requests")

        if self.image is None:
            self.get_logger().warn("No image received.")
            return response

        # TODO
        labels = self.tracking_client.detected_data["labels"]
        detections = self.tracking_client.detected_data["detections"]

        response_msg = BoundingBox3DMultiArray()
        for label, detection in zip(labels, detections):

            with tqdm.tqdm(total=10) as pbar:
                for attempt in range(10):
                    np_image = self.bridge.imgmsg_to_cv2(
                        self.image, desired_encoding="bgr8"
                    )
                    np_image = RealTimeSegmentationNode.crop_image(np_image)

                    np_depth_image = self.bridge.imgmsg_to_cv2(
                        self.depth_image, desired_encoding="passthrough"
                    )
                    np_depth_image = RealTimeSegmentationNode.crop_image(np_depth_image)

                    offset = np.random.randint(0, 5)
                    detections = [
                        [
                            int(
                                np.clip(
                                    detection[0] - (offset * 2),
                                    0,
                                    640,
                                )
                            ),
                            int(
                                np.clip(
                                    detection[1] - (offset * 2),
                                    0,
                                    480,
                                )
                            ),
                            int(
                                np.clip(
                                    detection[2] + (offset * 2),
                                    0,
                                    640,
                                )
                            ),
                            int(
                                np.clip(
                                    detection[3] + (offset * 2),
                                    0,
                                    480,
                                )
                            ),
                        ]
                    ]

                    zero_depth_image = np.zeros_like(np_depth_image)
                    zero_depth_image[
                        detection[0] : detection[2], detection[1] : detection[3]
                    ] = np_depth_image[
                        detection[0] : detection[2], detection[1] : detection[3]
                    ]

                    data = {
                        "detections": detections,
                        "labels": [label],
                        "use_depth": True,
                        "refiner_iterations": 5,
                        "depth_scale_to_m": 0.001,
                    }

                    results = self.tracking_client.send_pose_request_rgbd(
                        image=np_image, depth=zero_depth_image, json_data=data
                    )

                    cTo = results[0]["cTo"]
                    bbox = results[0]["boundingBox"]
                    score = results[0]["score"]
                    temp_average = 0.0

                    rotation_matrix = np.array(cTo).reshape(4, 4)[:3, :3]
                    rot = R.from_matrix(rotation_matrix)

                    z_world = np.array([0, 0, 1])
                    y_axis = rot.apply([0, 1, 0])

                    cosine_angle = np.clip(np.dot(y_axis, z_world), -1.0, 1.0)
                    angle_deg = np.abs(np.degrees(np.arccos(cosine_angle)) - 90.0)

                    self.get_logger().info(
                        f"{label}({attempt}): {score:.3f}/{angle_deg:.2f}"
                    )

                    pbar.update(1)
                    pbar.set_description(
                        f"{label}({attempt}): {score:.3f}/{angle_deg:.2f}"
                    )

                    flag = False
                    if score > 0.999 and angle_deg < 5.0:
                        result_img = np_image.copy()
                        cv2.rectangle(
                            result_img,
                            (int(bbox[0]), int(bbox[1])),
                            (int(bbox[2]), int(bbox[3])),
                            (0, 255, 0),
                            2,
                        )
                        cv2.rectangle(
                            result_img,
                            (int(detections[0][0]), int(detections[0][1])),
                            (int(detections[0][2]), int(detections[0][3])),
                            (255, 0, 0),
                            2,
                        )

                        self.test.publish(self.bridge.cv2_to_imgmsg(result_img, "bgr8"))
                        flag = True

                    elif score > 0.95:
                        temp_queue = Queue(max_size=10)

                        with tqdm.tqdm(total=100) as pbar2:
                            for _ in range(100):
                                np_image = self.bridge.imgmsg_to_cv2(
                                    self.image, desired_encoding="bgr8"
                                )
                                np_image = RealTimeSegmentationNode.crop_image(np_image)

                                np_depth_image = self.bridge.imgmsg_to_cv2(
                                    self.depth_image, desired_encoding="passthrough"
                                )
                                np_depth_image = RealTimeSegmentationNode.crop_image(
                                    np_depth_image
                                )

                                zero_depth_image = np.zeros_like(np_depth_image)
                                zero_depth_image[
                                    detection[0] + 3 : detection[2] - 3,
                                    detection[1] + 3 : detection[3] - 3,
                                ] = np_depth_image[
                                    detection[0] + 3 : detection[2] - 3,
                                    detection[1] + 3 : detection[3] - 3,
                                ]

                                temp_data = {
                                    "detections": detections,
                                    "initial_cTos": [cTo],
                                    "labels": [label],
                                    "refiner_iterations": 5,
                                    "use_depth": True,
                                    "depth_scale_to_m": 0.001,
                                }

                                temp_results = (
                                    self.tracking_client.send_pose_request_rgbd(
                                        image=np_image,
                                        depth=zero_depth_image,
                                        json_data=temp_data,
                                    )
                                )

                                temp_score = temp_results[0]["score"]
                                bbox = temp_results[0]["boundingBox"]
                                cTo = temp_results[0]["cTo"]

                                rotation_matrix = np.array(cTo).reshape(4, 4)[:3, :3]
                                rot = R.from_matrix(rotation_matrix)

                                z_world = np.array([0, 0, 1])
                                y_axis = rot.apply([0, 1, 0])

                                cosine_angle = np.clip(
                                    np.dot(y_axis, z_world), -1.0, 1.0
                                )
                                angle_deg = np.abs(
                                    np.degrees(np.arccos(cosine_angle)) - 90.0
                                )

                                temp_queue.push(temp_score)
                                temp_average = temp_queue.get_average()

                                self.get_logger().info(
                                    f"{label}({attempt}): {temp_average:.3f}/{angle_deg:.2f}"
                                )

                                pbar2.update(1)
                                pbar2.set_description(
                                    f"{label}({attempt})-SEG: {temp_average:.3f}/{angle_deg:.2f}"
                                )

                                result_img = np_image.copy()
                                cv2.rectangle(
                                    result_img,
                                    (int(bbox[0]), int(bbox[1])),
                                    (int(bbox[2]), int(bbox[3])),
                                    (0, 255, 0),
                                    2,
                                )
                                cv2.rectangle(
                                    result_img,
                                    (int(detections[0][0]), int(detections[0][1])),
                                    (int(detections[0][2]), int(detections[0][3])),
                                    (255, 0, 0),
                                    2,
                                )

                                self.test.publish(
                                    self.bridge.cv2_to_imgmsg(result_img, "bgr8")
                                )

                                if temp_average > 0.98 and angle_deg < 5.0:
                                    flag = True
                                    results = temp_results
                                    break

                    if flag:
                        bbox_3d = self.post_process_result(results[0], label)
                        response_msg.data.append(bbox_3d)

                        pbar.close()
                        break

        self.get_logger().info(f"Response: {len(response_msg.data)}")

        response.response = response_msg

        return response

    # <<< Callbacks

    def post_process_result(self, result: dict, label: str) -> BoundingBox3D:
        score = result["score"]
        cTo_matrix = np.array(result["cTo"]).reshape(4, 4)
        cls = self.tracking_client.clss[label]

        offset_matrix = np.zeros((4, 4))
        offset_matrix[0, 3] = 0.06  # TODO: Change this value
        cTo_matrix += offset_matrix

        cTo_matrix_ros = QuaternionAngle.transform_realsense_to_ros(cTo_matrix)
        translation_matrix = cTo_matrix_ros[:3, 3] + np.array(
            [0, 0, self.obj_bounds[label]["y"] / 2.0]
        )

        rotation_matrix = cTo_matrix_ros[:3, :3]
        rpy_matrix = QuaternionAngle.euler_from_rotation_matrix(rotation_matrix)
        quaternion_matrix = QuaternionAngle.quaternion_from_euler(*rpy_matrix)

        bbox_3d = BoundingBox3D(
            # id=object_id,
            cls=cls,  # black
            conf=score,
            pose=Pose(
                position=Point(**dict(zip(["x", "y", "z"], translation_matrix))),
                orientation=Quaternion(
                    **dict(zip(["x", "y", "z", "w"], quaternion_matrix))
                ),
            ),
            scale=Vector3(
                x=np.clip(self.obj_bounds[label]["x"], 0.0, 0.05),
                y=np.clip(self.obj_bounds[label]["y"], 0.0, 0.2),
                z=np.clip(self.obj_bounds[label]["z"], 0.0, 0.05),
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
