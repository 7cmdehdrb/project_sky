# Python
import os
import sys
import json
import numpy as np
import argparse
import array
import onnx
import onnxruntime
import torch
import cv2
from matplotlib import pyplot as plt

# ROS2
import rclpy
from rclpy.node import Node
from rclpy.time import Time
from rclpy.duration import Duration
from rclpy.qos import QoSProfile, qos_profile_system_default
import time

# Message
from std_msgs.msg import *
from geometry_msgs.msg import *
from sensor_msgs.msg import *
from nav_msgs.msg import *
from visualization_msgs.msg import *
from custom_msgs.msg import *
from custom_msgs.srv import (
    FCNOccupiedRequest,
    FCNRequest,
    MegaposeRequest,
    FCNIntegratedRequest,
)

# TF
from tf2_ros import *

# Custom
from base_package.header import PointCloudTransformer, QuaternionAngle
from base_package.manager import ObjectManager, ImageManager
from robot_control.control_manager import ControlAction
from fcn_network.fcn_manager import GridManager, FCNManager
from object_tracker.closest_object_classifier import ClosestObjectClassifierNode


class DirectFCNServer(object):
    def __init__(self, node: Node, *args, **kwargs):
        self._node = node

        # >>> Manager >>>
        self._image_manager = ImageManager(
            self._node,
            subscribed_topics=[
                {
                    "topic_name": "/camera/camera1/color/image_raw",
                    "callback": self._image_callback,
                }
            ],
            published_topics=[
                {
                    "topic_name": "/fcn_server/processed_image",
                },
                {
                    "topic_name": "/fcn_server/plot_image",
                },
            ],
            *args,
            **kwargs,
        )

        self._closest_object_manager = ClosestObjectClassifierNode(
            self._node, *args, **kwargs
        )

        self._fcn_manager = FCNManager(self._node, *args, **kwargs)
        providers = (
            ["CUDAExecutionProvider"]
            if onnxruntime.get_device() == "GPU"
            else ["CPUExecutionProvider"]
        )

        self.model_ort = onnxruntime.InferenceSession(
            "/home/irol/workspace/project_sky/src/fcn_network/resource/exported/policy.onnx",
            providers=providers,
        )

        # >>> Parameters >>>
        self._gain = kwargs.get("fcn_gain", 2.0)
        self._gamma = kwargs.get("fcn_gamma", 0.7)

        # >>> Data >>>
        self._image: Image = None
        self._target_objects: BoundingBox3DMultiArray = None
        self._fcn_result: np.ndarray = np.empty(640)
        self._action_policy: np.ndarray = np.array([0], dtype=np.int16)
        self._action_column: np.ndarray = np.array([0], dtype=np.int16)  # np.zeros(1)
        self._action = np.zeros((2, 1), dtype=np.float32)
        # <<< Data <<<

        # >>> Returns >>>
        self._fcn_result_data = np.array([0.0, 0.0, 0.0, 0.0])

        self._megapose_client = self._node.create_client(
            MegaposeRequest,
            "/megapose_request",
            qos_profile=qos_profile_system_default,
        )

        while not self._megapose_client.wait_for_service(timeout_sec=1.0):
            self._node.get_logger().info(
                "/megapose_request service not available, waiting again..."
            )

    @property
    def fcn_result_data(self):
        return self._fcn_result_data

    def reset(self):
        self._image: Image = None
        self._target_objects: BoundingBox3DMultiArray = None
        self._fcn_result: np.ndarray = np.empty(640)
        self._action_policy: np.ndarray = np.array([0], dtype=np.int16)
        self._action_column: np.ndarray = np.array([0], dtype=np.int16)  # np.zeros(1)
        self._action = np.zeros((2, 1), dtype=np.float32)

    def run(self, target_id: int = 0):
        self._target_objects = None
        observation: np.ndarray = self.get_observation(target_id=target_id)
        observation = np.expand_dims(observation, axis=0)

        if observation[0] is not None:
            try:
                if self._target_objects is None:
                    raise ValueError("Target objects is None")

                self._action = self.model_ort.run(["actions"], {"obs": observation})[0]

                self._action_policy = np.array(self._action[:, 0], dtype=np.int32)
                self._action_column = np.array(self._action[:, 1], dtype=np.int32)

                self._action_policy = np.clip(self._action_policy, 0, 2, dtype=np.int32)
                self._action_column = np.clip(self._action_column, 0, 3)

                # Define action
                action = self._action_policy[0]

                # Define target column
                target_col = self._action_column[0]

                self._node.get_logger().info(f"Target Column: {target_col}")

                target_row: str = "Z"

                for object in self._target_objects.data:
                    object: BoundingBox3D
                    if int(object.cls[1]) == target_col:
                        row = object.cls[0]
                        if ord(row) < ord(target_row):
                            target_row = row

                if target_row == "Z":
                    self.reset()
                    raise ValueError("Target row is None")

                target_id = f"{target_row}{self._action_column[0]}"

                # Define moving column
                moving_col = target_col + 1 if action == 1 else target_col - 1
                moving_row = target_row

                moving_id = f"{moving_row}{moving_col}"

                # 0 g 1 sr 2 sl
                response = ControlAction(
                    action=action != 0,
                    target_id=target_id,
                    goal_ids=[moving_id] if action != 0 else [],
                    target_object=None,
                )

                return response

            except ValueError as ve:
                self._node.get_logger().warn(f"ValueError in FCN Server: {ve}")

            except Exception as ex:
                self._node.get_logger().error(f"Error in FCN Server: {ex}")

        return None

    # >>> Callback Functions >>>
    def _image_callback(self, msg: Image):
        self._image = msg

    def _post_process_fcn_result(self, fcn_result: np.ndarray):
        col1 = np.max(fcn_result[:185])
        col2 = np.max(fcn_result[185:320])
        col3 = np.max(fcn_result[320:455])
        col4 = np.max(fcn_result[455:640])

        result = np.array([col1, col2, col3, col4], dtype=np.float32)

        self._fcn_result_data = result

        return result

    def _send_megapose_request(self) -> BoundingBox3DMultiArray:
        request = MegaposeRequest.Request()
        response: MegaposeRequest.Response = self._megapose_client.call(request)

        return response.response

    # <<< Callback Functions <<<
    def _get_closest_object(self) -> np.ndarray:
        closest_objects: dict = self._closest_object_manager.get_closest_object()

        result = [-1] * 4

        for key, value in closest_objects.items():
            if key != -1:
                result[key] = value["class_id"]

        return np.array(result, dtype=np.int32)

    def _get_fcn_result(self, np_image: np.ndarray, target_id: int) -> np.ndarray:
        fcn_raw_result: np.ndarray = self._fcn_manager.predict(np_image=np_image)
        fcn_target_raw_result: np.ndarray = fcn_raw_result[target_id]

        for _ in range(30):
            self.publish_output_image(image_output=fcn_target_raw_result)

        normalized_results = fcn_target_raw_result * np.exp(
            -self._gain * (1 - fcn_target_raw_result)
        )  # 지수 함수로 가중치 적용

        fcn_1d_result = np.sum(normalized_results, axis=0)

        if self._fcn_result is None:
            self._fcn_result = fcn_1d_result
        else:
            self._fcn_result = (
                fcn_1d_result * self._gamma + (1 - self._gamma) * self._fcn_result
            )

        for _ in range(30):
            self.publish_result_image(processed_data=fcn_1d_result, top_peak_idx=[])

        column_distribution = self._post_process_fcn_result(self._fcn_result)
        return column_distribution

    def _get_front_object_distance(self) -> np.ndarray:
        bbox_3d: BoundingBox3DMultiArray = self._send_megapose_request()
        self._target_objects = bbox_3d

        cols = {
            0: None,
            1: None,
            2: None,
            3: None,
        }

        distances = {
            "A": 1.5,
            "B": 2.5,
            "C": 3.5,
        }

        # In bounding box 3d, all objects has cls, which annotate the row/col of the object
        for bbox in bbox_3d.data:
            bbox: BoundingBox3D

            row = bbox.cls[0]
            col = int(bbox.cls[1])

            # In column, If the object is not detected, the value is None
            if cols[col] is None:
                cols[col] = bbox

            else:
                # If the object is detected, compare the distance
                if ord(row) < ord(cols[col].cls[0]):
                    cols[col] = bbox

        result = [0.0] * 4
        for key, value in cols.items():
            # key: int. the column number
            # value: BoundingBox3D. the object in the column
            if value is None:
                continue

            row = value.cls[0]
            distance = distances[row]
            result[key] = distance

        return result

    def get_observation(self, target_id: int = 0) -> np.ndarray:
        if self._image is None:
            self._node.get_logger().warn("Image is None")
            return None
            # raise ValueError("Image is None")

        np_image: np.ndarray = self._image_manager.decode_message(
            self._image, desired_encoding="rgb8"
        )
        np_image: np.ndarray = self._image_manager.crop_image(np_image)

        # >>> STEP 1. Get FCN peak data >>>
        column_distribution = self._get_fcn_result(
            np_image=np_image, target_id=target_id
        )
        # print(f"Column Distribution: {column_distribution}")

        # >>> STEP 2. Get Front Object (int id) >>>
        front_object = self._get_closest_object()
        # print(f"Front Object: {front_object}")

        # >>> STEP 3. Get Front Objects' Distance >>>
        front_object_distance = self._get_front_object_distance()
        # print(f"Front Object Distance: {front_object_distance}")

        # >>> STEP 4. Get Target ID >>>
        target_id = target_id
        # print(f"Target ID: {target_id}")

        result = np.concatenate(
            [
                column_distribution,
                front_object_distance,
                front_object,
                np.array([target_id], dtype=np.float32),
                np.array(self._action_policy, dtype=np.float32),
                np.array(self._action_column, dtype=np.float32),
            ],
            dtype=np.float32,
        )

        # print(f"Result: {result}")

        return result

    def publish_output_image(self, image_output: np.ndarray):
        """
        Publish the processed image.
        """
        target_output_normalized = cv2.normalize(
            image_output, None, 0, 255, cv2.NORM_MINMAX
        ).astype(np.uint8)
        msg = self._image_manager.encode_message(
            target_output_normalized, encoding="mono8"
        )

        self._image_manager.publish("/fcn_server/processed_image", msg)

    def publish_result_image(self, processed_data: np.ndarray, top_peak_idx: List[int]):
        """
        Publish the plot image.
        """
        fig = plt.figure(figsize=(16, 9))
        plt.plot(processed_data)

        for peak_idx in top_peak_idx:
            plt.axvline(x=peak_idx, color="r", linestyle="--", linewidth=5)

        plt.xlabel("Pixel")
        plt.ylabel("Intensity")
        plt.title("Post-processed Data")

        # Convert the plot to a ROS2 Image message
        fig.canvas.draw()
        plot_image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        plot_image = plot_image.reshape(fig.canvas.get_width_height()[::-1] + (3,))

        plot_image_msg = self._image_manager.encode_message(plot_image, encoding="rgb8")
        self._image_manager.publish("/fcn_server/plot_image", plot_image_msg)

        plt.close(fig)


def main():
    rclpy.init(args=None)

    from rclpy.utilities import remove_ros_args
    from base_package.header import str2bool

    # Remove ROS2 arguments
    argv = remove_ros_args(sys.argv)

    parser = argparse.ArgumentParser(description="FCN Server Node")

    # threshold, debug, model_file, fcn_image_transform, fcn_gain, fcn_gamma

    # >>> Closest Object Classifier >>>
    parser.add_argument(
        "--debug",
        type=str2bool,
        required=False,
        default=False,
    )

    parser.add_argument(
        "--threshold",
        type=int,
        required=False,
        default=50,
    )

    # >>> FCN Manager >>>
    parser.add_argument(
        "--model_file",
        type=str,
        required=True,
        help="Path or file name of the trained FCN model. If input is a file name, the file should be located in the 'resource' directory. Required",
    )
    parser.add_argument(
        "--fcn_image_transform", type=bool, required=False, default=True
    )
    parser.add_argument(
        "--fcn_gain",
        type=float,
        required=False,
        default=2.0,
    )
    parser.add_argument(
        "--fcn_gamma",
        type=float,
        required=False,
        default=0.7,
    )

    args = parser.parse_args(argv[1:])
    kagrs = vars(args)

    # Create the node
    node = Node("direct_fcn_server")

    # Create the DirectFCNServer instance
    server = DirectFCNServer(node, **kagrs)

    # Spin in a separate thread
    thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    thread.start()

    hz = 1.0
    rate = node.create_rate(hz)

    try:
        while rclpy.ok():
            server.run()
            rate.sleep()

    except KeyboardInterrupt:
        pass

    # Destroy the node
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
