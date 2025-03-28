# ROS2
import rclpy
from rclpy.node import Node
from rclpy.time import Time
from rclpy.duration import Duration
from rclpy.qos import QoSProfile, qos_profile_system_default

# ROS2 Messages
from std_msgs.msg import *
from geometry_msgs.msg import *
from sensor_msgs.msg import *
from nav_msgs.msg import *
from visualization_msgs.msg import *
from custom_msgs.srv import FCNRequest

# ROS2 TF
from tf2_ros import *

# Python Standard Libraries
import os
import sys
import json
from enum import Enum
import argparse

# Third-party Libraries
import numpy as np
import cv2
from matplotlib import pyplot as plt


# Custom Modules
from base_package.manager import ObjectManager, ImageManager
from fcn_network.fcn_manager import FCNModel, FCNManager, GridManager


class FCNServerNode(Node):
    def __init__(self, *arg, **kwargs):
        super().__init__("fcn_server_node")

        # >>> Managers >>>
        self._fcn_manager = FCNManager(self, *arg, **kwargs)
        self._grid_manager = GridManager(self, *arg, **kwargs)
        image_subscriptions = [
            {
                "topic_name": "/camera/camera1/color/image_raw",
                "callback": self.fcn_image_callback,
            },
        ]
        image_publications = [
            # {"topic_name": self.get_name() + "/processed_image"},
            # {"topic_name": self.get_name() + "/plot_image"},
        ]
        self._image_manager = ImageManager(
            self,
            published_topics=image_publications,
            subscribed_topics=image_subscriptions,
            *arg,
            **kwargs,
        )
        self._object_manager = ObjectManager()
        # <<< Managers <<<

        # >>> ROS2 >>>
        self._srv = self.create_service(
            FCNRequest, "/fcn_request", self.fcn_request_callback
        )
        # <<< ROS2 <<<

        # >>> Data >>>
        self._fcn_image: Image = None
        self._target_output = None
        self._post_processed_data = None
        # <<< Data <<<

        # >>> Main
        self.get_logger().info("FCN Server Node has been initialized.")
        self.create_timer(1.0, self.publish_processed_image)
        # <<< Main

    def fcn_image_callback(self, msg: Image):
        self._fcn_image = msg

    def fcn_request_callback(
        self, request: FCNRequest.Request, response: FCNRequest.Response
    ):
        """
        Args:
            request (FCNRequest.Request):
                str: target_cls
            response (FCNRequest.Response):
                int: target_col
                int[]: empty_cols
        Exceptions:
            response (FCNRequest.Response):
                int: target_col = -1
                int[]: empty_cols = []
        """
        # Exception handling
        if self._fcn_image is None:
            self.get_logger().warn("No image received.")
            return response

        self.get_logger().info(f"Request Received: {request.target_cls}")

        # Get the class name from the target class

        # Crop the image
        np_image = self._image_manager.decode_message(
            self._fcn_image, desired_encoding="rgb8"
        )
        np_image = self._image_manager.crop_image(np_image)

        # Predict the image
        outputs = self._fcn_manager.predict(np_image)

        # Get the target output
        target_output = outputs[self._object_manager.indexs[request.target_cls]]

        # TODO: Add the weights
        weights = [1.0] * self._grid_manager.get_colums_length()
        target_col, empty_cols, _ = self._fcn_manager.post_process_results(
            target_output, weights
        )

        # Set the response
        response.target_col = target_col
        response.empty_cols = empty_cols

        self.get_logger().info(
            f"Return response: target_col={target_col}, empty_cols={empty_cols}"
        )

        return response

    def publish_processed_image(
        self, image_output: np.ndarray, processed_data: np.ndarray, top_peak_idx: list
    ):
        """
        Publish the processed image and plot.
        """

        # >>> STEP 1. Publish FCN Processed Image >>>
        target_output_normalized = cv2.normalize(
            image_output, None, 0, 255, cv2.NORM_MINMAX
        ).astype(np.uint8)
        msg = self._image_manager.encode_message(
            target_output_normalized, encoding="mono8"
        )
        self._image_manager.publish(self.get_name() + "/processed_image", msg)

        # >>> STEP 2. Publish Plot Image >>>
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
        self._image_manager.publish(self.get_name() + "/plot_image", plot_image_msg)

        plt.close(fig)


def main():
    rclpy.init(args=None)

    parser = argparse.ArgumentParser(description="FCN Server Node")
    parser.add_argument(
        "--model_file",
        type=str,
        required=True,
        default="best_model.pth",
        help="Path or file name of the trained FCN model. If input is a file name, the file should be located in the 'resource' directory. Required",
    )
    parser.add_argument(
        "--grid_data_file",
        type=str,
        required=True,
        default="grid_data.json",
        help="Path or file name of the grid data. If input is a file name, the file should be located in the 'resource' directory. Required",
    )
    parser.add_argument(
        "--fcn_image_transform",
        type=bool,
        required=False,
        default=True,
        help="Whether to apply image transformation (default: True)",
    )
    parser.add_argument(
        "--fcn_gain",
        type=float,
        required=False,
        default=2.0,
        help="Gain value for post-processing (default: 2.0)",
    )

    args = parser.parse_args()
    kagrs = vars(args)

    node = FCNServerNode(**kagrs)

    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
