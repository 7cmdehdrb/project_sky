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
from std_srvs.srv import Empty
from custom_msgs.srv import LogRequest

# TF
from tf2_ros import *

# Python
import os
import sys
import numpy as np
import pandas as pd
import cv2

# Custom Modules
from base_package.manager import ObjectManager, ImageManager, Manager
from datetime import datetime


class LogManager(Manager):
    def __init__(self, node: Node, *args, **kwargs):
        super().__init__(node, *args, **kwargs)

        self._fcn_result = [0.0, 0.0, 0.0, 0.0]

        self._fcn_result_sub = self._node.create_subscription(
            Float64MultiArray,
            "/fcn_server/fcn_result",
            self.fcn_result_callback,
            qos_profile=qos_profile_system_default,
        )

        self._client = self._node.create_client(
            LogRequest,
            "/log_server/log",
            qos_profile=qos_profile_system_default,
        )

        while not self._client.wait_for_service(timeout_sec=1.0):
            self._node.get_logger().info(
                "Log server not available, waiting for it to be available..."
            )

    def fcn_result_callback(self, msg: Float64MultiArray):
        """
        Callback function for the FCN result.
        """
        self._fcn_result = msg.data

    def log(self, fcn_data: List[float], action: int, column: int, step: int):
        """
        Send a request to the log server.
        """
        request = LogRequest.Request()
        request.fcn_data = fcn_data if fcn_data is not None else self._fcn_result
        request.action = action
        request.column = column
        request.step = step

        self._node.get_logger().info(
            f"Sending request to log server: {request.fcn_data}, {request.action}, {request.column}, {request.step}"
        )

        response: LogRequest.Response = self._client.call(request)
        if response is not None:
            return response.success

        return False


class LogServerNode(Node):
    def __init__(self, *arg, **kwargs):
        super().__init__("log_server_node")

        # >>> Managers >>>

        # >>> Subscriptions >>>
        self._plot_image: Image = None
        self._processed_image: Image = None

        self._root_dir = (
            "/home/min/7cmdehdrb/ros2_ws/src/robot_control/resource/exp_result"
        )
        self._image_dir = os.path.join(self._root_dir, "images")

        if not os.path.exists(self._image_dir):
            os.makedirs(self._image_dir)

        if not os.path.exists(self._root_dir):
            os.makedirs(self._root_dir)

        image_subscriptions = [
            {
                "topic_name": "/fcn_server/processed_image",
                "callback": self.fcn_processed_image_callback,
            },
            {
                "topic_name": "/fcn_server/plot_image",
                "callback": self.fcn_plot_image_callback,
            },
        ]
        self._image_manager = ImageManager(
            self,
            subscribed_topics=image_subscriptions,
            published_topics=[],
            *arg,
            **kwargs,
        )

        # >>> SRV >>>
        self._srv = self.create_service(
            LogRequest,
            "/log_server/log",
            self.log_callback,
            qos_profile=qos_profile_system_default,
        )

        self._data = []

    # >>> Callbacks >>>
    def log_callback(self, request: LogRequest.Request, response: LogRequest.Response):
        if self._processed_image is None or self._plot_image is None:
            self.get_logger().warn(
                "Log callback called, but images are not available yet."
            )

        fcn_data = request.fcn_data
        action = request.action
        column = request.column
        step = request.step

        pdm_2d_np = self._image_manager.decode_message(self._processed_image)
        pdm_1d_np = self._image_manager.decode_message(self._plot_image)

        pdm_2d_filename = f"pdm_2d_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        cv2.imwrite(os.path.join(self._image_dir, pdm_2d_filename), pdm_2d_np)

        pdm_1d_filename = f"pdm_1d_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        cv2.imwrite(os.path.join(self._image_dir, pdm_1d_filename), pdm_1d_np)

        step_data = {
            "2d_pdm": pdm_2d_filename,
            "1d_pdm": pdm_1d_filename,
            "fcn_data": fcn_data,
            "action": action,
            "column": column,
            "step": step,
        }

        self._data.append(step_data)

    def fcn_processed_image_callback(self, msg):
        self._processed_image = msg

    def fcn_plot_image_callback(self, msg):
        self._plot_image = msg

    def export_data(self):
        """
        Export the data to a CSV file.
        """
        df = pd.DataFrame(self._data)
        timestamp = datetime.now().strftime("%m-%d-%H-%M-%S")
        df.to_csv(os.path.join(self._root_dir, f"{timestamp}.csv"), index=False)


def main():
    rclpy.init(args=None)

    node = LogServerNode()

    rclpy.spin(node)

    node.export_data()

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
