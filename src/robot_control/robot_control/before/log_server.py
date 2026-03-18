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

        self._attempt = kwargs.get("exp_attempt", -1)

        # >>> Managers >>>

        # >>> Subscriptions >>>
        self._plot_image: Image = None
        self._processed_image: Image = None

        self._root_dir = f"/home/irol/workspace/project_sky/src/robot_control/resource/exp_result/{self._attempt}"
        self._image_dir = os.path.join(self._root_dir, "images")

        if not os.path.exists(self._root_dir):
            os.makedirs(self._root_dir)

        elif self._attempt == -1:
            if os.path.exists(self._root_dir):
                self.get_logger().warn(
                    f"Attempt number set to test mode, data will overwrite the existing data in {self._root_dir} and original data will be lost."
                )
                for filename in os.listdir(self._root_dir):
                    if filename == "images":
                        for img in os.listdir(os.path.join(self._root_dir, filename)):
                            os.remove(os.path.join(self._root_dir, filename, img))
                    else:
                        os.remove(os.path.join(self._root_dir, filename))

        else:
            raise RuntimeError(
                f"Experiment attempt {self._attempt} already exists. Please use a different attempt number."
            )

        if not os.path.exists(self._image_dir):
            os.makedirs(self._image_dir)

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
        fcn_data = request.fcn_data
        action = request.action
        column = request.column
        step = request.step

        try:
            pdm_2d_np = self._image_manager.decode_message(
                self._processed_image, desired_encoding="rgb8"
            )
            pdm_1d_np = self._image_manager.decode_message(
                self._plot_image, desired_encoding="rgb8"
            )

            pdm_2d_filename = f"pdm_2d_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            cv2.imwrite(os.path.join(self._image_dir, pdm_2d_filename), pdm_2d_np)
            self.get_logger().info(f"Saved 2D PDM image: {pdm_2d_filename}")

            pdm_1d_filename = f"pdm_1d_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            cv2.imwrite(os.path.join(self._image_dir, pdm_1d_filename), pdm_1d_np)
            self.get_logger().info(f"Saved 1D PDM image: {pdm_1d_filename}")

        except Exception as e:
            self.get_logger().error(f"Error: {e}")
            pdm_2d_filename = "None"
            pdm_1d_filename = "None"

        finally:
            step_data = {
                "2d_pdm": pdm_2d_filename,
                "1d_pdm": pdm_1d_filename,
                "fcn_data": fcn_data.tolist(),
                "action": action,
                "column": column,
                "step": step,
            }

            self._data.append(step_data)

            self._plot_image = None
            self._processed_image = None

            response.success = True
            return response

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

    import argparse
    from rclpy.utilities import remove_ros_args
    from base_package.header import str2bool

    # Remove ROS2 arguments
    argv = remove_ros_args(sys.argv)

    parser = argparse.ArgumentParser(description="FCN Server Node")

    parser.add_argument(
        "--exp_attempt",
        type=int,
        required=True,
    )

    args = parser.parse_args(argv[1:])
    kagrs = vars(args)

    node = LogServerNode(**kagrs)

    try:
        rclpy.spin(node)

        node.destroy_node()
        rclpy.shutdown()

    except KeyboardInterrupt:
        node.get_logger().info("Keyboard interrupt, shutting down...")
        node.export_data()
    finally:
        node.get_logger().info("Node destroyed and shutdown complete.")
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":

    main()
