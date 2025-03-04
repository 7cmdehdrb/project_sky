# ROS2
import rclpy
from rclpy.node import Node
from rclpy.time import Time
from rclpy.duration import Duration
from rclpy.task import Future
from rclpy.qos import QoSProfile, qos_profile_system_default

# Message
from std_msgs.msg import *
from geometry_msgs.msg import *
from sensor_msgs.msg import *
from nav_msgs.msg import *
from visualization_msgs.msg import *
from custom_msgs.srv import FCNRequest, FCNOccupiedRequest

# TF
from tf2_ros import *

# Python
import numpy as np
import sys
import os

from fcn_network.fcn_server import FCNClassNames, get_class_name


class FCNClientNode(Node):
    def __init__(self):
        super().__init__("fcn_client_node")

        self.fcn_response: FCNRequest.Response = None
        self.fcn_occupied_response: FCNOccupiedRequest.Response = None
        self.success = False

        self.fcn_client = self.create_client(FCNRequest, "fcn_request")
        self.fcn_occupied_client = self.create_client(
            FCNOccupiedRequest, "fcn_occupied_request"
        )

        while not self.fcn_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().warn(
                f"Service called fcn_request not available, waiting again..."
            )

        while not self.fcn_occupied_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().warn(
                f"Service called fcn_occupied_request not available, waiting again..."
            )

        self.get_logger().info("Service is available.")

        self.run()

    def send_fcn_request(self, target_cls: str):
        request = FCNRequest.Request()
        request.target_cls = target_cls
        future: Future = self.fcn_client.call_async(request)
        future.add_done_callback(self.fcn_callback)

    def fcn_callback(self, future: Future):
        try:
            response: FCNRequest.Response = future.result()

            if response.target_col == -1:
                raise Exception("Invalid target class.")

            self.fcn_response = response

            self.get_logger().info(
                f"Received response - empty cols: {self.fcn_response.empty_cols}"
            )
            self.get_logger().info(
                f"Received response - target col: {self.fcn_response.target_col}"
            )
        except Exception as e:
            self.get_logger().error(f"Service call failed: {e}")
            self.fcn_response = None

    def send_fcn_occupied_request(self, fcn_response: FCNRequest.Response):
        request = FCNOccupiedRequest.Request()

        if fcn_response is None:
            self.get_logger().warn("FCN response is None.")
            return

        request.target_col = fcn_response.target_col
        request.empty_cols = fcn_response.empty_cols

        future: Future = self.fcn_client.call_async(request)
        future.add_done_callback(self.fcn_occupied_callback)

    def fcn_occupied_callback(self, future: Future):
        try:
            response: FCNOccupiedRequest.Response = future.result()
            self.fcn_occupied_response = response

        except Exception as e:
            self.get_logger().error(f"Service call failed: {e}")
            self.fcn_occupied_response = None

    def handle_fcn_request(self):
        def announce():
            txt = "\n"
            for i, cls in enumerate(FCNClassNames):
                txt += f"{i + 1}. {get_class_name(cls)}\t"
                if i % 3 == 2:
                    txt += "\n"
            return txt

        self.get_logger().info(announce())
        self.get_logger().info("Enter the target class name.")

        target_cls = input(">>> ")

        # Check if the target class name is valid
        if not (target_cls in [cls.name.lower() for cls in FCNClassNames]):
            os.system("clear")
            self.get_logger().warn("Invalid class name.")
            return None

        # Valid user input. Automatically update self.fcn_response
        os.system("clear")
        self.send_fcn_request(target_cls)

        if self.fcn_response is None:
            self.get_logger().warn("FCN response is None.")
            os.system("clear")
            return None

        return self.fcn_response

    def handle_fcn_occupied_request(self, fcn_response: FCNRequest.Response):
        self.send_fcn_occupied_request(fcn_response)

        if self.fcn_occupied_response is None:
            return None

        return self.fcn_occupied_response

    def run(self):
        while True:
            fcn_response: FCNRequest.Response = self.handle_fcn_request()

            if fcn_response is None:
                self.get_logger().warn("[Error on run()] FCN response is None.")
                continue

            fcn_occupied_response: FCNOccupiedRequest.Response = (
                self.handle_fcn_occupied_request(fcn_response)
            )

            if fcn_occupied_response is None:
                self.get_logger().warn(
                    "[Error on run()] FCN occupied response is None."
                )
                continue

            self.get_logger().info(
                f"FCN occupied response - Action: {fcn_occupied_response.action}"
            )
            self.get_logger().info(
                f"FCN occupied response - Col: {fcn_occupied_response.moving_cols}"
            )


def main():
    rclpy.init(args=None)

    node = FCNClientNode()

    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
