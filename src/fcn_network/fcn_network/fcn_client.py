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
import array


class FCNClientNode(Node):
    def __init__(self):
        super().__init__("fcn_client_node")

        self.fcn_response: FCNRequest.Response = None
        self.fcn_occupied_response: FCNOccupiedRequest.Response = None
        self.is_finished = True

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

        self.create_timer(1.0, self.run)

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

            self.get_logger().info(
                f"Received response - target col: {response.target_col}"
            )
            self.get_logger().info(
                f"Received response - empty cols: {response.empty_cols.tolist()}"
            )

            self.send_fcn_occupied_request(response)

        except Exception as e:
            self.get_logger().warn(f"Service call failed - fcn_callback: {e}")
            self.is_finished = True

    def send_fcn_occupied_request(self, fcn_response: FCNRequest.Response):
        request = FCNOccupiedRequest.Request()

        if fcn_response is None:
            raise Exception("FCN response is None.")

        empty_cols = fcn_response.empty_cols.tolist()
        target_col = fcn_response.target_col

        request.empty_cols = empty_cols
        request.target_col = target_col

        future: Future = self.fcn_occupied_client.call_async(request)
        future.add_done_callback(self.fcn_occupied_callback)

    def fcn_occupied_callback(self, future: Future):
        print("fcn_occupied_callback")
        try:
            response: FCNOccupiedRequest.Response = future.result()

            action = "Sweaping" if response.action else "Grasping"
            self.get_logger().info(f"Received response - Action: {action}")
            self.get_logger().info(
                f"Received response - Moving row: {response.moving_row}"
            )
            self.get_logger().info(
                f"Received response - Moving cols: {response.moving_cols.tolist()}"
            )

        except Exception as e:
            self.get_logger().warn(f"Service call failed - fcn_occupied_callback: {e}")

        finally:
            self.is_finished = True

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
            # os.system("clear")
            self.get_logger().warn("Invalid class name.")
            self.is_finished = True
            return None

        # Valid user input. Automatically update self.fcn_response
        os.system("clear")
        self.send_fcn_request(target_cls)

    def run(self):
        if self.is_finished:
            self.is_finished = False
            self.handle_fcn_request()


def main():
    rclpy.init(args=None)

    node = FCNClientNode()

    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
