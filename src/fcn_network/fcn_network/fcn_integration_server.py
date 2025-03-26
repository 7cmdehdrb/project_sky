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
from custom_msgs.srv import FCNRequest, FCNOccupiedRequest, FCNIntegratedRequest

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

        self.fcn_client = self.create_client(FCNRequest, "/fcn_request")
        self.fcn_occupied_client = self.create_client(
            FCNOccupiedRequest, "/fcn_occupied_request"
        )
        self.col = None

        self.test_subscription = self.create_subscription(
            String,
            "/fcn_target_cls",
            self.test_callback,
            qos_profile=qos_profile_system_default,
        )
        self.test_publisher = self.create_publisher(
            String, "/result_fcn_target", qos_profile=qos_profile_system_default
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
        self.timer = self.create_timer(0.1, self.timer_callback)

    def timer_callback(self):
        if self.fcn_response is not None:
            self.get_logger().info(f"FCN response: {self.fcn_response.target_col}")

            if self.fcn_response.target_col != -1:
                self.send_fcn_occupied_request(self.fcn_response)
                self.col = self.fcn_response.target_col

            self.fcn_response = None

        if self.fcn_occupied_response is not None:
            target_col = self.col
            target_row = self.fcn_occupied_response.moving_row

            text = String(data=f"{target_col},{target_row}")
            self.test_publisher.publish(text)

            self.get_logger().info(f"Result: {target_col}, {target_row}")

            self.fcn_occupied_response = None
            self.col = None

    def test_callback(self, msg: String):
        target_cls = msg.data

        print(f"Request target_cls: {target_cls}")

        fcn_response: FCNRequest.Response = self.send_fcn_request(target_cls)

    def send_fcn_request(self, target_cls: str):
        request = FCNRequest.Request()
        request.target_cls = target_cls
        future = self.fcn_client.call_async(request)
        future.add_done_callback(self.fcn_response_callback)

    def fcn_response_callback(self, future: Future):
        try:
            response: FCNRequest.Response = future.result()
            self.fcn_response = response
        except Exception as e:
            self.get_logger().error(f"Service call failed: {e}")

    def send_fcn_occupied_request(self, fcn_response: FCNRequest.Response):
        request = FCNOccupiedRequest.Request()

        if fcn_response is None:
            raise Exception("FCN response is None.")

        empty_cols = fcn_response.empty_cols.tolist()
        target_col = fcn_response.target_col

        request.empty_cols = empty_cols
        request.target_col = target_col

        future = self.fcn_occupied_client.call_async(request)
        future.add_done_callback(self.fcn_occupied_response_callback)

    def fcn_occupied_response_callback(self, future: Future):
        try:
            response: FCNOccupiedRequest.Response = future.result()
            self.fcn_occupied_response = response
        except Exception as e:
            self.get_logger().error(f"Service call failed")


def main():
    rclpy.init(args=None)

    node = FCNClientNode()

    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
