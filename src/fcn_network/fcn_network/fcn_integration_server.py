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
import array
from base_package.manager import ObjectManager


class FCNClientNode(Node):
    def __init__(self):
        super().__init__("fcn_client_node")

        # >>> Manager >>>
        self._object_manager = ObjectManager()
        # <<< Manager <<<

        # >>> Service Responses
        self._fcn_response: FCNRequest.Response = None
        self._fcn_occupied_response: FCNOccupiedRequest.Response = None
        # <<< Service Responses

        # >>> Service Clients
        self._fcn_client = self.create_client(FCNRequest, "/fcn_request")
        self._fcn_occupied_client = self.create_client(
            FCNOccupiedRequest, "/fcn_occupied_request"
        )
        # <<< Service Clients

        # >> ROS >>>
        self._trigger_subscription = self.create_subscription(
            String,
            "/fcn_target_cls",
            self.trigger_callback,
            qos_profile=qos_profile_system_default,
        )
        self._result_publisher = self.create_publisher(
            String, "/fcn_target_result", qos_profile=qos_profile_system_default
        )
        self._target_cls: str = None
        # <<< ROS <<<

        while not self._fcn_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().warn(
                f"Service called fcn_request not available, waiting again..."
            )

        while not self._fcn_occupied_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().warn(
                f"Service called fcn_occupied_request not available, waiting again..."
            )

        self.get_logger().info("Service is available.")

        self.create_timer(1.0, self.run)

    def trigger_callback(self, msg: String):
        if msg.data in self._object_manager.names.keys():
            self._target_cls = msg.data
            self.get_logger().info(f"Received class name: {msg.data}")
        else:
            self.get_logger().warn(f"Invalid class name: {msg.data}")

    def send_fcn_request(self, target_cls: str):
        request = FCNRequest.Request()
        request.target_cls = target_cls
        future: Future = self._fcn_client.call_async(request)
        future.add_done_callback(self.fcn_response_callback)

    def send_fcn_occupied_request(self, fcn_response: FCNRequest.Response):
        request = FCNOccupiedRequest.Request()

        if fcn_response is None:
            self.get_logger().warn("FCN response is None.")
            return None

        empty_cols = fcn_response.empty_cols.tolist()
        target_col = fcn_response.target_col

        request.empty_cols = empty_cols
        request.target_col = target_col

        future: Future = self._fcn_occupied_client.call_async(request)
        future.add_done_callback(self.fcn_occupied_response_callback)

    def fcn_response_callback(self, future: Future):
        self._fcn_response = future.result()

    def fcn_occupied_response_callback(self, future: Future):
        self.fcn_occupied_response = future.result()

    def run(self):
        if self._target_cls is None:
            return None

        # >>> STEP 1: Send FCN Request
        if self._fcn_response is None:
            self._fcn_response = self.send_fcn_request(self._target_cls)

        # >>> STEP 2: Send FCN Occupied Request
        if self._fcn_response is not None and self._fcn_occupied_response is None:
            self._fcn_occupied_response = self.send_fcn_occupied_request(
                self._fcn_response
            )
            self.get_logger().info(
                f"FCN Occupied Response: {self._fcn_occupied_response}"
            )

        # >>> STEP 3. Post-Process FCN Occupied Response and Publish Results
        if self._fcn_occupied_response is not None:
            response_text = f"{self._fcn_occupied_response.moving_row},{self._fcn_response.target_col}"
            self.get_logger().info(f"Response: {response_text}")
            self._result_publisher.publish(String(data=response_text))

            # >>> STEP 4. Reset
            self._target_cls = None
            self._fcn_response = None
            self._fcn_occupied_response = None


def main():
    rclpy.init(args=None)

    node = FCNClientNode()

    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
