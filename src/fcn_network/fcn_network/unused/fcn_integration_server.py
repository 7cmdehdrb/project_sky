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
from base_package.manager import ObjectManager, Manager


class FCN_Integration_Client_Manager(Manager):
    def __init__(self, node: Node, *args, **kwargs):
        super().__init__(node, *args, **kwargs)

        self._publisher = self._node.create_publisher(
            String,
            "/fcn_target_cls",
            qos_profile=qos_profile_system_default,
        )

        self._subscriber = self._node.create_subscription(
            String,
            "/fcn_target_result",
            self.fcn_result_callback,
            qos_profile=qos_profile_system_default,
        )

        self._response = [None, None]

    def fcn_result_callback(self, msg: String):
        current_row, current_col = self._response

        if current_row is None and current_col is None:
            row, col = msg.data.split(",")
            if row is not None and col is not None:
                self._response = [row, col]  # A, 1
                self._node.get_logger().info(
                    f"FCN Integration Response: {self._response}"
                )

    def send_fcn_integration_request(self, target_cls: str):
        self._node.get_logger().info(f"Sending FCN Integration Request: {target_cls}")
        self._publisher.publish(String(data=target_cls))

    @property
    def fcn_result(self):
        row, col = self._response
        if row is not None and col is not None:
            self._response = [None, None]
            return [row, col]

        return [None, None]


class FCN_Integration_Server_Node(Node):
    def __init__(self, *arg, **kwargs):
        super().__init__("fcn_client_node")

        # >>> Manager >>>
        self._object_manager = ObjectManager(self, *arg, **kwargs)
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
        if self._target_cls is not None:
            return None

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
        self._fcn_occupied_response = future.result()

    def run(self):
        if self._target_cls is None:
            return None

        # >>> STEP 1: Send FCN Request
        if self._fcn_response is None:
            self.send_fcn_request(self._target_cls)
            self.get_logger().info(f"Send FCN Request: {self._target_cls}")
            return None

        # >>> STEP 2: Send FCN Occupied Request
        if self._fcn_response is not None and self._fcn_occupied_response is None:
            self.send_fcn_occupied_request(self._fcn_response)
            self.get_logger().info(f"Send FCN Occupied Request: {self._fcn_response}")
            return None

        # >>> STEP 3. Post-Process FCN Occupied Response and Publish Results
        if self._fcn_occupied_response is not None:
            if self._fcn_occupied_response.moving_row == "Z":
                self.get_logger().warn("No available row to move.")
                self.reset()
                return None

            # Target to move
            response_text = f"{self._fcn_occupied_response.moving_row},{self._fcn_response.target_col}"
            self.get_logger().info(f"Publish response: {response_text}")
            self._result_publisher.publish(String(data=response_text))

            # >>> STEP 4. Reset
            self.reset()

    def reset(self):
        self._target_cls = None
        self._fcn_response = None
        self._fcn_occupied_response = None


def main():
    rclpy.init(args=None)

    node = FCN_Integration_Server_Node()

    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
