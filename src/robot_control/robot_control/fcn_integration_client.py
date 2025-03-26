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


class FCN_IntegratedClient(object):
    def __init__(self, node: Node):
        self._node = node

        self.cls = None

        self._cls_list = [
            "cup_1",
            "cup_2",
            "cup_3",
            "mug_1",
            "mug_2",
            "mug_3",
            "bottle_1",
            "bottle_2",
            "bottle_3",
            "can_1",
            "can_2",
            "can_3",
        ]

        self.cls_subscriber = self._node.create_subscription(
            String,
            self._node.get_name() + "/fcn_target_cls",
            self.cls_callback,
            qos_profile=qos_profile_system_default,
        )
        self._client = self._node.create_client(
            FCNIntegratedRequest, "/fcn_integrated_request"
        )

    def cls_callback(self, msg: String):
        if msg.data in self._cls_list:
            self.cls = msg.data

    def send_fcn_integrated_request(self):
        if self.cls is None:
            return None

        request = FCNIntegratedRequest.Request()
        request.target_cls = self.cls

        response: FCNIntegratedRequest.Response = self._client.call(request)
        return response
