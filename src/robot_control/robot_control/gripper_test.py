# ROS2
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.action.client import ClientGoalHandle
from rclpy.time import Time
from rclpy.duration import Duration
from rclpy.qos import QoSProfile, qos_profile_system_default

# Message
from std_msgs.msg import *
from geometry_msgs.msg import *
from sensor_msgs.msg import *
from nav_msgs.msg import *
from visualization_msgs.msg import *
from moveit_msgs.msg import *
from trajectory_msgs.msg import *
from shape_msgs.msg import *
from control_msgs.action import GripperCommand
from builtin_interfaces.msg import Duration as BuiltinDuration

# TF
from tf2_ros import *

# Python
import sys
import os
import numpy as np
from enum import Enum
import time
import threading
import copy
from rotutils import *
from dataclasses import dataclass

# Custom
from moveit2_commander import (
    FK_ServiceManager,
    IK_ServiceManager,
    CartesianPath_ServiceManager,
    KinematicPath_ServiceManager,
    GetPlanningScene_ServiceManager,
    ApplyPlanningScene_ServiceManager,
    ExecuteTrajectory_ServiceManager,
)

from robot_control.controller import RobotiqController

class GripperTestNode(Node):
    def __init__(self):
        super().__init__("gripper_test_node")
        self.controller = RobotiqController(self)

        # 테스트: 그리퍼 열기/닫기
        self.get_logger().info("그리퍼 테스트 시작: 3초 후에 그리퍼가 닫힙니다.")
        threading.Timer(3.0, self.test_gripper).start()

    def test_gripper(self):
        # 그리퍼 닫기
        self.get_logger().info("그리퍼 닫는 중...")
        self.controller.control_gripper(open=False)
        time.sleep(2)  # 2초 대기

        # 그리퍼 열기
        self.get_logger().info("그리퍼 여는 중...")
        self.controller.control_gripper(open=True)
        time.sleep(2)  # 2초 대기

        self.get_logger().info("그리퍼 테스트 완료.")

def main(args=None):
    rclpy.init(args=args)
    node = GripperTestNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()