#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from control_msgs.action import GripperCommand


class GripperActionClient(object):
    def __init__(self, node: Node):
        self._node = node

        # 액션 서버 이름은 launch와 환경에 맞게 수정
        self._action_client = ActionClient(
            self._node,
            GripperCommand,
            "/gripper/robotiq_gripper_controller/gripper_cmd",
        )

    def control_gripper(self, open: bool = True):
        position = 0.0 if open else 0.8
        return self.send_goal(position)

    def send_goal(self, position: float, max_effort: float = 0.0):
        goal_msg = GripperCommand.Goal()
        goal_msg.command.position = position
        goal_msg.command.max_effort = max_effort

        self._action_client.wait_for_server()
        response = self._action_client.send_goal(
            goal_msg, feedback_callback=self.feedback_callback
        )
        return response

    def feedback_callback(self, feedback_msg):
        # feedback은 optional, GripperCommand는 주로 result만 봄
        pass
