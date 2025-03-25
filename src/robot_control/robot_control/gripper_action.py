#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from control_msgs.action import GripperCommand
from rclpy.task import Future


class GripperActionClient(object):
    def __init__(self, node: Node):
        self._node = node

        # 액션 서버 이름은 launch와 환경에 맞게 수정
        self._action_client = ActionClient(
            self._node,
            GripperCommand,
            "/gripper/robotiq_gripper_controller/gripper_cmd",
        )
        self._is_finished = False

    def control_gripper(self, open: bool = True):
        position = 0.0 if open else 0.8
        return self.send_goal(position)

    def send_goal(self, position: float, max_effort: float = 0.0):
        goal_msg = GripperCommand.Goal()
        goal_msg.command.position = position
        goal_msg.command.max_effort = max_effort

        self._action_client.wait_for_server()

        future: Future = self._action_client.send_goal(
            goal_msg, feedback_callback=self.feedback_callback
        )
        future.add_done_callback(self.goal_response_callback)

    def goal_response_callback(self, future: Future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self._is_finished = True
            return None

        future = goal_handle.get_result_async()
        future.add_done_callback(self.get_result_callback)

    def get_result_callback(self, future):
        self._is_finished = True

    def feedback_callback(self, feedback_msg):
        # feedback은 optional, GripperCommand는 주로 result만 봄
        self._node.get_logger().info(f"Feedback: {feedback_msg}")
        pass


def main(args=None):
    rclpy.init(args=args)
    node = rclpy.create_node("gripper_action_client")
    gripper_action_client = GripperActionClient(node)

    # close gripper
    # gripper_action_client.control_gripper(open=False)
    # print("Gripper closed")

    # open gripper
    gripper_action_client.control_gripper(open=True)
    print("Gripper opened")

    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == "__main__":
    main()
