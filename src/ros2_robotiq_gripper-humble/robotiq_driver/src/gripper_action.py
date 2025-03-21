#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from control_msgs.action import GripperCommand


class GripperActionClient(Node):
    def __init__(self):
        super().__init__("gripper_action_client")

        # 액션 서버 이름은 launch와 환경에 맞게 수정
        self._action_client = ActionClient(
            self, GripperCommand, "/gripper/robotiq_gripper_controller/gripper_cmd"
        )

    def send_goal(self, position, max_effort=0.0):
        goal_msg = GripperCommand.Goal()
        goal_msg.command.position = position
        goal_msg.command.max_effort = max_effort

        self._action_client.wait_for_server()
        self.get_logger().info(
            f"Sending goal: position={position}, max_effort={max_effort}"
        )
        self._send_goal_future = self._action_client.send_goal_async(
            goal_msg, feedback_callback=self.feedback_callback
        )
        self._send_goal_future.add_done_callback(self.goal_response_callback)

    def goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().info("Goal rejected 😢")
            return

        self.get_logger().info("Goal accepted! ✅")
        self._get_result_future = goal_handle.get_result_async()
        self._get_result_future.add_done_callback(self.get_result_callback)

    def get_result_callback(self, future):
        result = future.result().result
        self.get_logger().info(
            f"Result received! Position: {result.position}, Effort: {result.effort}"
        )
        rclpy.shutdown()

    def feedback_callback(self, feedback_msg):
        # feedback은 optional, GripperCommand는 주로 result만 봄
        pass


def main(args=None):
    rclpy.init(args=args)
    gripper_client = GripperActionClient()

    # 여기서 열고 닫는 값 지정 (0.0 = open, 0.8 = close)
    gripper_client.send_goal(position=0.0, max_effort=10.0)

    rclpy.spin(gripper_client)


if __name__ == "__main__":
    main()
