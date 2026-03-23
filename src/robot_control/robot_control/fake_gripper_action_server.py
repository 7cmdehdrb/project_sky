import time
import rclpy
from rclpy.action import ActionServer
from rclpy.node import Node
from control_msgs.action import GripperCommand


class DummyGripperServer(Node):
    def __init__(self):
        super().__init__("dummy_robotiq_server")
        self._action_server = ActionServer(
            self,
            GripperCommand,
            "/gripper/robotiq_gripper_controller/gripper_cmd",
            self.execute_callback,
        )
        self.get_logger().info(
            "🟢 가짜 Robotiq 그리퍼 Action Server가 시작되었습니다. 대기 중..."
        )

    def execute_callback(self, goal_handle):
        req: GripperCommand.Goal = goal_handle.request

        req_position = req.command.position
        req_effort = req.command.max_effort

        self.get_logger().info(
            f"📥 Goal 수신: 목표 위치={req_position}, 최대 토크={req_effort}"
        )

        # 1. 로봇의 물리적 구동 시간을 시뮬레이션 (2초 대기)
        self.get_logger().info("동작 수행 중 (2초 대기)...")
        time.sleep(2.0)

        # 2. Goal 수행 성공 처리
        goal_handle.succeed()

        # 3. Result 생성 및 반환
        result = GripperCommand.Result()
        result.position = req_position  # 요청받은 위치에 도달했다고 가정
        result.effort = 0.0  # 테스트용 가짜 데이터
        result.stalled = False
        result.reached_goal = True

        self.get_logger().info(
            f"✅ 동작 완료 및 Result 반환: 최종 위치={result.position}"
        )
        return result


def main(args=None):
    rclpy.init(args=args)
    dummy_server = DummyGripperServer()
    try:
        rclpy.spin(dummy_server)
    except KeyboardInterrupt:
        dummy_server.get_logger().info("서버 종료 요청됨.")
    finally:
        dummy_server.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
