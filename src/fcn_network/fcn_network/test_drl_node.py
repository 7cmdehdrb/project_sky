import rclpy
from rclpy.node import Node
from custom_msgs.srv import GetPolicyAction


class MockMainNode(Node):
    def __init__(self):
        super().__init__("mock_main_node")

        # Node A 클라이언트 생성
        self.client_a = self.create_client(GetPolicyAction, "get_policy_action")

        self.get_logger().info("Node A (Policy Server) 대기 중...")
        while not self.client_a.wait_for_service(timeout_sec=1.0):
            self.get_logger().info("Node A가 켜질 때까지 기다리는 중...")

        self.get_logger().info("🟢 Node A 확인 완료! 5초마다 제어 요청을 시작합니다.")

        self.request_count = 0
        # 5초 타이머 주기적 실행
        self.timer = self.create_timer(5.0, self.send_request)

    def send_request(self):
        self.request_count += 1

        # 임의의 Target ID 생성 (예: 1~3 순환)
        dummy_target_id = (self.request_count % 3) + 1

        req = GetPolicyAction.Request()
        req.target_id = dummy_target_id

        self.get_logger().info(
            f"▶️ [Main] {self.request_count}번째 요청 발송 (Target ID: {dummy_target_id})..."
        )

        # 비동기 호출
        future = self.client_a.call_async(req)
        future.add_done_callback(
            lambda fut, req_num=self.request_count: self.response_callback(fut, req_num)
        )

    def response_callback(self, future, req_num):
        try:
            result = future.result()
            self.get_logger().info(
                f"✅ [Main] {req_num}번째 응답 수신 성공! -> Action: {result.action_type}, Column: {result.target_column}"
            )
        except Exception as e:
            self.get_logger().error(f"❌ [Main] {req_num}번째 호출 실패: {e}")


def main(args=None):
    rclpy.init(args=args)
    node = MockMainNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
