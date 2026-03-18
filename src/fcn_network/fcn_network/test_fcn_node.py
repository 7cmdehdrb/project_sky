import rclpy
from rclpy.node import Node
from rclpy.task import Future
from custom_msgs.srv import GetFCNResult


class MockClientNode(Node):
    def __init__(self):
        super().__init__("mock_client_node")
        self.client = self.create_client(GetFCNResult, "get_fcn_prediction")

        self.get_logger().info("서비스 서버 대기 중...")
        while not self.client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info("FCN Service 대기 중...")

        self.get_logger().info("🟢 서비스 서버 확인 완료! 3초 주기로 요청 발송 시작.")
        self.timer = self.create_timer(3.0, self.send_request)
        self.req_cnt = 0

    def send_request(self):
        self.req_cnt += 1
        self.get_logger().info(f"▶️ [{self.req_cnt}]번째 추론 요청 전송...")

        req = GetFCNResult.Request()
        req.weight = [
            1.0,
            1.0,
            1.0,
            1.0,
        ]  # 예시 가중치 (실제로는 의미 있는 값으로 변경 필요)
        req.target_class_idx = 5  # 예시 타겟 클래스 인덱스

        # 비동기 요청
        future = self.client.call_async(req)
        future.add_done_callback(
            lambda fut, idx=self.req_cnt: self.response_callback(fut, idx)
        )

    def response_callback(self, future: Future, req_idx: int):
        try:
            result: GetFCNResult.Response = future.result()
            self.get_logger().info(f"✅ [{req_idx}]번째 응답 수신: {result.data}")
        except Exception as e:
            self.get_logger().error(f"❌ [{req_idx}]번째 요청 실패: {e}")


def main(args=None):
    rclpy.init(args=args)
    node = MockClientNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
