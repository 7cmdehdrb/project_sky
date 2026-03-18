import rclpy
from rclpy.node import Node
from example_interfaces.srv import AddTwoInts


class MainNode(Node):
    def __init__(self):
        super().__init__("main_node")

        self.client_a = self.create_client(AddTwoInts, "service_a")
        self.client_b = self.create_client(AddTwoInts, "service_b")

        # 1. A, B 모두 살아있는지 이중 체크
        self.get_logger().info("노드 A와 B가 모두 켜질 때까지 대기합니다...")
        while not self.client_b.wait_for_service(timeout_sec=1.0):
            self.get_logger().info("노드 B 대기 중...")
        while not self.client_a.wait_for_service(timeout_sec=1.0):
            self.get_logger().info("노드 A 대기 중...")

        self.get_logger().info(
            "🟢 노드 A, B 모두 확인 완료! 주기적 요청(5초)을 시작합니다."
        )

        self.request_count = 0
        # 2. 5초마다 실행되는 타이머 생성
        self.timer = self.create_timer(5.0, self.timer_callback)

    def timer_callback(self):
        self.request_count += 1
        req_id = self.request_count

        req = AddTwoInts.Request()
        req.a = req_id  # 식별하기 쉽게 a에 카운트를 넣음
        req.b = 10

        self.get_logger().info(f"▶️ [Main] {req_id}번째 요청 발송...")

        # 3. 비동기 호출 (타이머를 멈추지 않음)
        future = self.client_a.call_async(req)

        # 응답이 오면 future_callback 함수가 실행되도록 연결
        future.add_done_callback(lambda fut, id=req_id: self.future_callback(fut, id))

    def future_callback(self, future, req_id):
        try:
            result = future.result()
            self.get_logger().info(
                f"✅ [Main] {req_id}번째 최종 결과 수신 성공: {result.sum}"
            )
        except Exception as e:
            self.get_logger().error(f"❌ [Main] {req_id}번째 호출 실패: {e}")


def main(args=None):
    rclpy.init(args=args)
    node = MainNode()
    # 메인은 타이머와 콜백만 처리하므로 기본 SingleThreadedExecutor로 충분합니다.
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
