import rclpy
from rclpy.node import Node
from example_interfaces.srv import AddTwoInts
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
import time


class NodeA(Node):
    def __init__(self):
        super().__init__("node_a")
        self.cb_group = ReentrantCallbackGroup()

        # 1. B 클라이언트를 먼저 생성
        self.client_b = self.create_client(
            AddTwoInts, "service_b", callback_group=self.cb_group
        )

        # 2. B가 살아있는지 확실하게 체크 (B가 없으면 시작 안 함)
        self.get_logger().info("노드 B(service_b)가 켜질 때까지 대기합니다...")
        while not self.client_b.wait_for_service(timeout_sec=1.0):
            self.get_logger().info("노드 B 대기 중...")
        self.get_logger().info("🟢 노드 B 확인 완료!")

        # 3. B가 확인된 후, 자신의 서비스를 생성 (메인 노드가 이제 접근 가능)
        self.srv_a = self.create_service(
            AddTwoInts, "service_a", self.handle_service_a, callback_group=self.cb_group
        )
        self.get_logger().info("🟢 노드 A 준비 완료. (service_a 오픈)")

    async def handle_service_a(self, request, response):
        req_id = request.a
        self.get_logger().info(f"[A] 메인 요청 수신 (ID:{req_id}). 노드 B로 전달...")

        req_to_b = AddTwoInts.Request()
        req_to_b.a = request.a
        req_to_b.b = request.b

        # 비동기 호출로 B의 응답을 기다림
        future = self.client_b.call_async(req_to_b)
        result_from_b = await future

        self.get_logger().info(
            f"[A] B 응답 수신 (ID:{req_id}): {result_from_b.sum}. 자체 3초 연산 시작..."
        )

        # A 자체 무거운 연산 모사
        for i in range(3):
            time.sleep(1.0)

        response.sum = result_from_b.sum + 100
        self.get_logger().info(f"[A] 최종 연산 완료 (ID:{req_id}). 메인으로 반환.")
        return response


def main(args=None):
    rclpy.init(args=args)
    node = NodeA()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
