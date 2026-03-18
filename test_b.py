import rclpy
from rclpy.node import Node
from example_interfaces.srv import AddTwoInts
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
import time


class NodeB(Node):
    def __init__(self):
        super().__init__("node_b")
        self.cb_group = ReentrantCallbackGroup()
        self.srv = self.create_service(
            AddTwoInts, "service_b", self.handle_service_b, callback_group=self.cb_group
        )

        self.get_logger().info("🟢 노드 B 준비 완료. (service_b 오픈)")

    def handle_service_b(self, request, response):
        req_id = request.a  # 식별을 위해 request.a 값을 ID처럼 사용
        self.get_logger().info(f"[B] 요청 수신 (ID:{req_id}). 3초 연산 시작...")

        # 무거운 연산 모사
        for i in range(3):
            time.sleep(1.0)

        response.sum = request.a + request.b
        self.get_logger().info(f"[B] 연산 완료 (ID:{req_id}).")
        return response


def main(args=None):
    rclpy.init(args=args)
    node = NodeB()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
