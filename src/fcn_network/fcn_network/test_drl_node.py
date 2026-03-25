import rclpy
from rclpy.node import Node
from custom_msgs.srv import GetPolicyAction


class MockMainNode(Node):
    def __init__(self, num_columns: int = 4):
        super().__init__("mock_main_node")

        if num_columns not in (4, 5):
            raise ValueError("num_columns must be 4 or 5")

        self.num_columns = num_columns

        # action description mapping
        self._action_descriptions = {
            0: "잡기",
            1: "오른쪽 밀기",
            2: "왼쪽 밀기",
        }

        self.client_a = self.create_client(GetPolicyAction, "get_policy_action")

        self.get_logger().info(
            f"Node A (Policy Server) 대기 중... (열 개수: {num_columns})"
        )

        while not self.client_a.wait_for_service(timeout_sec=1.0):
            self.get_logger().info("Node A가 켜질 때까지 기다리는 중...")

        self.get_logger().info("🟢 Node A 확인 완료!")
        self.get_logger().info(
            "✨ 엔터를 눌러 제어 요청을 보내세요. 종료하려면 'q' + 엔터를 입력하세요."
        )

        self.request_count = 0

    def send_request(self):
        self.request_count += 1

        dummy_target_id = 0

        req = GetPolicyAction.Request()
        req.target_id = dummy_target_id

        self.get_logger().info(
            f"▶️ [Main] {self.request_count}번째 요청 발송 (Target ID: {dummy_target_id}).."
        )

        future = self.client_a.call_async(req)
        future.add_done_callback(
            lambda fut, req_num=self.request_count: self.response_callback(fut, req_num)
        )

    def response_callback(self, future: rclpy.Future, req_num: int):
        try:
            result = future.result()
            action = result.action_type
            col = result.target_column

            # 안전하게 처리 (action이 범위 밖일 수도 있으므로)
            action_str = self._action_descriptions.get(action, f"알 수 없음 ({action})")

            # 시각화 생성 (■ = 타겟, □ = 나머지)
            target_visual = ["□"] * self.num_columns
            target_visual[col] = "■"
            visual_str = "".join(target_visual)

            self.get_logger().info(
                f"✅ [Main] {req_num}번째 응답 수신 성공! Action: {action} | Target Column: {col}\n"
                f"{action_str} -> {visual_str}"
            )
        except Exception as e:
            self.get_logger().error(f"❌ [Main] {req_num}번째 호출 실패: {e}")


def main(args=None):
    # ✅ num_columns = 4 (예시) — 필요시 5로 변경 가능
    NUM_COLUMNS = 4  # ← 여기서 조정하세요 (4 또는 5)

    rclpy.init(args=args)
    try:
        node = MockMainNode(num_columns=NUM_COLUMNS)
    except ValueError as e:
        print(f"[ERROR] Failed to initialize node: {e}")
        return

    # rclpy.spin()은 별도 스레드에서 실행
    import threading

    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()

    try:
        while rclpy.ok():
            user_input = (
                input(" Press Enter to send request, 'q' to quit: ").strip().lower()
            )
            if user_input == "q":
                break
            elif user_input == "":
                node.send_request()
            else:
                node.get_logger().info("Press Enter or 'q' only.")

    except KeyboardInterrupt:
        node.get_logger().info("Interrupted by user (Ctrl+C)")
    finally:
        node.get_logger().info("Shutting down...")
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
