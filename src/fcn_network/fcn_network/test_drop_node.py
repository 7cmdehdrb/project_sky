import rclpy
import time
from rclpy.node import Node
from custom_msgs.srv import GetNextDropCell  # 패키지에 맞게 주석 해제


class DropGridSyncClient:
    """
    메인 노드 객체를 주입받아 ROS2 클라이언트 역할을 수행하는 유틸리티 클래스입니다.
    자체적으로 spin을 돌리지 않으며, 외부(메인 로직)의 spin 상태에 의존하여 통신합니다.
    """

    def __init__(self, node: Node):
        self._node = node  # 주입받은 메인 노드 인스턴스

        # 메인 노드의 기능을 빌려 클라이언트 생성
        self.client = self._node.create_client(GetNextDropCell, "request_drop_cell")

        # 서버 대기 로직 (이 역시 메인 노드의 로거 사용)
        while not self.client.wait_for_service(timeout_sec=1.0):
            self._node.get_logger().info("DropGrid 서버를 기다리는 중입니다...")

        self._node.get_logger().info("DropGrid 서버와 연결되었습니다.")

    def request_next_drop_cell_sync(self):
        """
        서버에 동기식으로 요청을 날리고 응답을 반환합니다.
        (주의: 이 함수가 호출되는 스레드와 메인 노드가 spin되는 스레드는 분리되어 있어야 합니다.)
        """
        request = GetNextDropCell.Request()
        future = self.client.call_async(request)

        # 메인 노드가 다른 스레드에서 spin 되고 있다고 가정하고,
        # 메인 로직 스레드만 future 결과가 나올 때까지 잠시 대기(블로킹)합니다.
        while rclpy.ok() and not future.done():
            time.sleep(0.01)  # CPU 점유율 폭주 방지

        if future.result() is not None:
            return future.result()
        else:
            self._node.get_logger().error("서비스 응답을 받지 못했습니다.")
            return None


class MainRobotControlNode(Node):
    def __init__(self):
        super().__init__("main_robot_control_node")

        # 1. 인스턴스화 시 노드 자신(self)을 주입하여 Client 초기화
        self.drop_client = DropGridSyncClient(self)

        self.get_logger().info("메인 제어 노드가 준비되었습니다.")

    def run_control_sequence(self):
        """
        이 함수는 ROS2 이벤트 루프(spin)를 블로킹하지 않도록
        별도의 스레드(Main Thread)에서 실행될 메인 제어 로직입니다.
        """
        self.get_logger().info("--- 제어 시퀀스 시작 ---")

        iteration = 1
        # r = self.create_rate(1.0)  # 1Hz 루프 주기 (필요에 따라 조정)
        while rclpy.ok():
            self.get_logger().info(f"[작업 {iteration}] 다음 목표점 계산 요청...")

            response = self.drop_client.request_next_drop_cell_sync()

            if response and response.success:
                self.get_logger().info(
                    f"[응답 수신 완료] 타겟 셀: {response.row_id}{response.col_id} | "
                    f"Position: {response.center_coord.x:.2f},  {response.center_coord.y:.2f}, {response.center_coord.z:.2f} | "
                    f"Frame: {response.frame_id}"
                )

                # 로봇 매니퓰레이터 제어, IK 계산 등의 메인 작업 수행 구간
            else:
                self.get_logger().warn("빈 그리드가 없으므로 작업을 마칩니다.")
                break

            iteration += 1
            time.sleep(1.0)  # CPU 점유율 폭주 방지
            # r.sleep()  # 루프 주기 유지


def main(args=None):
    rclpy.init(args=args)

    main_node = MainRobotControlNode()

    # 3. 중요: 메인 제어 로직(순차적 흐름)과 ROS2 Spin(이벤트 대기)을 분리
    # ROS2의 통신 콜백 처리는 백그라운드 스레드에 위임
    import threading

    spin_thread = threading.Thread(target=rclpy.spin, args=(main_node,), daemon=True)
    spin_thread.start()

    try:
        # 메인 스레드에서는 절차적인 동기식 제어 루프만 실행
        main_node.run_control_sequence()

    except KeyboardInterrupt:
        main_node.get_logger().info("사용자에 의한 중단")
    finally:
        main_node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
