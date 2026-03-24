import os
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_system_default

# ROS2 Messages
from std_msgs.msg import *
from sensor_msgs.msg import *
from visualization_msgs.msg import *
from geometry_msgs.msg import *
from custom_msgs.srv import GetNextDropCell

# 구현하신 클래스 임포트 (경로는 패키지 구조에 맞게 수정하세요)
from fcn_network.drop_grid_manager import DropGridManager, DropPriority, DropDirection
from base_package.header import PointCloudTransformer
from base_package.transform_manager import TransformManager


import time


class DropGridNode(Node):
    def __init__(self):
        super().__init__("drop_grid_node")

        self.declare_parameter(
            "drop_grid_json_path",
            "/home/min/7cmdehdrb/project_sky/src/fcn_network/resource/drop_grid_data.json",
        )

        self._grid_json_path = (
            self.get_parameter("drop_grid_json_path").get_parameter_value().string_value
        )

        self.drop_grid_manager = DropGridManager(
            resource_path=self._grid_json_path,
            priority=DropPriority.COL_FIRST,
            row_dir=DropDirection.FORWARD,
            col_dir=DropDirection.FORWARD,
        )

        self.marker_pub = self.create_publisher(
            MarkerArray,
            self.get_name() + "/drop_grid_markers",
            qos_profile=qos_profile_system_default,
        )

        # --- 추가 1: ROS Service Server 구축 ---
        # GetNextDropCell 타입의 서비스를 'request_drop_cell' 이름으로 생성
        self.drop_service = self.create_service(
            GetNextDropCell, "request_drop_cell", self.handle_drop_request
        )
        self.get_logger().info("Service Server 'request_drop_cell' 가 준비되었습니다.")

        self.get_logger().info(
            f"DropGridManager 초기화 완료 (경로: {self._grid_json_path})"
        )

    # --- 추가 2: Service Callback ---
    def handle_drop_request(
        self, request: GetNextDropCell.Request, response: GetNextDropCell.Response
    ):
        """서비스 요청이 들어오면 다음 셀을 획득하고 Drop 처리 후 응답합니다."""

        self.get_logger().info("서비스 요청이 들어왔습니다. 다음 셀을 계산합니다...")

        # 1. 다음에 넣어야 하는 셀 획득
        target_cell = self.drop_grid_manager.get_next_drop_cell()

        if target_cell is None:
            self.get_logger().warn("모든 그리드 셀이 이미 채워졌습니다 (Drop 실패).")
            response.success = False
            response.row_id = ""
            response.col_id = -1
            return response

        # 2. Response에 데이터 담기
        response.success = True
        response.row_id = target_cell.row
        response.col_id = target_cell.col
        response.center_coord = PoseStamped(
            header=Header(
                stamp=self.get_clock().now().to_msg(),
                frame_id="camera1_link",  # 실제 프레임에 맞게 수정하세요
            ),
            pose=Pose(
                position=target_cell.position,
                orientation=Quaternion(x=0.0, y=0.0, z=0.0, w=1.0),  # 필요에 따라 수정
            ),
        )
        response.size = target_cell.size

        # 3. drop 콜하여 상태를 '점유'로 변경 및 다음 인덱스로 진행
        self.drop_grid_manager.drop()

        self.get_logger().info(f"Drop 완료: 셀 [{target_cell.id}]")

        return response

    def publish_grid_markers(self):

        header = Header(
            stamp=self.get_clock().now().to_msg(),
            frame_id="camera1_link",  # 뭔가 더럽게 마음에 안드네
        )

        marker_array = self.drop_grid_manager.get_marker_array(
            header=header, points=None
        )

        self.marker_pub.publish(marker_array)


def main(args=None):
    rclpy.init(args=args)

    node = DropGridNode()

    import threading

    th = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    th.start()  # 별도의 스레드에서 노드 스핀 시작 (서비스 대기)

    r = node.create_rate(1.0)  # 1Hz 루프 주기 (필요에 따라 조정)
    while rclpy.ok():
        node.publish_grid_markers()
        r.sleep()  # 루프 주기 유지

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
