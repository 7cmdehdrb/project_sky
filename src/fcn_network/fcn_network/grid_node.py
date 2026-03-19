import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_system_default
import os

# PointCloud2 처리를 위한 모듈 (Transformer 내부에서 pc2를 쓰므로 필수)
from std_msgs.msg import *
from sensor_msgs.msg import PointCloud2
from visualization_msgs.msg import MarkerArray
from std_msgs.msg import Float32MultiArray

# 구현하신 클래스 임포트 (경로는 패키지 구조에 맞게 수정하세요)
from fcn_network.grid_manager import GridManager
from base_package.header import PointCloudTransformer
from base_package.transform_manager import TransformManager


class GridDistancePublisherNode(Node):
    def __init__(self, grid_json_path: str):
        super().__init__("grid_distance_publisher_node")

        self._grid_json_path = grid_json_path
        self.get_logger().info(f"GridManager 초기화 중... (경로: {grid_json_path})")

        self.grid_manager = GridManager(resource_path=grid_json_path)
        self.transform_manager = TransformManager(node=self)

        # 2. 데이터 버퍼 (가장 최근 수신된 PointCloud2 메세지 저장)
        self._latest_pc_msg = None

        # 3. Subscriber (PointCloud2 수신)
        self.pc_sub = self.create_subscription(
            PointCloud2,
            "/camera/camera1/depth/color/points",  # 실제 사용하는 뎁스 카메라 PC 토픽명으로 변경
            self.pc_callback,
            qos_profile=qos_profile_system_default,
        )

        # 4. Publishers
        self.marker_pub = self.create_publisher(
            MarkerArray,
            "/grid_markers",
            qos_profile=qos_profile_system_default,
        )

        self.distance_pub = self.create_publisher(
            Float32MultiArray,
            "/front_object_distance",  # Policy 노드에서 구독할 토픽
            qos_profile=qos_profile_system_default,
        )

        self._wtf = self.create_publisher(
            PointCloud2,
            "/processed_pointcloud",  # 디버깅용 토픽 (처리된 포인트 클라우드 발행, 필요시 활용)
            qos_profile=qos_profile_system_default,
        )

        # 5. 2Hz 타이머 (0.5초 주기)
        self.timer = self.create_timer(0.5, self.process_and_publish)

        self.get_logger().info("🟢 Grid Distance Publisher 노드 준비 완료.")

    def pc_callback(self, msg: PointCloud2):
        """메세지 수신 시 버퍼에 최신화만 수행 (비동기 처리 최적화)"""
        self._latest_pc_msg = msg

    def process_and_publish(self):
        """2Hz 주기로 호출되어 Numpy 변환 -> 마커 갱신 -> 거리 계산 및 발행 수행"""
        if self._latest_pc_msg is not None:
            msg = self._latest_pc_msg

            # (1) PointCloud2 -> Numpy 변환 (RGB 미사용)
            try:
                mat = self.transform_manager.get_transform_matrix(
                    target_frame="camera1_link",
                    source_frame=self._latest_pc_msg.header.frame_id,
                )

                points_np = PointCloudTransformer.pointcloud2_to_numpy(msg, rgb=False)
                points_np = PointCloudTransformer.transform_pointcloud(
                    points=points_np, transform_matrix=mat
                )
            except Exception as e:
                self.get_logger().error(f"PointCloud 변환 실패: {e}")
                return

            # (3) Grid 업데이트 및 Marker Array 획득
            self.grid_manager.update_occupancy(points_np)

        else:
            self.get_logger().warn("아직 PointCloud2 메세지를 수신하지 못했습니다.")
            points_np = None  # 초기에는 포인트 데이터가 없으므로 None 처리

        # (2) Grid 업데이트 및 Marker Array 획득
        marker_array = self.grid_manager.get_marker_array(
            header=Header(
                stamp=self.get_clock().now().to_msg(), frame_id="camera1_link"
            ),
            points=points_np,
        )

        # (3) Marker Array 발행 (Rviz 확인용)
        self.marker_pub.publish(marker_array)

        # (4) 가장 가까운 ROW 문자 탐색 (예: {0: 'A', 1: 'C', 2: None, 3: 'B'})
        frontmost_dict = self.grid_manager.get_frontmost_row_by_col()

        # (5) ROW 문자를 기반으로 거리(Distance) 계산
        distance_list = []

        # 컬럼 인덱스 순서(0, 1, 2, 3...)대로 정렬하여 순회
        for col in sorted(frontmost_dict.keys()):
            row_str = frontmost_dict[col]

            if row_str is None:
                # 비어있을 경우 거리를 0.0으로 처리 (혹은 무한대 등 시스템 상황에 맞게 변경)
                distance_list.append(0.0)
            else:
                # 아스키코드를 활용한 자동 거리 증가 (A=65)
                # 'A' -> 1.5, 'B' -> 2.5, 'C' -> 3.5 ...
                dist = 1.5 + (ord(row_str) - 65) * 1.0
                distance_list.append(float(dist))

        # (6) Float32MultiArray로 발행
        dist_msg = Float32MultiArray(data=distance_list)
        self.distance_pub.publish(dist_msg)

        # (선택) 디버깅용 로그
        # self.get_logger().info(f"Published Distances: {distance_list}")


def main(args=None):
    rclpy.init(args=args)
    node = GridDistancePublisherNode(
        grid_json_path="/home/min/7cmdehdrb/project_sky/src/fcn_network/resource/grid_data.json"
    )
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
