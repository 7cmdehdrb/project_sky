# ROS2
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_system_default

# Message
from std_msgs.msg import Header
from sensor_msgs.msg import Image

# Python
import numpy as np
import cv2
from cv_bridge import CvBridge


class FakeCameraPublisher(Node):
    def __init__(self):
        super().__init__("fake_camera_publisher")

        # RGB 이미지 Publisher
        self.rgb_publisher = self.create_publisher(
            Image,
            "/camera/camera1/color/image_raw",
            qos_profile=qos_profile_system_default,
        )

        # Depth 이미지 Publisher (토픽명 변경)
        self.depth_publisher = self.create_publisher(
            Image,
            "/camera/camera1/depth/image_rect_raw",
            qos_profile=qos_profile_system_default,
        )

        self.bridge = CvBridge()

        # 1. 원본 RGB 이미지 로드
        image_path = (
            "/home/min/7cmdehdrb/project_sky/src/object_tracker/resource/image_0018.png"
        )
        self.rgb_image = cv2.imread(image_path, cv2.IMREAD_COLOR)

        if self.rgb_image is None:
            self.get_logger().error(f"이미지를 불러올 수 없습니다: {image_path}")
            return

        # 2. 가짜 Depth 이미지(16UC1) 생성
        # RGB -> Grayscale 변환 후, 16비트로 확장 및 반전 연산 (700~1200 범위)
        gray = cv2.cvtColor(self.rgb_image, cv2.COLOR_BGR2GRAY)
        self.depth_image = 1200 - (gray.astype(np.uint16) * 2)

        self.get_logger().info("RGB 및 Mock Depth 이미지 발행 준비 완료 (30Hz)")

        hz = 30
        self.timer = self.create_timer(float(1.0 / hz), self.run)

    def run(self):
        # 공통 Header 생성 (동일한 타임스탬프 부여)
        header = Header(
            stamp=self.get_clock().now().to_msg(),
            frame_id="camera1_color_optical_frame",
        )

        # RGB 이미지 메시지 변환 및 발행
        rgb_msg = self.bridge.cv2_to_imgmsg(self.rgb_image, "bgr8", header=header)
        self.rgb_publisher.publish(rgb_msg)

        # Depth 이미지 메시지 변환 및 발행 (인코딩: 16UC1)
        depth_msg = self.bridge.cv2_to_imgmsg(self.depth_image, "16UC1", header=header)
        self.depth_publisher.publish(depth_msg)


def main():
    rclpy.init(args=None)

    node = FakeCameraPublisher()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("종료 요청이 들어와 노드를 정지합니다.")
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
