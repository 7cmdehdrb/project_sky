# Python
import argparse
import sys
from typing import List

# NumPy
import numpy as np

# OpenCV & CV Bridge (오버레이 기능용 추가)
import cv2
from cv_bridge import CvBridge

# ROS2
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_system_default

# ROS2 Messages
from custom_msgs.msg import BoundingBox, BoundingBoxMultiArray
from sensor_msgs.msg import Image
from std_msgs.msg import Int32MultiArray

# Custom Packages
from base_package.image_manager import ImageManager
from base_package.object_manager import ObjectManager


class ClosestObjectClassifierNode(Node):
    def __init__(self, *args, **kwargs):
        super().__init__("closest_object_classifier")

        self._node = self

        self._detected_objects = []

        # 4개의 구역을 나누는 X축 픽셀 경계값
        self._boundary = [170, 300, 460]

        self._depth_raw = None

        self._image_manager = ImageManager(
            self._node,
            subscribed_topics=[
                {
                    "topic_name": "/camera/camera1/depth/image_rect_raw",
                    "callback": self.depth_callback,
                },
            ],
            published_topics=[],
            *args,
            **kwargs,
        )

        self._object_manager = ObjectManager(self._node, *args, **kwargs)

        self._node.create_subscription(
            BoundingBoxMultiArray,
            "/real_time_segmentation_node/segmented_bbox",
            self.bbox_callback,
            qos_profile=qos_profile_system_default,
        )

        # 가장 가까운 객체 ID 배열 발행 (기존)
        self._result_publisher = self._node.create_publisher(
            Int32MultiArray,
            self.get_name() + "/closest_object_ids",
            qos_profile_system_default,
        )

        # 시각화된 오버레이 이미지 발행 (신규)
        self._overlay_publisher = self._node.create_publisher(
            Image,
            self.get_name() + "/closest_object_overlay",
            qos_profile_system_default,
        )
        self.bridge = CvBridge()

    @property
    def detected_objects(self):
        return self._detected_objects

    @detected_objects.setter
    def detected_objects(self, value: List[int]):
        self._detected_objects = value

    def depth_callback(self, msg: Image):
        self._depth_raw = self._image_manager.decode_message(
            msg, desired_encoding="16UC1"
        )

    def bbox_callback(self, msg: BoundingBoxMultiArray):
        detected_objects = []
        for bbox in msg.data:
            bbox: BoundingBox
            class_name = str(bbox.cls)
            mask = np.reshape(np.array(bbox.mask_data), (bbox.mask_row, bbox.mask_col))
            detected_objects.append({"class_name": class_name, "mask": mask})

        self._detected_objects = detected_objects
        self.process_and_publish_closest_objects()

    def remove_outliers(self, depth_image: np.ndarray):
        try:
            result = depth_image[depth_image < 1240]
            return result
        except Exception as e:
            self._node.get_logger().error(f"Error removing outliers: {e}")
            return depth_image

    def publish_overlay_image(
        self, depth_image: np.ndarray, columns: dict, num_cols: int
    ):
        """
        Depth 이미지를 Grayscale로 변환 후, Segmentation 마스크와
        구역별 가장 가까운 객체의 이름을 오버레이하여 발행합니다.
        """
        h, w = depth_image.shape[:2]

        # 1. 16비트 Depth 이미지를 8비트 그레이스케일(0~255)로 정규화 후 BGR 컬러 이미지로 변환
        depth_vis = cv2.normalize(
            depth_image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U
        )
        overlay_img = cv2.cvtColor(depth_vis, cv2.COLOR_GRAY2BGR)

        # 2. 구역 경계선(Boundary) 그리기
        for b_val in self._boundary:
            cv2.line(overlay_img, (int(b_val), 0), (int(b_val), h), (255, 255, 255), 2)

        # 3. 모든 탐지된 객체의 마스크 덮어쓰기
        for obj in self._detected_objects:
            mask = obj["mask"].astype(bool)
            class_name = obj["class_name"]

            # 앞서 작성했던 get_color 활용 (RGB -> BGR 변환)
            try:
                r, g, b = self._object_manager.get_color(class_name)
                color_bgr = (int(b), int(g), int(r))
            except AttributeError:
                color_bgr = (0, 255, 255)  # 함수가 없으면 노란색으로 처리

            # 색상 마스크 생성 후 투명도(Alpha) 블렌딩 적용
            colored_mask = np.zeros_like(overlay_img)
            colored_mask[mask] = color_bgr
            cv2.addWeighted(colored_mask, 0.5, overlay_img, 1.0, 0, overlay_img)

        # 4. 각 컬럼별로 가장 가까운 객체의 텍스트(클래스 이름) 출력
        font = cv2.FONT_HERSHEY_SIMPLEX
        for col_idx, objects_in_col in columns.items():
            if objects_in_col:
                # 거리순 정렬되어 있으므로 0번째가 가장 가까운 객체
                closest_obj = objects_in_col[0]
                class_name = closest_obj["class_name"]

                # 해당 컬럼의 중심 X 좌표 계산
                start_x = self._boundary[col_idx - 1] if col_idx > 0 else 0
                end_x = self._boundary[col_idx] if col_idx < len(self._boundary) else w
                center_x = int((start_x + end_x) / 2)

                text = str(self._object_manager.names[class_name])  # 정규화된 이름 사용
                text_size = cv2.getTextSize(text, font, 0.6, 2)[0]
                text_x = center_x - text_size[0] // 2
                text_y = 30  # 화면 상단에 텍스트 배치

                # 글씨가 잘 보이도록 검은색 배경 박스 추가
                cv2.rectangle(
                    overlay_img,
                    (text_x - 5, text_y - text_size[1] - 5),
                    (text_x + text_size[0] + 5, text_y + 5),
                    (0, 0, 0),
                    -1,
                )
                cv2.putText(
                    overlay_img, text, (text_x, text_y), font, 0.6, (0, 255, 0), 2
                )

        # 5. 오버레이된 이미지를 ROS 토픽으로 발행
        img_msg = self.bridge.cv2_to_imgmsg(overlay_img, encoding="bgr8")
        self._overlay_publisher.publish(img_msg)

    def process_and_publish_closest_objects(self):
        num_cols = len(self._boundary) + 1
        default_ids = [-1] * num_cols

        if self._depth_raw is None:
            self._node.get_logger().warn(
                "Depth image not available yet. Publishing default closest IDs."
            )
            result_msg = Int32MultiArray()
            result_msg.data = default_ids
            self._result_publisher.publish(result_msg)
            return

        depth_image = self._image_manager.crop_image(self._depth_raw)
        zero_pixel = np.zeros((480, 40), dtype=np.uint16)
        depth_image = np.hstack([depth_image, zero_pixel])
        depth_image = depth_image[:, 40:]

        columns = {i: [] for i in range(num_cols)}

        for obj in self._detected_objects:
            mask = obj["mask"].astype(bool)

            mask_depth = depth_image[mask]
            mask_depth = mask_depth[mask_depth > 0]
            mask_depth = self.remove_outliers(mask_depth)

            if len(mask_depth) == 0:
                continue

            mean_distance = np.mean(mask_depth)

            mask_x = np.where(mask)[1]
            if len(mask_x) == 0:
                continue
            center_x = np.mean(mask_x)

            col_idx = num_cols - 1
            for i, b_val in enumerate(self._boundary):
                if center_x < b_val:
                    col_idx = i
                    break

            obj_id = self._object_manager.get_object_id(obj["class_name"])

            # 시각화 때 사용할 class_name도 함께 저장하도록 수정
            columns[col_idx].append(
                {
                    "id": obj_id,
                    "distance": mean_distance,
                    "class_name": obj["class_name"],
                }
            )

        closest_ids = [-1] * num_cols

        for col_idx, objects_in_col in columns.items():
            if objects_in_col:
                objects_in_col.sort(key=lambda x: x["distance"])
                closest_ids[col_idx] = objects_in_col[0]["id"]

        result_msg = Int32MultiArray()
        result_msg.data = closest_ids
        self._result_publisher.publish(result_msg)

        # 연산이 모두 끝난 후 시각화 이미지 오버레이 및 발행 함수 호출
        self.publish_overlay_image(depth_image, columns, num_cols)

        self.get_logger().info(f"Published closest object IDs: {closest_ids}")


def main(args=None):
    rclpy.init(args=args)
    classifier_node = ClosestObjectClassifierNode()
    rclpy.spin(node=classifier_node)
    classifier_node.destroy_node()


if __name__ == "__main__":
    main()
