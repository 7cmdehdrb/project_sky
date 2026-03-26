# Python
import argparse
import json
import os
import sys
import re
from collections import UserDict
from PIL import Image as PILImage

# OpenCV
import cv2
from cv_bridge import CvBridge

# NumPy
import numpy as np

# ROS2
import rclpy
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import QoSProfile, qos_profile_system_default
from rclpy.time import Time

# ROS2 Messages
from custom_msgs.msg import BoundingBox, BoundingBoxMultiArray
from geometry_msgs.msg import *
from nav_msgs.msg import *
from sensor_msgs.msg import *
from std_msgs.msg import *
from visualization_msgs.msg import *

# TF
from tf2_ros import *

# YOLO
from ultralytics import YOLO
from ultralytics.engine.results import Boxes, Masks, Results

# Custom Packages
from ament_index_python.packages import get_package_share_directory
from base_package.object_manager import ObjectManager
from base_package.image_manager import ImageManager


# ==========================================
# 2. Refactored YoloManager
# ==========================================
class YoloManager:
    def __init__(self, node: Node, model_path: str, *arg, **kwargs):
        self._node = node
        self._bridge = CvBridge()
        self._image_manager = ImageManager(
            self._node, subscribed_topics=[], published_topics=[], *arg, **kwargs
        )

        self._model_path = model_path
        self._node.get_logger().info(f"Loading YOLO model from: {self._model_path}")

        self._model = YOLO(self._model_path, verbose=False)
        self._model.eval()

    def preprocess_image(self, image_data):
        """ROS Image, NumPy Array, PIL Image 어떤 타입이든 Numpy 기반 RGB 이미지로 통일"""
        if isinstance(image_data, Image):
            np_image = self._bridge.imgmsg_to_cv2(image_data, desired_encoding="rgb8")
        elif isinstance(image_data, np.ndarray):
            np_image = image_data
        elif isinstance(image_data, PILImage.Image):
            np_image = np.array(image_data)
        else:
            raise ValueError("Unsupported image type provided to YoloManager.")

        # 크롭 로직 내재화
        np_image = self._image_manager.crop_image(img=np_image)

        pil_image = PILImage.fromarray(np_image)
        return pil_image, np_image

    def predict(self, image_data: Image | np.ndarray | PILImage.Image):
        """이미지 전처리 후 YOLO 추론 수행. 결과 객체와 원본 Numpy 이미지를 함께 반환"""
        pil_image, np_image = self.preprocess_image(image_data)
        results = self._model(pil_image, verbose=False)
        return results[0], np_image


# ==========================================
# 3. RealTimeSegmentationNode
# ==========================================
class RealTimeSegmentationNode(Node):
    def __init__(
        self,
        *arg,
        **kwargs,
    ):
        super().__init__("real_time_segmentation_node")

        self.declare_parameters(
            namespace="",
            parameters=[
                (
                    "model_file",
                    "/home/min/7cmdehdrb/project_sky/src/object_tracker/resource/Yolo/best_45_0326.pt",
                ),
                (
                    "obj_bounds_file",
                    "/home/min/7cmdehdrb/project_sky/src/object_tracker/resource/obj_bounds.json",
                ),
                ("conf_threshold", 0.7),
            ],
        )

        model_file = self.get_parameter("model_file").get_parameter_value().string_value
        obj_bounds_file = (
            self.get_parameter("obj_bounds_file").get_parameter_value().string_value
        )
        conf_threshold = (
            self.get_parameter("conf_threshold").get_parameter_value().double_value
        )

        self.get_logger().info(f"Model file: {model_file}")
        self.get_logger().info(f"Object bounds file: {obj_bounds_file}")
        self.get_logger().info(f"Confidence threshold: {conf_threshold}")

        # >>> Managers >>>
        self._yolo_manager = YoloManager(self, model_path=model_file, *arg, **kwargs)
        self._object_manager = ObjectManager(self, *arg, **kwargs)
        self._image_manager = ImageManager(
            self,
            subscribed_topics=[
                {
                    "topic_name": "/camera/camera1/color/image_raw",
                    "callback": self.image_callback,
                }
            ],
            published_topics=[{"topic_name": self.get_name() + "/segmented_image"}],
            *arg,
            **kwargs,
        )
        # <<< Managers <<<

        # >>> ROS2 Publishers >>>
        self.segmented_bbox_publisher = self.create_publisher(
            BoundingBoxMultiArray,
            self.get_name() + "/segmented_bbox",
            qos_profile=qos_profile_system_default,
        )

        # >>> Load Parameters & Data >>>
        self._conf_threshold = conf_threshold
        self._load_obj_bounds(obj_bounds_file)

    def _load_obj_bounds(self, file_name):
        if not os.path.isabs(file_name):
            raise ValueError(f"Object bounds file path must be absolute: {file_name}")

        with open(file_name, "r") as f:
            self._obj_bounds: dict = json.load(f)

    def image_callback(self, msg: Image):
        img_msg, bbox_msg = self.do_segmentation(msg=msg)

        self._image_manager.publish(
            topic_name=self.get_name() + "/segmented_image", msg=img_msg
        )
        self.segmented_bbox_publisher.publish(bbox_msg)

    def do_segmentation(self, msg: Image):
        # 1. YoloManager를 통한 통합 추론 (전처리 자동 수행)
        result: Results
        result, np_image = self._yolo_manager.predict(msg)

        boxes: Boxes = result.boxes
        classes: dict = result.names
        masks: Masks = result.masks

        np_boxes = boxes.xyxy.cpu().numpy()
        np_confs = boxes.conf.cpu().numpy()
        np_cls = boxes.cls.cpu().numpy()

        # [수정] mask_data 처리 문제 해결: 원본 이미지 크기로 마스크 복원
        np_masks = None
        if masks is not None:
            # 원본 이미지 사이즈를 명시적으로 넘겨서 리사이즈된 마스크 데이터를 얻습니다.
            np_masks = masks.data.cpu().numpy()
            # YOLO v8 이상에서는 마스크 리사이즈를 위한 유틸리티 함수를 제공하기도 하나,
            # 가장 안전한 방법은 OpenCV를 이용해 직접 리사이즈 하는 것입니다.
            # (아래 for문 내에서 개별 마스크마다 리사이즈 수행)

        bboxes = BoundingBoxMultiArray()
        overlay_data_list = []

        # 원본 이미지의 크기 저장 (리사이즈 시 필요)
        orig_h, orig_w = np_image.shape[:2]

        # 2. 필터링 및 데이터 추출
        for idx in range(len(boxes)):
            conf = np_confs[idx]
            if conf < self._conf_threshold:
                continue

            raw_cls = classes[int(np_cls[idx])]
            # 새로운 ObjectManager 적용: 항상 깔끔한 이름 반환 (예: "coca_cola")
            clean_cls = self._object_manager.names[raw_cls]

            x1, y1, x2, y2 = map(int, np_boxes[idx])
            detected_ratio = abs(x1 - x2) / max(abs(y1 - y2), 1)  # 0 나누기 방지

            # 3. 비율 기반 필터링
            desired_data = self._obj_bounds.get(clean_cls)
            difference_ratio = 1.0  # 기본값 (json에 데이터가 없을 경우 등)

            # if desired_data:
            #     numerator = (float(desired_data["x"]) + float(desired_data["z"])) / 2.0
            #     denominator = float(desired_data["y"])
            #     desired_ratio = numerator / denominator

            #     difference_ratio = (
            #         detected_ratio / desired_ratio
            #         if detected_ratio > desired_ratio
            #         else desired_ratio / detected_ratio
            #     )

            # 4. BoundingBox 메시지 생성 (깔끔해진 이름 사용)
            mask_flat = []
            mask_shape = (0, 0)

            # [수정] 마스크 데이터 리사이즈 및 이진화 처리
            if np_masks is not None and np_masks[idx] is not None:
                current_mask = np_masks[idx]

                # YOLO 마스크는 주로 (160, 160) 처럼 축소되어 나오므로, 원본 이미지 사이즈로 cv2.resize 수행
                resized_mask = cv2.resize(
                    current_mask, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR
                )

                # 0.5를 기준으로 이진화(0 또는 1) 후 정수형으로 변환
                binary_mask: np.ndarray = (resized_mask > 0.5).astype(np.int32)

                mask_shape = binary_mask.shape
                mask_flat = binary_mask.flatten().tolist()

            bboxes.data.append(
                BoundingBox(
                    id=int(np_cls[idx]),
                    cls=clean_cls,  # 정제된 이름
                    conf=float(conf),
                    bbox=[x1, y1, x2, y2],
                    mask_row=mask_shape[0],
                    mask_col=mask_shape[1],
                    mask_data=mask_flat,
                )
            )

            # 시각화를 위한 데이터 저장 (마스크 오버레이는 이제 안 하므로 mask 정보 제외)
            overlay_data_list.append(
                {
                    "bbox": (x1, y1, x2, y2),
                    "cls_name": clean_cls,
                    "conf": conf,
                    "diff_ratio": difference_ratio,
                }
            )

        # 5. 오버레이 드로잉 (별도 함수 호출)
        np_image = self._draw_overlay(np_image, overlay_data_list)
        segmented_image = self._image_manager.encode_message(np_image, encoding="rgb8")

        return segmented_image, bboxes

    def _draw_overlay(self, np_image, overlay_data_list):
        """이미지에 바운딩 박스와 텍스트만 그립니다."""
        for data in overlay_data_list:
            x1, y1, x2, y2 = data["bbox"]
            cls_name = data["cls_name"]

            # 색상 결정
            draw_color = self._object_manager.get_color(text=cls_name)

            # [수정] 마스크 오버레이 영역 완전 삭제

            # 바운딩 박스 및 라벨 텍스트
            label = f"{data['cls_name']}, {data['conf']:.2f}"
            cv2.rectangle(np_image, (x1, y1), (x2, y2), draw_color, 2)
            cv2.putText(
                img=np_image,
                text=label,
                org=(x1, max(y1 - 10, 10)),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.5,
                color=draw_color,
                thickness=2,
            )

        return np_image


def main(args=None):
    rclpy.init(args=args)

    """
    Callback에서 YOLO 추론 → BoundingBoxMultiArray 생성 → 이미지에 바운딩 박스, 마스크, 텍스트 오버레이 → ROS2 토픽으로 발행.
    별도의 run() 함수 없이, 모든 로직이 Node 클래스 내부에 깔끔하게 캡슐화되어 있습니다.
    """

    node = RealTimeSegmentationNode()

    rclpy.spin(node=node)
    node.destroy_node()


if __name__ == "__main__":
    main()
