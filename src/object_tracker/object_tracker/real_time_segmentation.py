# ROS2
import rclpy
from rclpy.node import Node
from rclpy.time import Time
from rclpy.duration import Duration
from rclpy.qos import QoSProfile, qos_profile_system_default

# Message
from std_msgs.msg import *
from geometry_msgs.msg import *
from sensor_msgs.msg import *
from nav_msgs.msg import *
from visualization_msgs.msg import *
from custom_msgs.msg import BoundingBox, BoundingBoxMultiArray

# TF
from tf2_ros import *

# Python
import os
import sys
import numpy as np
import json
from PIL import ImageEnhance
from PIL import Image as PILImage

# OpenCV
import cv2
from cv_bridge import CvBridge


# YOLO
from ultralytics.engine.results import Results, Masks, Boxes
from ultralytics import YOLO


class RealTimeSegmentationNode(Node):
    def __init__(
        self,
        camera_topic: str = "/camera/camera1/color/image_raw",
    ):
        super().__init__("real_time_segmentation_node")

        resource_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "resource"
        )
        model_path = os.path.join(resource_path, "best_hg.pt")

        # Load YOLO v11 Model
        self.model = YOLO(model_path, verbose=False)
        self.model.eval()
        # Load CV Bridge
        self.bridge = CvBridge()

        # ROS
        self.raw_image_subscriber = self.create_subscription(
            Image,
            camera_topic,
            self.image_callback,
            qos_profile=qos_profile_system_default,
        )
        self.segmented_image_publisher = self.create_publisher(
            Image,
            self.get_name() + "/segmented_image",
            qos_profile=qos_profile_system_default,
        )
        self.segmented_bbox_publisher = self.create_publisher(
            BoundingBoxMultiArray,
            self.get_name() + "/segmented_bbox",
            qos_profile=qos_profile_system_default,
        )
        self.image = None

        # Parameters
        resource_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "..", "resource"
        )

        with open(
            os.path.join(resource_dir, "sim_stats.json"),
            "r",
        ) as f:
            self.stat = json.load(f)

        self.conf_threshold = 0.7
        self.color_dict = {
            0: (255, 0, 0),  # Red
            1: (0, 255, 0),  # Green
            2: (0, 0, 255),  # Blue
            3: (255, 255, 0),  # Yellow
            4: (255, 0, 255),  # Magenta
            5: (0, 255, 255),  # Cyan
            6: (128, 0, 0),  # Dark Red
            7: (0, 128, 0),  # Dark Green
            8: (0, 0, 128),  # Navy
            9: (128, 128, 0),  # Olive
            10: (128, 0, 128),  # Purple
            11: (0, 128, 128),  # Teal
            12: (192, 192, 192),  # Silver
            13: (255, 165, 0),  # Orange
            14: (0, 0, 0),  # Black
        }
        self.do_adjust_color = False
        self.do_crop_image = True
        self.do_publish_segmented_image = True

    def image_callback(self, msg: Image):
        self.image = msg

        img_msg, bbox_msg = self.image_loop(msg=msg)

        if self.do_publish_segmented_image:
            self.segmented_image_publisher.publish(img_msg)

        self.segmented_bbox_publisher.publish(bbox_msg)

    @staticmethod
    def adjust_sim_image(img: np.array, stats: dict):
        # YOLO 모델 입력을 위해 RGB 변환
        pil_img = PILImage.fromarray(img)
        img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

        # Color Statistics
        curr_brightness = np.mean(img_hsv[:, :, 2])
        curr_saturation = np.mean(img_hsv[:, :, 1])
        curr_contrast = np.std(img_hsv[:, :, 2])

        # enhancement factor 계산 (0으로 나누는 경우 방지)
        brightness_factor = stats["avg_brightness"] / (curr_brightness + 1e-8)
        saturation_factor = stats["avg_saturation"] / (curr_saturation + 1e-8)
        contrast_factor = stats["avg_contrast"] / (curr_contrast + 1e-8)

        # PIL ImageEnhance 모듈로 순차적으로 조정
        # 1) 명도 보정
        enhancer = ImageEnhance.Brightness(pil_img)
        pil_img = enhancer.enhance(brightness_factor)

        # 2) 채도 보정
        enhancer = ImageEnhance.Color(pil_img)
        pil_img = enhancer.enhance(saturation_factor)

        # 3) 대조 보정
        enhancer = ImageEnhance.Contrast(pil_img)
        pil_img = enhancer.enhance(contrast_factor)

        return pil_img

    @staticmethod
    def crop_image(img: np.array, crop_w: int = 640, crop_h: int = 480):
        h, w = img.shape[:2]

        if h == crop_h and w == crop_w:
            return img

        # Crop Image
        h, w = img.shape[:2]  # h=720, w=1280
        crop_w, crop_h = 640, 480
        start_x = int((w - crop_w) // 2.05)
        start_y = int((h - crop_h) // 2.7)

        # 크롭 범위 보정
        start_x = max(0, min(start_x, w - crop_w))
        start_y = max(0, min(start_y, h - crop_h))

        # 이미지 크롭 (640x480)
        cropped_img = img[start_y : start_y + crop_h, start_x : start_x + crop_w]

        return cropped_img

    def image_loop(self, msg: Image):
        # Load Image
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")
        cv_image_array = np.asanyarray(cv_image)

        if self.do_crop_image:
            cv_image_array = RealTimeSegmentationNode.crop_image(cv_image_array)

        if self.do_adjust_color:
            cv_image_array = RealTimeSegmentationNode.adjust_sim_image(
                cv_image_array, self.stat
            )

        pil_image = PILImage.fromarray(cv_image_array)

        # YOLO 세그멘테이션 수행
        results = self.model(pil_image, verbose=False)

        if len(results) == 0:
            self.get_logger().info("No object detected.")
            return None, None

        result: Results = results[0]

        boxes: Boxes = result.boxes
        classes: dict = result.names

        np_boxes = boxes.xyxy.cpu().numpy()
        np_confs = boxes.conf.cpu().numpy()
        np_cls = boxes.cls.cpu().numpy()

        # 바운딩 박스 그리기
        if boxes is None:
            self.get_logger().info("No box detected.")
            return None, None

        bboxes = BoundingBoxMultiArray()

        for idx in range(len(boxes)):
            id = int(np_cls[idx])
            conf = np_confs[idx]  # 신뢰도
            cls = classes[id]
            x1, y1, x2, y2 = map(int, np_boxes[idx])

            if conf < self.conf_threshold:
                continue

            bboxes.data.append(
                BoundingBox(
                    id=int(np_cls[idx]),
                    cls=str(cls),
                    conf=float(conf),
                    bbox=[x1, y1, x2, y2],
                )
            )

            label = f"{cls}, {conf:.2f}"
            cv2.rectangle(
                cv_image_array,
                (int(x1), int(y1)),
                (int(x2), int(y2)),
                self.color_dict[int(np_cls[idx])],
                2,
            )
            cv2.putText(
                img=cv_image_array,
                text=label,
                org=(int(x1), int(y1 - 10)),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.5,
                color=self.color_dict[int(np_cls[idx])],
                thickness=2,
            )

        segmented_image = self.bridge.cv2_to_imgmsg(cv_image_array, encoding="rgb8")
        return segmented_image, bboxes


def main(args=None):
    rclpy.init(args=args)

    node = RealTimeSegmentationNode()

    rclpy.spin(node=node)

    node.destroy_node()


if __name__ == "__main__":
    main()
