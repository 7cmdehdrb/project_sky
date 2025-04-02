# Python
import argparse
import json
import os
import sys
from PIL import Image as PILImage
from PIL import ImageEnhance

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
from base_package.manager import ImageManager, Manager, ObjectManager


class YoloManager(Manager):
    def __init__(self, node: Node, *arg, **kwargs):
        super().__init__(node, *arg, **kwargs)

        # >>> Load Files >>>
        tracker_package_path = get_package_share_directory("object_tracker")

        resource_path = os.path.join(
            tracker_package_path, "../ament_index/resource_index/packages"
        )

        model_path = kwargs["model_file"]
        if not os.path.isfile(model_path):
            model_path = os.path.join(resource_path, model_path)
        # <<< Load Files <<<

        # Load YOLO v11 Model
        self._model = YOLO(kwargs["model_file"], verbose=False)
        self._model.eval()

    def predict(self, image: PILImage):
        return self._model(image, verbose=False)


class RealTimeSegmentationNode(Node):
    def __init__(
        self,
        *arg,
        **kwargs,
    ):
        super().__init__("real_time_segmentation_node")

        # >>> Managers >>>
        self._yolo_manager = YoloManager(self, *arg, **kwargs)
        self._object_manager = ObjectManager(self, *arg, **kwargs)
        subscribed_topics = [
            {
                "topic_name": "/camera/camera1/color/image_raw",
                "callback": self.image_callback,
            },
        ]
        published_topics = [
            {"topic_name": self.get_name() + "/segmented_image"},
        ]
        self._image_manager = ImageManager(
            self,
            subscribed_topics=subscribed_topics,
            published_topics=published_topics,
            *arg,
            **kwargs,
        )
        # <<< Managers <<<

        # >>> ROS2 >>>
        self.segmented_bbox_publisher = self.create_publisher(
            BoundingBoxMultiArray,
            self.get_name() + "/segmented_bbox",
            qos_profile=qos_profile_system_default,
        )
        # <<< ROS2 <<<

        # >>> Data >>>
        self._camera_image: Image = None
        # <<< Data <<<

        # >>> Load Files >>>
        tracker_package_path = get_package_share_directory("object_tracker")

        resource_path = os.path.join(
            tracker_package_path, "../ament_index/resource_index/packages"
        )

        model_path = kwargs["obj_bounds_file"]
        if not os.path.isfile(model_path):
            model_path = os.path.join(resource_path, model_path)

        with open(
            os.path.join(resource_path, "obj_bounds.json"),
            "r",
        ) as f:
            self._obj_bounds = json.load(f)

        # <<< Load Files <<<

        # >>> Parameters >>>
        self._conf_threshold = float(kwargs["conf_threshold"])
        # <<< Parameters <<<

    def image_callback(self, msg: Image):
        self._camera_image = msg

        img_msg, bbox_msg = self.do_segmentation(msg=msg)

        self._image_manager.publish(
            topic_name=self.get_name() + "/segmented_image", msg=img_msg
        )
        self.segmented_bbox_publisher.publish(bbox_msg)

    def do_segmentation(self, msg: Image):
        # Load Image
        np_image = self._image_manager.decode_message(msg, desired_encoding="rgb8")
        np_image = self._image_manager.crop_image(np_image)

        pil_image = PILImage.fromarray(np_image)

        # YOLO 세그멘테이션 수행
        result: Results = self._yolo_manager.predict(pil_image)[0]
        boxes: Boxes = result.boxes
        classes: dict = result.names

        np_boxes = boxes.xyxy.cpu().numpy()
        np_confs = boxes.conf.cpu().numpy()
        np_cls = boxes.cls.cpu().numpy()

        # 바운딩 박스 그리기
        bboxes = BoundingBoxMultiArray()

        for idx in range(len(boxes)):
            id = int(np_cls[idx])  # 클래스 ID. 0, 1, 2, ...
            cls = classes[id]  # 클래스 이름. "cup_1" 등
            conf = np_confs[idx]  # 신뢰도

            x1, y1, x2, y2 = map(int, np_boxes[idx])

            # >>> STEP 1. 신뢰도 확인
            if conf < self._conf_threshold:
                continue

            # >>> STEP 2. 비율 확인
            detected_ratio = abs(x1 - x2) / abs(y1 - y2)

            desired_data = self._obj_bounds[
                self._object_manager.names[cls]
            ]  # "cup_1" -> "cup_sky"

            numerator = (float(desired_data["x"]) + float(desired_data["z"])) / 2.0
            denominator = float(desired_data["y"])

            desired_ratio = numerator / denominator
            difference_ratio = (
                detected_ratio / desired_ratio
                if detected_ratio > desired_ratio
                else desired_ratio / detected_ratio
            )

            # >>> STEP 3. 바운딩 박스 추가
            bboxes.data.append(
                BoundingBox(
                    id=int(np_cls[idx]),
                    cls=str(cls),
                    conf=float(conf), # if difference_ratio > 1.2 else 0.0,
                    bbox=[x1, y1, x2, y2],
                )
            )

            # >>> STEP 4. 바운딩 박스 그리기
            label = f"{cls}, {conf:.2f}"
            cv2.rectangle(
                np_image,
                (int(x1), int(y1)),
                (int(x2), int(y2)),
                (
                    self._object_manager.color_dict[int(np_cls[idx])]
                    if difference_ratio > 1.2
                    else (0, 0, 0)
                ),
                2,
            )
            cv2.putText(
                img=np_image,
                text=label,
                org=(int(x1), int(y1 - 10)),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.5,
                color=(
                    self._object_manager.color_dict[int(np_cls[idx])]
                    if difference_ratio > 1.2
                    else (0, 0, 0)
                ),
                thickness=2,
            )

        segmented_image = self._image_manager.encode_message(np_image, encoding="rgb8")
        return segmented_image, bboxes


def main(args=None):
    rclpy.init(args=args)

    parser = argparse.ArgumentParser(description="FCN Server Node")
    parser.add_argument(
        "--model_file",
        type=str,
        required=True,
        default="best_model.pth",
        help="Path or file name of the model. If input is a file name, the file should be located in the 'resource' directory. Required",
    )
    parser.add_argument(
        "--obj_bounds_file",
        type=str,
        required=False,
        default="obj_bounds.json",
        help="Path or file name of object bounds. If input is a file name, the file should be located in the 'resource' directory. Required",
    )
    parser.add_argument(
        "--conf_threshold",
        type=float,
        required=False,
        default=0.7,
        help="Confidence threshold for object detection (default: 0.7)",
    )

    args = parser.parse_args()
    kagrs = vars(args)

    node = RealTimeSegmentationNode(**kagrs)

    rclpy.spin(node=node)

    node.destroy_node()


if __name__ == "__main__":
    main()
