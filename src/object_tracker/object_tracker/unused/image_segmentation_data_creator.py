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

# TF
from tf2_ros import *

# Python
import numpy as np
import cv2
import cv_bridge
from enum import Enum
import json


"""
8":"/World/Mug_2","9":"/World/Mug_3"
"""


class SegmentationType(Enum):
    BACKGROUND = 0
    UNLABELLED = 1
    alive = 2  # b1
    coca_cola = 3  # c1
    cyder = 4  # c2
    green_tea = 5  # b2
    yello_smoothie_transformed = 7  # b3
    yello_peach_transformed = 6  # c3
    Mug_2 = 8  # black
    Mug_3 = 9  # gray
    Mug_4 = 10  # yello
    Cup_1 = 11  # sky
    Cup_2 = 12  # white
    Cup_4 = 13  # blue


segmentation_type = {
    "BACKGROUND": 0,
    "UNLABELLED": 1,
    "alive": 2,
    "coca_cola": 3,
    "cyder": 4,
    "green_tea": 5,
    "yello_smoothie_transformed": 7,
    "yello_peach_transformed": 6,
    "Mug_2": 8,
    "Mug_3": 9,
    "Mug_4": 10,
    "Cup_1": 11,
    "Cup_2": 12,
    "Cup_4": 13,
}

roboflow_segmentation = {
    "alive": 0,
    "green_tea": 1,
    "yello_smoothie_transformed": 2,
    "coca_cola": 3,
    "cyder": 4,
    "yello_peach_transformed": 5,
    "Cup_1": 6,
    "Cup_2": 7,
    "Cup_4": 8,
    "Mug_2": 9,
    "Mug_3": 10,
    "Mug_4": 11,
}


class SegmentationDataCreator(Node):
    def __init__(self):
        super().__init__("segmentation_data_creator_node")
        self.bridge = cv_bridge.CvBridge()

        self.label_data = {
            2: 0,
            3: 3,
            4: 4,
            5: 1,
            6: 5,
            7: 2,
            8: 9,
            9: 10,
            10: 11,
            11: 6,
            12: 7,
            13: 8,
        }

        self.label_path = (
            "/home/min/7cmdehdrb/ros2_ws/src/object_tracker/resource/labels/train"
        )
        self.image_path = (
            "/home/min/7cmdehdrb/ros2_ws/src/object_tracker/resource/images/train"
        )
        self.txt = ""
        self.image = None

        self.instance_segmentation_sub = self.create_subscription(
            Image,
            "/instance_segmentation",
            self.instance_segmentation_callback,
            qos_profile_system_default,
        )

        self.image_sub = self.create_subscription(
            Image,
            "/rgb",
            self.image_callback,
            qos_profile_system_default,
        )

        self.trigger_sub = self.create_subscription(
            UInt16,
            "/trigger",
            self.trigger_callback,
            qos_profile_system_default,
        )

    def trigger_callback(self, msg: UInt16):
        if self.image is not None:
            image_path = f"{self.image_path}/{msg.data}.png"
            cv2.imwrite(image_path, self.image)
            self.get_logger().info(f"Saved image: {image_path}")

        f = open(f"{self.label_path}/{msg.data}.txt", "w")
        f.write(self.txt)
        f.close()
        self.get_logger().info(f"Saved label: {self.label_path}/{msg.data}.txt")

    def image_callback(self, msg):
        img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")

        resize = (640, 480)
        ratio = resize[0] / resize[1]
        height = 720

        img = img[
            0 : 0 + height,
            0 : 0 + int(height * ratio),
            :,
        ]

        resize_img = cv2.resize(img, resize)
        self.image = resize_img

    def instance_segmentation_callback(self, msg: Image):
        # 32SC1 -> numpy 배열로 변환
        img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")

        resize = (640, 480)
        ratio = resize[0] / resize[1]
        height = 720

        img = img[
            0 : 0 + height,
            0 : 0 + int(height * ratio),
        ]

        width = img.shape[1]
        height = img.shape[0]

        # raw_image = np.zeros(
        #     (height, width), dtype=np.uint8
        # )  # Grayscale 이미지 (0=Black)

        txt = ""
        for i in range(2, 14):
            mask = (img == i).astype(np.uint8)

            # 외곽선 검출
            contours, _ = cv2.findContours(
                mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            for contour in contours:
                if len(contour) < 3:  # 폴리곤이 되려면 최소 3점 이상 필요
                    continue

                # 좌표 정규화 (0~1 범위)
                normalized_contour = [
                    (x / width, y / height) for [x, y] in contour[:, 0, :]
                ]

                txt += f"{str(self.label_data[i])} "
                for x, y in normalized_contour:
                    txt += f"{x} {y} "
                txt = txt[:-1]
                txt += "\n"

            self.txt = txt


def main():
    rclpy.init(args=None)

    node = SegmentationDataCreator()

    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
