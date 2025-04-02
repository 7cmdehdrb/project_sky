# ROS2
import rclpy
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.publisher import Publisher
from rclpy.qos import QoSProfile, qos_profile_system_default
from rclpy.time import Time

# ROS2 Messages
from geometry_msgs.msg import *
from nav_msgs.msg import *
from sensor_msgs.msg import *
from std_msgs.msg import *
from visualization_msgs.msg import *
from custom_msgs.msg import *
from tf2_geometry_msgs.tf2_geometry_msgs import PoseStamped as TF2PoseStamped
from builtin_interfaces.msg import Duration as BuiltinDuration

# TF
from tf2_ros import *

# Python Libraries
import cv2
import numpy as np
import cv_bridge
from PIL import Image as PILImage
from PIL import ImageEnhance


class Manager(object):
    def __init__(self, node: Node, *args, **kwargs):
        self._node = node


class ImageManager(Manager):
    def __init__(
        self,
        node: Node,
        subscribed_topics: list = [],
        published_topics: list = [],
        *args,
        **kwargs,
    ):
        """
        subscribed_topics: list
            [
                {
                    "topic_name": str,
                    "callback": callable
                }
            ]
        published_topics: list
            [
                {
                    "topic_name": str
                }
            ]
        """
        super().__init__(node, *args, **kwargs)

        self._bridge = cv_bridge.CvBridge()

        self._subscribers = [
            {
                "topic_name": sub["topic_name"],
                "subscriber": self._node.create_subscription(
                    Image,
                    sub["topic_name"],
                    sub["callback"],
                    qos_profile=qos_profile_system_default,
                ),
            }
            for sub in subscribed_topics
        ]
        self._publishers = [
            {
                "topic_name": pub["topic_name"],
                "publisher": self._node.create_publisher(
                    Image,
                    pub["topic_name"],
                    qos_profile=qos_profile_system_default,
                ),
            }
            for pub in published_topics
        ]

    def get_publisher(self, topic_name: str) -> Publisher:
        for pub in self._publishers:
            if pub["topic_name"] == topic_name:
                return pub["publisher"]

    def get_subscriber(self, topic_name: str):
        for sub in self._subscribers:
            if sub["topic_name"] == topic_name:
                return sub["subscriber"]

    def encode_message(self, image: np.ndarray, encoding: str = "bgr8"):
        return self._bridge.cv2_to_imgmsg(image, encoding=encoding)

    def decode_message(self, image_msg: Image, desired_encoding: str = "bgr8"):
        return self._bridge.imgmsg_to_cv2(image_msg, desired_encoding=desired_encoding)

    def publish(self, topic_name: str, msg: Image):
        self.get_publisher(topic_name).publish(msg)

    @staticmethod
    def crop_image(img: np.ndarray):
        """
        This function crops the image to 640x480. If the image is already 640x480, it returns the image as is.
        """
        h, w = img.shape[:2]
        th, tw = 640, 480

        # 이미지 크기가 640x480이면 그대로 반환
        if h == th and w == tw:
            return img

        # Crop Image
        h, w = img.shape[:2]
        crop_w, crop_h = 640, 480
        start_x = int((w - crop_w) // 2.05)
        start_y = int((h - crop_h) // 2.7)

        # 크롭 범위 보정
        start_x = max(0, min(start_x, w - crop_w))
        start_y = max(0, min(start_y, h - crop_h))

        # 이미지 크롭 (640x480)
        cropped_img = img[start_y : start_y + crop_h, start_x : start_x + crop_w]

        return cropped_img

    @staticmethod
    def adjust_sim_image(img: np.array, stats: dict):
        """
        This function adjusts the brightness, saturation, and contrast of the image based on the statistics of the image.
        The statistics have THESE keys:
            avg_rgb: float[3]
            avg_saturation: float
            avg_brightness: float
            avg_contrast: float
        """
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


# Dictionary class
class ObjectManager(Manager):
    def __init__(self, node: Node, *args, **kwargs):
        super().__init__(node, *args, **kwargs)
        
        self.names = {
            "can_1": "coca_cola",
            "can_2": "cyder",
            "can_3": "yello_peach",
            "cup_1": "cup_sky",
            "cup_2": "cup_white",
            "cup_3": "cup_blue",
            "mug_1": "mug_black",
            "mug_2": "mug_gray",
            "mug_3": "mug_yello",
            "bottle_1": "alive",
            "bottle_2": "green_tea",
            "bottle_3": "yello_smoothie",
        }

        self.classes = {v: k for k, v in self.names.items()}

        self.indexs = {k: i for i, k in enumerate(self.names.keys())}

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


class TransformManager(Manager):
    def __init__(self, node: Node, *args, **kwargs):
        super().__init__(node, *args, **kwargs)

        self._tf_buffer = Buffer(node=self._node, cache_time=Duration(seconds=2))
        self._tf_listener = TransformListener(node=self._node, buffer=self._tf_buffer)
        self._tf_broadcaster = TransformBroadcaster(self._node)

    def check_transform_valid(self, target_frame: str, source_frame: str):
        try:
            valid = self._tf_buffer.can_transform(
                target_frame,
                source_frame,
                self._node.get_clock().now().to_msg(),
                timeout=Duration(seconds=0.1),
            )
            
            if not valid:
                raise Exception("Transform is not valid")
            
            return valid
        except Exception as e:
            self._node.get_logger().warn(
                f"Cannot Lookup Transform Between {target_frame} and {source_frame}"
            )
            self._node.get_logger().warn(e)
            return False

    def transform_bbox_3d(
        self,
        bbox_3d: BoundingBox3DMultiArray,
        target_frame: str = "world",
        source_frame: str = "camera1_link",
    ) -> BoundingBox3DMultiArray:
        """
        Transform the bounding box in camera frame to the world frame.
        """

        if self.check_transform_valid(target_frame, source_frame):
            # Initialize the transformed bounding box
            transformed_bbox = BoundingBox3DMultiArray()

            try:
                # Transform the bounding box
                for bbox in bbox_3d.data:
                    bbox: BoundingBox3D

                    source_pose = TF2PoseStamped(
                        header=Header(
                            stamp=self._node.get_clock().now().to_msg(),
                            frame_id=source_frame,
                        ),
                        pose=bbox.pose,
                    )

                    target_pose = self._tf_buffer.transform(
                        object_stamped=source_pose,
                        target_frame=target_frame,
                        timeout=Duration(seconds=1),
                    )

                    bbox.pose = target_pose.pose
                    transformed_bbox.data.append(bbox)

                return transformed_bbox
            except Exception as e:
                self._node.get_logger().warn(
                    f"Cannot Transform BoundingBox3DMultiArray from {source_frame} to {target_frame}"
                )
                self._node.get_logger().warn(e)
                return None

        return None
