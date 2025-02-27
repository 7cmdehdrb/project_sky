# ROS2
import rclpy
import rclpy.logging
from rclpy.node import Node
from rclpy.time import Time
from rclpy.duration import Duration
from rclpy.qos import QoSProfile, qos_profile_system_default

# Message
import rclpy.time
import rclpy.time
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
from scipy.spatial.transform import Rotation as R
from quaternion import QuaternionAngle


class ImageSlicer(Node):
    def __init__(self):
        super().__init__("image_slice_node")

        self.bridge = cv_bridge.CvBridge()

        self.image_sub = self.create_subscription(
            Image,
            f"/camera/camera1/color/image_raw",
            self.image_callback,
            qos_profile_system_default,
        )

        self.image_pub = self.create_publisher(
            Image, "/camera/camera1/color/image_sliced", qos_profile_system_default
        )

    def image_callback(self, image: Image):
        cv_image = self.bridge.imgmsg_to_cv2(image, desired_encoding="bgr8")

        # Resize Image Size Definition
        resize_shape = (640, 480)

        # Slice Image Size Definition
        new_width = 800
        new_height = int(new_width * (resize_shape[1] / resize_shape[0]))

        start_width = 200
        start_height = 20

        if cv_image.shape[0] < start_height + new_height:
            self.get_logger().warn("Height Size Over")

        print(start_height + new_height)

        # 720 1280
        sliced_cv_image = cv_image[
            start_height : start_height + new_height,
            start_width : start_width + new_width,
            :,
        ]

        resized_cv_image = cv2.resize(sliced_cv_image, dsize=resize_shape)

        msg = self.bridge.cv2_to_imgmsg(resized_cv_image)

        self.image_pub.publish(msg)


def main():
    rclpy.init(args=None)

    node = ImageSlicer()

    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
