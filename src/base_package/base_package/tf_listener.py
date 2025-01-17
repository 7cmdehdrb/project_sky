# ROS2
import rclpy
from rclpy.node import Node
from rclpy.time import Time
from rclpy.duration import Duration
from rclpy.qos import QoSProfile

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


class TFListener(Node):
    def __init__(self):
        super().__init__("tf_listener")
        self.tfBuffer = Buffer(node=self, cache_time=Duration(seconds=0.1))
        self.listener = TransformListener(self.tfBuffer, self)

        self.timer = self.create_timer(0.1, self.test)

    def test(self):
        if self.tfBuffer.can_transform(
            "camera1_link",
            "camera1_color_frame",
            self.get_clock().now(),
            Duration(seconds=0.1),
        ):
            transform = self.tfBuffer.lookup_transform(
                "camera1_link",
                "camera1_color_frame",
                self.get_clock().now(),
                Duration(seconds=0.1),
            )
            self.tfBuffer.transform()

            print(transform.transform)

        else:
            print("No transform")


def main():
    rclpy.init(args=None)

    node = TFListener()

    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
