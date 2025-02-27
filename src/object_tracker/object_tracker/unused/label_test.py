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


class LabelTest(Node):
    def __init__(self):
        super().__init__("label_test")
        self.get_logger().info("Label Test Node has been initialized.")

        self.label_pub = self.create_subscription(
            String,
            "/semantic_labels",
            self.label_callback,
            qos_profile=qos_profile_system_default,
        )

    def label_callback(self, msg: String):
        print(f"Received: {msg.data}")


def main():
    rclpy.init(args=None)

    node = LabelTest()

    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
