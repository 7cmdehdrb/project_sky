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
import time


class TriggerServer(Node):
    def __init__(self):
        super().__init__("trigger_server")
        self.get_logger().info("Trigger Server Node has been initialized.")

        self.data = 1002

        self.first_trigger_sub = self.create_subscription(
            UInt16,
            "/first_trigger",
            self.first_trigger_callback,
            qos_profile=qos_profile_system_default,
        )
        self.trigger_pub = self.create_publisher(
            UInt16, "/trigger", qos_profile=qos_profile_system_default
        )

    def first_trigger_callback(self, msg: UInt16):
        if msg.data == 1:
            time.sleep(1.0)
            self.trigger_pub.publish(UInt16(data=self.data))
            self.data += 1


def main():
    rclpy.init(args=None)

    node = TriggerServer()

    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
