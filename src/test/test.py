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
from builtin_interfaces.msg import Duration as BuiltinDuration

# TF
from tf2_ros import *

# Python
import sys
import os
import numpy as np
from enum import Enum
import time
import threading

from base_package.transform_manager import TransformManager


class MyNode(Node):
    def __init__(self):
        super().__init__("my_node")

        self._tf_manager = TransformManager(self)

        self.p1 = PoseStamped(
            header=Header(
                frame_id="helios_camera", stamp=self.get_clock().now().to_msg()
            ),
            pose=Pose(
                position=Point(x=0.0, y=0.0, z=0.0),
                orientation=Quaternion(x=0.0, y=0.0, z=0.0, w=1.0),
            ),
        )

        self._timer = self.create_timer(0.1, self.tf_callback)

    def tf_callback(self):

        self.p1.header.stamp = self.get_clock().now().to_msg()

        p2 = self._tf_manager.transform_pose(
            pose=self.p1,
            target_frame="base_link",
            source_frame=self.p1.header.frame_id,
        )

        self.get_logger().info(
            f"Transformed Pose: -> {p2.pose.position.x:.3f}, {p2.pose.position.y:.3f}, {p2.pose.position.z:.3f}"
        )


def main(args=None):
    rclpy.init(args=args)

    node = MyNode()

    th = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    th.start()

    hz = 30.0
    r = node.create_rate(hz)
    try:
        while rclpy.ok():
            r.sleep()
    except KeyboardInterrupt:
        node.get_logger().info("KeyboardInterrupt received, shutting down.")
    except Exception as e:
        node.get_logger().error(f"Exception in main loop: {e}")
    finally:
        th.join(timeout=1.0)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
