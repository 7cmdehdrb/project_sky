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


class FakeCollisionPublisher(Node):
    def __init__(self):
        super().__init__("fake_collision_publisher")

        self.fake_collision_pub = self.create_publisher(
            MarkerArray, "fake_collision", qos_profile=qos_profile_system_default
        )

        self.tfBuffer = Buffer(node=self, cache_time=Duration(seconds=1))
        self.listener = TransformListener(
            buffer=self.tfBuffer, node=self, qos=qos_profile_system_default
        )
        self.braodcaster = TransformBroadcaster(
            node=self, qos=qos_profile_system_default
        )

        self.tf_timer = self.create_timer(0.001, self.tf_callback)
        self.timer = self.create_timer(0.2, self.timer_callback)

    def timer_callback(self):
        marker_array = MarkerArray()

        header = Header(frame_id="camera1_link", stamp=self.get_clock().now().to_msg())

        coca_cola = Marker(
            header=header,
            ns="coca_cola",
            id=0,
            type=Marker.CUBE,
            action=Marker.ADD,
            pose=Pose(
                position=Point(x=0.7, y=-0.10, z=-0.11),
                orientation=Quaternion(x=0.0, y=0.0, z=0.0, w=1.0),
            ),
            scale=Vector3(x=0.08, y=0.08, z=0.12),
            color=ColorRGBA(r=1.0, g=0.0, b=0.0, a=0.7),
        )
        marker_array.markers.append(coca_cola)

        gray_cup = Marker(
            header=header,
            ns="gray_cup",
            id=1,
            type=Marker.CUBE,
            action=Marker.ADD,
            pose=Pose(
                position=Point(x=0.7, y=0.10, z=-0.11),
                orientation=Quaternion(x=0.0, y=0.0, z=0.0, w=1.0),
            ),
            scale=Vector3(x=0.08, y=0.08, z=0.12),
            color=ColorRGBA(r=1.0, g=0.0, b=0.0, a=0.7),
        )
        marker_array.markers.append(gray_cup)

        blue_cup = Marker(
            header=header,
            ns="blue_cup",
            id=2,
            type=Marker.CUBE,
            action=Marker.ADD,
            pose=Pose(
                position=Point(x=0.7, y=-0.34, z=-0.11),
                orientation=Quaternion(x=0.0, y=0.0, z=0.0, w=1.0),
            ),
            scale=Vector3(x=0.08, y=0.08, z=0.12),
            color=ColorRGBA(r=1.0, g=0.0, b=0.0, a=0.7),
        )
        marker_array.markers.append(blue_cup)

        alive = Marker(
            header=header,
            ns="alive",
            id=3,
            type=Marker.CUBE,
            action=Marker.ADD,
            pose=Pose(
                position=Point(x=0.85, y=-0.34, z=-0.08),
                orientation=Quaternion(x=0.0, y=0.0, z=0.0, w=1.0),
            ),
            scale=Vector3(x=0.08, y=0.08, z=0.18),
            color=ColorRGBA(r=1.0, g=0.0, b=0.0, a=0.7),
        )
        marker_array.markers.append(alive)

        yello_mug = Marker(
            header=header,
            ns="yello_mug",
            id=4,
            type=Marker.CUBE,
            action=Marker.ADD,
            pose=Pose(
                position=Point(x=0.85, y=0.1, z=-0.11),
                orientation=Quaternion(x=0.0, y=0.0, z=0.0, w=1.0),
            ),
            scale=Vector3(x=0.1, y=0.1, z=0.12),
            color=ColorRGBA(r=1.0, g=0.0, b=0.0, a=0.7),
        )
        marker_array.markers.append(yello_mug)

        green_tea = Marker(
            header=header,
            ns="green_tea",
            id=5,
            type=Marker.CUBE,
            action=Marker.ADD,
            pose=Pose(
                position=Point(x=0.95, y=-0.34, z=-0.07),
                orientation=Quaternion(x=0.0, y=0.0, z=0.0, w=1.0),
            ),
            scale=Vector3(x=0.08, y=0.08, z=0.2),
            color=ColorRGBA(r=1.0, g=0.0, b=0.0, a=0.7),
        )
        marker_array.markers.append(green_tea)

        yello_smoothie = Marker(
            header=header,
            ns="yello_smoothie",
            id=6,
            type=Marker.CUBE,
            action=Marker.ADD,
            pose=Pose(
                position=Point(x=0.95, y=-0.20, z=-0.07),
                orientation=Quaternion(x=0.0, y=0.0, z=0.0, w=1.0),
            ),
            scale=Vector3(x=0.08, y=0.08, z=0.2),
            color=ColorRGBA(r=1.0, g=0.0, b=0.0, a=0.7),
        )
        marker_array.markers.append(yello_smoothie)

        # 0.08, 0.12

        self.fake_collision_pub.publish(marker_array)

    def tf_callback(self):
        tf_msg = TransformStamped(
            header=Header(frame_id="base_link", stamp=self.get_clock().now().to_msg()),
            child_frame_id="camera1_link",
            transform=Transform(
                translation=Vector3(x=0.0, y=0.0, z=0.0),
                rotation=Quaternion(x=0.0, y=0.0, z=0.0, w=1.0),
            ),
        )

        # print(tf_msg)

        self.braodcaster.sendTransform(tf_msg)


def main():
    rclpy.init(args=None)

    node = FakeCollisionPublisher()

    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
