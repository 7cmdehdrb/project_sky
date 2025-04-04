# Python
import os
import sys
import json
import numpy as np
import argparse
import array

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
from custom_msgs.srv import FCNOccupiedRequest

# TF
from tf2_ros import *

# Custom
from base_package.header import PointCloudTransformer, QuaternionAngle
from base_package.manager import ObjectManager
from fcn_network.fcn_manager import GridManager


class FakePointCloudPublisher(Node):
    def __init__(self, *args, **kwargs):
        super().__init__("fake_pointcloud_publisher_node")

        # >>> Grid Manager >>>
        self._grid_manager = GridManager(self, *args, **kwargs)
        self._grid_data = self._grid_manager.get_grid_data()

        self._target_ids = ["A2", "B0", "C0", "C1", "C2", "C3"]
        # A1 B1

        self._pub = self.create_publisher(
            PointCloud2,
            "/camera/camera1/depth/color/points",
            qos_profile_system_default,
        )

        self._pcd = self.get_pcd()

        self._timer = self.create_timer(
            0.1,
            self.publish,
        )

    def publish(self):
        # Publish the fake point cloud
        pcd = self.get_pcd()

        self._pub.publish(pcd)
        self.get_logger().info("Fake point cloud published.")

    def get_pcd(self):
        data = np.empty((0, 3), dtype=np.float32)

        for id in self._target_ids:
            row = id[0]
            col = int(id[1])

            grid: GridManager.Grid = self._grid_manager.get_grid(row=row, col=col)
            center_point: Point = grid.center_coord

            for _ in range(500):
                # Create a fake point cloud
                point = np.array(
                    [
                        center_point.x + np.random.uniform(-0.05, 0.05),
                        center_point.y + np.random.uniform(-0.05, 0.05),
                        center_point.z + np.random.uniform(-0.1, 0.1),
                    ],
                    dtype=np.float32,
                )
                data = np.vstack((data, point))

        pcd = PointCloudTransformer.numpy_to_pointcloud2(
            points=data,
            frame_id="camera1_link",
            stamp=self.get_clock().now().to_msg(),
            rgb=False,
        )

        return pcd


def main():
    rclpy.init(args=None)

    from rclpy.utilities import remove_ros_args

    # Remove ROS2 arguments
    argv = remove_ros_args(sys.argv)

    parser = argparse.ArgumentParser(description="FCN Server Node")

    parser.add_argument(
        "--obj_bounds_file",
        type=str,
        default=True,
        help="Path or file name of object bounds. If input is a file name, the file should be located in the 'resource' directory. Required",
    )

    parser.add_argument(
        "--grid_data_file",
        type=str,
        required=True,
        default="grid_data.json",
        help="Path or file name of object bounds. If input is a file name, the file should be located in the 'resource' directory. Required",
    )

    args = parser.parse_args(argv[1:])
    kagrs = vars(args)

    fake_pointcloud_publisher = FakePointCloudPublisher(**kagrs)

    try:
        rclpy.spin(fake_pointcloud_publisher)
    except KeyboardInterrupt:
        fake_pointcloud_publisher.get_logger().info("Keyboard interrupt")

    fake_pointcloud_publisher.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
