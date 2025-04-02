# Python Standard Libraries
import io
import json
import os
import socket
import struct
import sys
import time
import argparse
import array

# Third-Party Libraries
import cv2
import numpy as np
import tqdm
from cv_bridge import CvBridge
from scipy.spatial.transform import Rotation as R

# ROS2 Libraries
import rclpy
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import QoSProfile, qos_profile_system_default
from rclpy.time import Time

# ROS2 Message Types
from geometry_msgs.msg import *
from nav_msgs.msg import *
from sensor_msgs.msg import *
from std_msgs.msg import *
from visualization_msgs.msg import *
from custom_msgs.msg import (
    BoundingBox,
    BoundingBox3D,
    BoundingBox3DMultiArray,
    BoundingBoxMultiArray,
)
from custom_msgs.srv import MegaposeRequest

# ROS2 TF
from tf2_ros import *

# Custom Modules
from base_package.header import QuaternionAngle, Queue, PointCloudTransformer
from base_package.manager import ImageManager, Manager, ObjectManager
from fcn_network.fcn_manager import GridManager
from object_tracker.megapose_client import MegaPoseClient
from object_tracker.segmentation_manager import SegmentationManager
from ament_index_python.packages import get_package_share_directory


class ObjectPoseEstimator(Node):
    def __init__(self, *args, **kwargs):
        super().__init__("object_pose_estimator")

        self.pcd_subscirber = self.create_subscription(
            PointCloud2,
            "/camera/camera1/depth/color/points",
            callback=self.pointcloud_callback,
            qos_profile=qos_profile_system_default,
        )

        self._object_manager = ObjectManager(node=self, *args, **kwargs)
        self._grid_manager = GridManager(node=self, *args, **kwargs)

        # >>> ROS2 >>>
        self.megapose_srv = self.create_service(
            MegaposeRequest,
            "/megapose_request",
            self.megapose_request_callback,
            qos_profile=qos_profile_system_default,
        )
        # <<< ROS2 <<<

        self._pointcloud_msg: PointCloud2 = None
        self._grids, self._grids_dict = self._grid_manager.create_grid()

        # NO MAIN LOOP. This node is only runnning for megapose_request callbacks.

    def pointcloud_callback(self, msg: PointCloud2):
        self._pointcloud_msg = msg

    def megapose_request_callback(
        self, request: MegaposeRequest.Request, response: MegaposeRequest.Response
    ):
        # Initialize response message
        response_msg = BoundingBox3DMultiArray()

        # Slice PointCloud
        points = PointCloudTransformer.pointcloud2_to_numpy(
            msg=self._pointcloud_msg, rgb=False
        )
        transform_matrix = QuaternionAngle.transform_realsense_to_ros(np.eye(4))
        transformed_points = PointCloudTransformer.transform_pointcloud(
            points, transform_matrix
        )

        for grid in self._grids:
            grid: GridManager.Grid

            points_in_grid: np.ndarray = grid.slice_and_get_points(transformed_points)

            if points_in_grid.shape[0] < 10:
                continue

            center_point = np.mean(points_in_grid, axis=0)
            x_min, y_min, z_min = np.min(points_in_grid, axis=0)
            x_max, y_max, z_max = np.max(points_in_grid, axis=0)
            x_scale = np.clip(np.abs(x_max - x_min), 0.0, 0.05)
            y_scale = np.clip(np.abs(y_max - y_min), 0.0, 0.05)
            z_scale = np.clip(np.abs(z_max - z_min), 0.0, 0.1)

            bbox = BoundingBox3D(
                id=((ord(grid.row_id) - 64) * 10) + grid.col_id,
                cls=f"{grid.row_id}{grid.col_id}",
                pose=Pose(
                    position=Point(
                        x=float(center_point[0]),
                        y=float(center_point[1]),
                        z=float(center_point[2]),
                    ),
                    orientation=Quaternion(x=0.0, y=0.0, z=0.0, w=1.0),
                ),
                scale=Vector3(
                    x=float(y_scale) * 0.7, y=float(y_scale) * 0.7, z=float(z_scale)
                ),
            )

            print(f"Grid {grid.row_id}{grid.col_id}: {bbox.pose.position}")
            response_msg.data.append(bbox)

        print(len(response_msg.data))

        response.response = response_msg

        return response


def main():
    rclpy.init(args=None)

    parser = argparse.ArgumentParser(description="FCN Server Node")

    parser.add_argument(
        "--grid_data_file",
        type=str,
        required=False,
        default="grid_data.json",
        help="Path or file name of object bounds. If input is a file name, the file should be located in the 'resource' directory. Required",
    )

    args = parser.parse_args()
    kagrs = vars(args)

    node = ObjectPoseEstimator(**kagrs)

    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
