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

        # self.pcd_subscirber = self.create_subscription(
        #     PointCloud2,
        #     "/camera/camera1/depth/color/points",
        #     callback=self.pointcloud_callback,
        #     qos_profile=qos_profile_system_default,
        # )

        self._object_manager = ObjectManager(node=self, *args, **kwargs)
        self._grid_manager = GridManager(node=self, *args, **kwargs)

        # >>> ROS2 >>>
        self.megapose_srv = self.create_service(
            MegaposeRequest,
            "/megapose_request",
            self.megapose_request_callback,
            qos_profile=qos_profile_system_default,
        )
        self._target_ids = ["A0", "A1", "A2", "A3", "B1", "B3", "C0", "C1", "C2", "C3"]
        # <<< ROS2 <<<

        # NO MAIN LOOP. This node is only runnning for megapose_request callbacks.

    def get_fake_objects(self, ids: List[str]):
        """
        Get fake objects for testing.
        """
        data = BoundingBox3DMultiArray()

        for id in ids:
            for grid in self._grid_manager._grids:
                grid: GridManager.Grid

                if id == f"{grid.row}{grid.col}":
                    data.data.append(
                        BoundingBox3D(
                            id=(ord(grid.row) - ord("A") + 1) * 10 + grid.col,
                            pose=Pose(
                                position=grid.center_coord,
                                orientation=Quaternion(x=0.0, y=0.0, z=0.0, w=1.0),
                            ),
                            scale=Vector3(x=0.05, y=0.05, z=0.1),
                        )
                    )
                    break

        return data

    def megapose_request_callback(
        self, request: MegaposeRequest.Request, response: MegaposeRequest.Response
    ):
        # Initialize response message
        bbox_3d = self.get_fake_objects(self._target_ids)

        response.response = bbox_3d

        return response


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

    node = ObjectPoseEstimator(**kagrs)

    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
