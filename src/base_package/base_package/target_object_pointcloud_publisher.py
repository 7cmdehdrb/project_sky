# ROS2
import rclpy
from rclpy.node import Node
import rclpy.node
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
import open3d as o3d
from enum import Enum
from header import QuaternionAngle, PointCloudTransformer
import os


class TargetObject(Enum):
    alive = 0
    coca_cola = 1
    cyder = 2
    green_tea = 3
    yello_peach = 4
    yello_smoothie = 5


class Target_PCD_Publisher(Node):
    class Database(object):
        def __init__(self, node: Node):
            self.node = node
            self.workspace_path = "/"

            colcon_prefix_path = os.environ.get("COLCON_PREFIX_PATH", "")

            if colcon_prefix_path:
                install_path = colcon_prefix_path.split(":")[0]
                self.workspace_path = install_path.split("/install")[0]
            else:
                raise Exception("COLCON_PREFIX_PATH is not set. Source your workspace.")

            self.db = self.create_database()

        def get_resource_path(self):
            resource_path = "src/base_package/base_package/resource"
            resource_full_path = os.path.join(self.workspace_path, resource_path)

            return resource_full_path

        def get_pcd_file_names(self):
            resource_full_path = self.get_resource_path()
            pcd_file_names = os.listdir(resource_full_path)
            pcd_file_names = [
                file_name for file_name in pcd_file_names if file_name.endswith(".ply")
            ]

            return pcd_file_names

        def create_database(self):
            root_dir = self.get_resource_path()
            pcd_file_names = self.get_pcd_file_names()

            database = {}

            for pcd_file_name in pcd_file_names:
                target_object_name = pcd_file_name.replace(".ply", "")
                target_object_id = TargetObject[target_object_name].value

                pcd_file_path = os.path.join(root_dir, pcd_file_name)

                database[target_object_id] = pcd_file_path

            return database

    def __init__(self):
        super().__init__("target_pcd_publisher_node")

        # Database
        self.database = self.Database(self)

        # Resize matrix. mm -> m
        self.resize_matrix = np.array(
            [
                [0.001, 0.0, 0.0, 0.0],
                [0.0, 0.001, 0.0, 0.0],
                [0.0, 0.0, 0.001, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )

        self.target_object_id = -1
        self.pointcloud = None

        # ROS Publisher & Subscriber
        self.target_object_sub = self.create_subscription(
            UInt8,
            "/target_pcd_publisher_node/target_object_id",
            self.callback_target_object_id,
            qos_profile=qos_profile_system_default,
        )
        self.target_pointcloud_pub = self.create_publisher(
            PointCloud2,
            "/target_pcd_publisher_node/target_object_pointcloud",
            qos_profile=qos_profile_system_default,
        )

        self.timer = self.create_timer(0.1, self.run)

    def callback_target_object_id(self, msg: UInt8):
        target_object_id = msg.data

        if self.target_object_id == target_object_id:
            # Skip if the target object ID is the same
            return False

        try:
            pcd_file_path = self.database.db[target_object_id]
        except KeyError:
            self.get_logger().warn(f"Invalid target object ID: {target_object_id}")
            return False
        except Exception as ex:
            self.get_logger().error(ex)
            return False

        points = np.asarray(o3d.io.read_point_cloud(pcd_file_path).points)
        resized_points = PointCloudTransformer.transform_pointcloud(
            points, self.resize_matrix
        )

        self.pointcloud = PointCloudTransformer.numpy_to_pointcloud2(
            resized_points,
            frame_id="base_link",
            stamp=self.get_clock().now().to_msg(),
            rgb=False,
        )

        return True

    def run(self):
        if self.pointcloud is None:
            return False

        self.pointcloud.header = Header(
            frame_id="base_link",
            stamp=self.get_clock().now().to_msg(),
        )

        self.target_pointcloud_pub.publish(self.pointcloud)


def main():
    rclpy.init(args=None)

    node = Target_PCD_Publisher()

    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
