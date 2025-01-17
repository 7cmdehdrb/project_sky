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
from struct import pack
import numpy as np
import sensor_msgs_py.point_cloud2 as pc2
import open3d as o3d


class PCDSubscriber(Node):
    class PointCloudSubscriber:
        def __init__(self, node: Node, camera_id: str):
            self.node = node

            self.camera_id = camera_id
            self.pointcloud_topic = f"/camera/{camera_id}/depth/color/points"

            self.pointcloud_sub = self.node.create_subscription(
                PointCloud2,
                self.pointcloud_topic,
                self.callback,
                qos_profile_system_default,
            )

            self.transform_matrix_sub = self.node.create_subscription(
                Float32MultiArray,
                f"{self.camera_id}_transform_matrix",
                self.transform_matrix_callback,
                qos_profile_system_default,
            )

            self.msg = PointCloud2()
            self.transform_matrix = np.eye(4)

        def callback(self, msg: PointCloud2):
            self.msg = msg

        def transform_matrix_callback(self, msg: Float32MultiArray):
            data = msg.data

            # Float32MultiArray 데이터를 4x4 NumPy 배열로 변환
            transform_matrix = np.array(data).reshape(4, 4)

            axis_rotate_matrix = np.array(
                [
                    [0, 0, 1, 0],  # Z -> X
                    [-1, 0, 0, 0],  # X -> -Y
                    [0, -1, 0, 0],  # Y -> -Z
                    [0, 0, 0, 1],  # Homogeneous coordinate
                ]
            )

            self.transform_matrix = transform_matrix @ axis_rotate_matrix

    def __init__(self):
        super().__init__("pcd_subscriber_node")

        self.pcd = o3d.geometry.PointCloud()

        self.get_logger().info("Waiting for 1 second before starting...")
        self.get_clock().sleep_for(Duration(seconds=1))

        self.camera1 = self.PointCloudSubscriber(self, "camera1")
        self.camera2 = self.PointCloudSubscriber(self, "camera2")
        self.camera3 = self.PointCloudSubscriber(self, "camera3")
        # self.camera4 = self.PointCloudSubscriber(self, "camera4")

        self.cameras = [
            self.camera1,
            self.camera2,
            self.camera3,
            # self.camera4,
            # None
        ]

        self.pointcloud_publisher = self.create_publisher(
            PointCloud2, "/combined_pointcloud", qos_profile_system_default
        )

        hz = 10
        self.loop = self.create_timer(float(1 / hz), self.publish_pointcloud)

        self.current_time = self.get_clock().now()

    def publish_pointcloud(self):
        msg = self.post_process_pointcloud()

        if msg is not None:
            self.pointcloud_publisher.publish(msg)

    def post_process_pointcloud(self):
        # Calculate the time difference between the current and previous callback

        current_time = self.get_clock().now()
        dt = (current_time - self.current_time).nanoseconds / 1e9

        print(f"dt: {dt}, hz: {1/dt}")

        self.current_time = current_time

        rgb = True

        combined_points = np.empty((0, 6)) if rgb else np.empty((0, 3))

        for camera in self.cameras:
            camera: PCDSubscriber.PointCloudSubscriber

            if len(camera.msg.data) == 0:
                self.get_logger().warn(f"Empty point cloud from {camera.camera_id}")
                continue

            camera_points = self.pointcloud2_to_numpy(camera.msg, rgb=rgb)

            # Outlier removal. Realsense Axis
            camera_points = self.ROI_Color_filter(
                points=camera_points,
                ROI=True,
                x_range=(-3.0, 3.0),  # y
                y_range=(-3.0, 3.0),  # z
                z_range=(-0.1, 3.0),  # x
                rgb=False if rgb else False,
            )[::2]

            camera_transform_matrix = camera.transform_matrix

            transformed_camera1_points = self.transform_pointcloud(
                points=camera_points, transform_matrix=camera_transform_matrix
            )

            combined_points = np.concatenate(
                [
                    combined_points,
                    transformed_camera1_points,
                ],
                axis=0,
            )

        # Ros Axis
        ROI_filtered_points = self.ROI_Color_filter(
            points=combined_points,
            ROI=False,  # ROI Filter
            x_range=(-2.0, 2.0),
            y_range=(-2.0, 2.0),
            z_range=(-2.0, 2.0),
            rgb=False if rgb else False,  # Filter only Red
            r_range=(150, 255),
            g_range=(0, 100),
            b_range=(0, 100),
        )

        try:

            filtered_msg = self.numpy_to_pointcloud2(
                ROI_filtered_points,
                frame_id="base_link",
                stamp=self.get_clock().now().to_msg(),
                rgb=rgb,
            )

        except Exception as e:
            self.get_logger().error(f"Exception: {e}")
            return None

        return filtered_msg

    @staticmethod
    def numpy_to_pointcloud2(points: np.array, frame_id, stamp, rgb=True):
        # Create the header
        header = Header(frame_id=frame_id, stamp=stamp)
        fields = [
            PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
        ]
        width = points.shape[0]

        if rgb:
            fields += [
                PointField(name="rgb", offset=12, datatype=PointField.FLOAT32, count=1)
            ]

            # RGB 값을 uint32로 병합
            rgb_uint32 = (
                (points[:, 3].astype(np.uint32) << 16)
                | (points[:, 4].astype(np.uint32) << 8)
                | points[:, 5].astype(np.uint32)
            )

            # uint32 데이터를 float32로 변환
            rgb_values = rgb_uint32.view(np.float32)

            # Create the structured array with fields x, y, z, rgb
            structured_array = np.zeros(
                width,
                dtype=[
                    ("x", np.float32),
                    ("y", np.float32),
                    ("z", np.float32),
                    ("rgb", np.float32),
                ],
            )
            structured_array["x"] = points[:, 0]
            structured_array["y"] = points[:, 1]
            structured_array["z"] = points[:, 2]
            structured_array["rgb"] = rgb_values

            # Convert the structured array to binary data
            data = structured_array.tobytes()

            # Create the PointCloud2 message
            cloud = PointCloud2(
                header=header,
                fields=fields,
                height=1,
                width=width,
                is_dense=True,
                is_bigendian=False,
                point_step=16,
                row_step=16 * width,
                data=data,
            )

        else:
            # Create the structured array with fields x, y, z, rgb
            structured_array = np.zeros(
                width,
                dtype=[
                    ("x", np.float32),
                    ("y", np.float32),
                    ("z", np.float32),
                ],
            )
            structured_array["x"] = points[:, 0]
            structured_array["y"] = points[:, 1]
            structured_array["z"] = points[:, 2]

            # Convert the structured array to binary data
            data = structured_array.tobytes()

            # Create the PointCloud2 message
            cloud = PointCloud2(
                header=header,
                fields=fields,
                height=1,
                width=width,
                is_dense=True,
                is_bigendian=False,
                point_step=12,
                row_step=12 * width,
                data=data,
            )

        return cloud

    @staticmethod
    def transform_pointcloud(points: np.array, transform_matrix: np.array) -> np.array:
        """
        Apply a transformation matrix to a point cloud.
        :param points: numpy array of shape (N, 6) with x, y, z, r, g, b
        :param transform_matrix: 4x4 numpy transformation matrix
        :return: transformed numpy array of shape (N, 6)
        """
        # Extract x, y, z coordinates
        coords = points[:, :3]  # Shape (N, 3)

        # Add a column of ones to create homogeneous coordinates
        ones = np.ones((coords.shape[0], 1))
        hom_coords = np.hstack([coords, ones])  # Shape (N, 4)

        # Apply the transformation matrix
        transformed_hom_coords = (transform_matrix @ hom_coords.T).T  # Shape (N, 4)

        # Replace the original coordinates with transformed coordinates
        points[:, :3] = transformed_hom_coords[:, :3]
        return points

    @staticmethod
    def pointcloud2_to_numpy(msg: PointCloud2, rgb: bool = True) -> np.array:
        # Return [x, y, z] or [x, y, z, r, g, b] depending on the value of rgb

        fields = ["x", "y", "z"]
        if rgb:
            fields += ["rgb"]

        # Extract XYZ values from the PointCloud2 message
        structured_array = pc2.read_points(msg, field_names=fields, skip_nans=True)

        # Extract fields into a 2D array (XYZ + RGB)
        xyz = np.stack(
            [structured_array["x"], structured_array["y"], structured_array["z"]],
            axis=-1,
        )

        # Extract RGB values
        if rgb:
            rgb_float = structured_array["rgb"]
            rgb_float: np.ndarray

            rgb_int = rgb_float.view(
                np.int32
            )  # Interpret the float as int to extract RGB
            r = (rgb_int >> 16) & 0xFF
            g = (rgb_int >> 8) & 0xFF
            b = rgb_int & 0xFF
            rgb = np.stack([r, g, b], axis=-1)

            # Combine XYZ and RGB
            xyzrgb = np.hstack([xyz, rgb])
            return xyzrgb

        return xyz

    @staticmethod
    def ROI_Color_filter(
        points: np.array,
        x_range: tuple = None,
        y_range: tuple = None,
        z_range: tuple = None,
        r_range: tuple = None,
        g_range: tuple = None,
        b_range: tuple = None,
        ROI: bool = True,
        rgb: bool = True,
    ) -> np.array:
        if points.shape[1] != 3 and points.shape[1] != 6:
            print(f"points shape: {points.shape[1]}")
            raise ValueError("Invalid shape of the input points")

        ROI_filter = np.ones(points.shape[0], dtype=bool)
        RGB_filter = np.zeros(points.shape[0], dtype=bool)

        if ROI:
            ROI_filter = (
                (points[:, 0] > x_range[0])  # min_x
                & (points[:, 0] < x_range[1])  # max_x
                & (points[:, 1] > y_range[0])  # min_y
                & (points[:, 1] < y_range[1])  # max_y
                & (points[:, 2] > z_range[0])  # min_z
                & (points[:, 2] < z_range[1])  # max_z
            )

        if rgb:
            RGB_filter = (
                (points[:, 3] > r_range[0])  # min_r
                & (points[:, 3] < r_range[1])  # max_r
                & (points[:, 4] > g_range[0])  # min_g
                & (points[:, 4] < g_range[1])  # max_g
                & (points[:, 5] > b_range[0])  # min_b
                & (points[:, 5] < b_range[1])  # max_b
            )

        combined_filter = ROI_filter & ~RGB_filter

        return points[combined_filter]

    @staticmethod
    def numpy_voxel_filter(points, voxel_size=0.01):
        """
        Perform voxel grid filtering on [x, y, z, r, g, b] NumPy array.

        Args:
            points (numpy.ndarray): Nx6 array with [x, y, z, r, g, b].
            voxel_size (float): Size of the voxel grid.

        Returns:
            numpy.ndarray: Filtered Nx6 array.
        """
        # Quantize coordinates to voxel grid
        voxel_indices = np.floor(points[:, :3] / voxel_size).astype(np.int32)

        # Use unique voxel indices to identify unique points
        _, unique_indices = np.unique(voxel_indices, axis=0, return_index=True)

        # Filter points
        filtered_points = points[unique_indices]

        return filtered_points


def main():
    rclpy.init(args=None)

    node = PCDSubscriber()

    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
