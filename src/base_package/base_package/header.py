# ROS
from rclpy.time import Time

# Message
from std_msgs.msg import *
from geometry_msgs.msg import *
from sensor_msgs.msg import *
from nav_msgs.msg import *
from visualization_msgs.msg import *
import sensor_msgs_py.point_cloud2 as pc2

# Python
import numpy as np
from scipy.spatial.transform import Rotation as R


class PointCloudTransformer:
    @staticmethod
    def numpy_to_pointcloud2(
        points: np.ndarray, frame_id: str, stamp: Time, rgb: bool = True
    ) -> PointCloud2:
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
    def transform_pointcloud(
        points: np.ndarray, transform_matrix: np.ndarray
    ) -> np.ndarray:
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
    def pointcloud2_to_numpy(msg: PointCloud2, rgb: bool = True) -> np.ndarray:
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
        points: np.ndarray,
        x_range: tuple = None,
        y_range: tuple = None,
        z_range: tuple = None,
        r_range: tuple = None,
        g_range: tuple = None,
        b_range: tuple = None,
        ROI: bool = True,
        rgb: bool = True,
    ) -> np.ndarray:
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
    def numpy_voxel_filter(points: np.ndarray, voxel_size=0.01) -> np.ndarray:
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


class Queue(object):
    def __init__(self, max_size=10):
        self._max_size = max_size
        self._queue = []

    def push(self, item: float):
        if len(self._queue) >= self._max_size:
            self._queue.pop(0)
        self._queue.append(item)
        return self._queue

    def get(self):
        return self._queue

    def push_and_get_average(self, item: float):
        self.push(item)
        return self.get_average()

    def get_average(self):
        if len(self._queue) != self._max_size:
            return 0.0

        return np.mean(self._queue, axis=0)


import argparse


def str2bool(v: str):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")
