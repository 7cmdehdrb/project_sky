import numpy as np
import sensor_msgs_py.point_cloud2 as pc2
from scipy.spatial.transform import Rotation as R
from std_msgs.msg import *
from geometry_msgs.msg import *
from sensor_msgs.msg import *
from nav_msgs.msg import *
from visualization_msgs.msg import *


class QuaternionAngle:
    @staticmethod
    def euler_from_quaternion(quaternion):
        """
        In: [x, y, z, w], Out: roll, pitch, yaw
        """
        x = quaternion[0]
        y = quaternion[1]
        z = quaternion[2]
        w = quaternion[3]

        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = np.arctan2(sinr_cosp, cosr_cosp)

        sinp = 2 * (w * y - z * x)
        pitch = np.arcsin(sinp)

        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = np.arctan2(siny_cosp, cosy_cosp)

        return roll, pitch, yaw

    @staticmethod
    def quaternion_from_euler(roll, pitch, yaw):
        """
        In: roll, pitch, yaw, Out: x, y, z, w
        """
        qx = np.sin(roll / 2) * np.cos(pitch / 2) * np.cos(yaw / 2) - np.cos(
            roll / 2
        ) * np.sin(pitch / 2) * np.sin(yaw / 2)
        qy = np.cos(roll / 2) * np.sin(pitch / 2) * np.cos(yaw / 2) + np.sin(
            roll / 2
        ) * np.cos(pitch / 2) * np.sin(yaw / 2)
        qz = np.cos(roll / 2) * np.cos(pitch / 2) * np.sin(yaw / 2) - np.sin(
            roll / 2
        ) * np.sin(pitch / 2) * np.cos(yaw / 2)
        qw = np.cos(roll / 2) * np.cos(pitch / 2) * np.cos(yaw / 2) + np.sin(
            roll / 2
        ) * np.sin(pitch / 2) * np.sin(yaw / 2)

        return qx, qy, qz, qw

    @staticmethod
    def euler_from_rotation_matrix(rotation_matrix):
        """
        In: rotation_matrix, Out: roll, pitch, yaw
        """
        # 회전 행렬을 scipy의 Rotation 객체로 변환
        rotation = R.from_matrix(rotation_matrix)

        # Roll, Pitch, Yaw 추출 (XYZ 순서)
        roll, pitch, yaw = rotation.as_euler("xyz", degrees=False)

        return roll, pitch, yaw

    @staticmethod
    def rotation_matrix_from_euler(roll, pitch, yaw):
        """
        In: roll, pitch, yaw, Out: rotation_matrix
        """
        rotation = R.from_euler("xyz", [roll, pitch, yaw], degrees=False)
        return rotation.as_matrix()

    @staticmethod
    def create_transform_matrix(translation, rotation):
        """
        In: translation, rotation, Out: transform_matrix
        """
        transform_matrix = np.eye(4)
        transform_matrix[:3, :3] = rotation
        transform_matrix[:3, 3] = translation
        return transform_matrix

    @staticmethod
    def transform_realsense_to_ros(transform_matrix: np.array) -> np.array:
        """
        Realsense 좌표계를 ROS 좌표계로 변환합니다.

        Realsense 좌표계:
            X축: 이미지의 가로 방향 (오른쪽으로 증가).
            Y축: 이미지의 세로 방향 (아래쪽으로 증가).
            Z축: 카메라 렌즈가 바라보는 방향 (깊이 방향).

        ROS 좌표계:
            X축: 앞으로 나아가는 방향.
            Y축: 왼쪽으로 이동하는 방향.
            Z축: 위로 이동하는 방향.

        Args:
            transform_matrix (np.ndarray): 4x4 변환 행렬.

        Returns:
            np.ndarray: 변환된 4x4 변환 행렬 (ROS 좌표계 기준).
        """
        if transform_matrix.shape != (4, 4):
            raise ValueError("Input transformation matrix must be a 4x4 matrix.")

        # Realsense에서 ROS로 좌표계를 변환하는 회전 행렬
        realsense_to_ros_rotation = np.array(
            [[0, 0, 1], [-1, 0, 0], [0, -1, 0]]  # X -> Z  # Y -> -X  # Z -> -Y
        )

        # 변환 행렬의 분해
        rotation = transform_matrix[:3, :3]  # 3x3 회전 행렬
        translation = transform_matrix[:3, 3]  # 3x1 평행 이동 벡터

        # 좌표계 변환
        rotation_ros = realsense_to_ros_rotation @ rotation
        translation_ros = realsense_to_ros_rotation @ translation

        # 새로운 변환 행렬 구성
        transform_matrix_ros = np.eye(4)
        transform_matrix_ros[:3, :3] = rotation_ros
        transform_matrix_ros[:3, 3] = translation_ros

        return transform_matrix_ros

    @staticmethod
    def invert_transformation(matrix):
        """
        Inverts a 4x4 transformation matrix.

        Parameters:
            matrix (np.ndarray): A 4x4 transformation matrix representing A > B.

        Returns:
            np.ndarray: The inverted transformation matrix representing B > A.
        """
        if matrix.shape != (4, 4):
            raise ValueError("Input matrix must be a 4x4 matrix.")

        # Extract rotation and translation components
        rotation = matrix[:3, :3]
        translation = matrix[:3, 3]

        # Invert the rotation (transpose for orthogonal matrix)
        rotation_inv = rotation.T

        # Invert the translation
        translation_inv = -np.dot(rotation_inv, translation)

        # Construct the inverted transformation matrix
        inverted_matrix = np.eye(4)
        inverted_matrix[:3, :3] = rotation_inv
        inverted_matrix[:3, 3] = translation_inv

        return inverted_matrix


class PointCloudTransformer:
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
