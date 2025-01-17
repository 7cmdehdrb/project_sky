import numpy as np
from scipy.spatial.transform import Rotation as R


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
