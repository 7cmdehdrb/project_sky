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
import sensor_msgs_py.point_cloud2 as pc2
import open3d as o3d
from scipy.spatial.transform import Rotation as R
from open3d.t.geometry import PointCloud  # type: ignore
from open3d.core import Tensor, Device  # type: ignore


class ScanMatchingNode(Node):
    def __init__(self):
        super().__init__("scan_matching_node")

        self.subscription1 = self.create_subscription(
            PointCloud2,
            "/camera/camera1/depth/color/points",
            self.callback_pointcloud1,
            qos_profile_system_default,
        )
        self.subscription2 = self.create_subscription(
            PointCloud2,
            "/camera/camera2/depth/color/points",
            self.callback_pointcloud2,
            qos_profile_system_default,
        )

        self.transform_matrix_pub = self.create_publisher(
            Float32MultiArray,
            f"/scan_matching/transform_matrix",
            qos_profile_system_default,
        )

        self.points1 = np.empty((0, 3))
        self.points2 = np.empty((0, 3))
        self.transform_matrix = np.eye(4)

        # Test
        self.transform_matrix = np.array(
            [
                0.6109728217124939,
                -0.1306668072938919,
                0.7807934284210205,
                0.09586578607559204,
                -0.7890997529029846,
                -0.021392466500401497,
                0.6138924360275269,
                -0.32003140449523926,
                -0.06351226568222046,
                -0.9911954998970032,
                -0.11617935448884964,
                -0.01914910040795803,
                0.0,
                0.0,
                0.0,
                1.0,
            ]
        ).reshape(4, 4)

        self.threshold = 0.02

        self.scan_matching = (
            o3d.t.pipelines.registration.TransformationEstimationPointToPoint()
        )

        self.criteria = o3d.t.pipelines.registration.ICPConvergenceCriteria(
            max_iteration=200,  # 최대 반복 횟수
            relative_fitness=1e-6,  # Fitness 변화 임계값
            relative_rmse=1e-6,  # RMSE 변화 임계값
        )

        self.timer = self.create_timer(0.1, self.try_scan_matching)

        self.current_time = self.get_clock().now()

    def callback_pointcloud1(self, msg):
        self.points1 = self.pointcloud2_to_numpy(msg)[::3]

    def callback_pointcloud2(self, msg):
        self.points2 = self.pointcloud2_to_numpy(msg)[::3]

    @staticmethod
    def pointcloud2_to_numpy(msg: PointCloud2) -> np.array:
        fields = ["x", "y", "z"]

        # Extract XYZ values from the PointCloud2 message
        structured_array = pc2.read_points(msg, field_names=fields, skip_nans=True)

        # Extract fields into a 2D array (XYZ + RGB)
        return np.stack(
            [structured_array["x"], structured_array["y"], structured_array["z"]],
            axis=-1,
        )

    def try_scan_matching(self):
        if self.points1.shape[0] == 0 or self.points2.shape[0] == 0:
            self.get_logger().warn("Missing point clouds. Skipping scan matching...")
            self.get_logger().warn(f"Point Cloud 1: {self.points1.shape} points")
            self.get_logger().warn(f"Point Cloud 2: {self.points2.shape} points")
            return None

        current_time = self.get_clock().now()
        dt = (current_time - self.current_time).nanoseconds / 1e9

        print(f"dt: {dt}, hz: {1/dt}")

        self.current_time = current_time

        source = self.pointcloud_to_gpu(self.points1)
        target = self.pointcloud_to_gpu(self.points2)

        # Perform GPU-based ICP
        result = self.perform_icp(
            source,
            target,
            self.threshold,
            self.transform_matrix,
            self.scan_matching,
            self.criteria,
        )

        self.transform_matrix = result.transformation.cpu().numpy()

        ros_transform_matrix = self.transform_realsense_to_ros(self.transform_matrix)

        limited_transform_matrix = np.eye(4)
        limited_transform_matrix[:3, :3] = self.clamp_rotation_matrix(
            ros_transform_matrix[:3, :3],
            roll_limit=np.deg2rad(20.0),
            # roll_limit=None,
            pitch_limit=np.deg2rad(20.0),
            # pitch_limit=None,
            yaw_limit=None,
        )
        limited_transform_matrix[:3, 3] = ros_transform_matrix[:3, 3]

        # Publish the transformation matrix
        data = Float32MultiArray(data=list(limited_transform_matrix.flatten()))
        self.transform_matrix_pub.publish(data)

    @staticmethod
    def perform_icp(
        source, target, threshold, init_transformation, scan_matching, criteria
    ):
        return o3d.t.pipelines.registration.icp(
            source=source,
            target=target,
            max_correspondence_distance=threshold,
            init_source_to_target=Tensor(
                init_transformation, dtype=o3d.core.Dtype.Float32
            ),
            estimation_method=scan_matching,
            criteria=criteria,
        )

    @staticmethod
    def pointcloud_to_gpu(points: np.ndarray):
        # GPU 포인트 클라우드 인스턴스를 바로 생성
        gpu_pointcloud = PointCloud(
            device=Device("CUDA:0"),
        )
        gpu_pointcloud.point["positions"] = Tensor(
            points, dtype=o3d.core.Dtype.Float32, device=Device("CUDA:0")
        )
        return gpu_pointcloud

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
    def clamp_rotation_matrix(
        rotation_matrix: np.array, roll_limit=None, pitch_limit=None, yaw_limit=None
    ) -> np.array:
        if rotation_matrix.shape != (3, 3):
            raise ValueError("Input rotation matrix must be a 3x3 matrix.")

        # SVD를 사용하여 회전 행렬을 분해
        rotation = R.from_matrix(rotation_matrix)

        # Roll, Pitch, Yaw 추출 (XYZ 순서)
        roll, pitch, yaw = rotation.as_euler("xyz", degrees=False)

        roll_zero = np.deg2rad(-90.0)

        # 각도 제한
        if roll_limit is not None:
            roll -= roll_zero  # roll_zero
            roll = np.clip(roll, -roll_limit, roll_limit)
            roll += roll_zero
        if pitch_limit is not None:
            pitch = np.clip(pitch, -pitch_limit, pitch_limit)
        if yaw_limit is not None:
            yaw = np.clip(yaw, -yaw_limit, yaw_limit)

        print(
            f"roll: {np.rad2deg(roll)}, pitch: {np.rad2deg(pitch)}, yaw: {np.rad2deg(yaw)}"
        )

        # 제한된 각도로 회전 행렬 재구성
        rotation = R.from_euler("xyz", [roll, pitch, yaw], degrees=False)
        return rotation.as_matrix()


def main():

    rclpy.init(args=None)

    node = ScanMatchingNode()

    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
