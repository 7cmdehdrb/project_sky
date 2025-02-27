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
import time
import sensor_msgs_py.point_cloud2 as pc2
import open3d as o3d
from scipy.spatial.transform import Rotation as R
from open3d.t.geometry import PointCloud  # type: ignore
from open3d.core import Tensor, Device  # type: ignore
from header import QuaternionAngle, PointCloudTransformer


class LowPassFilter:
    def __init__(self, cutoff_freq, ts):
        self.ts = ts
        self.cutoff_freq = cutoff_freq
        self.pre_out = 0.0
        self.tau = self.calc_filter_coef()

    def calc_filter_coef(self):
        w_cut = 2 * np.pi * self.cutoff_freq
        return 1 / w_cut

    def filter(self, data):
        out = (self.tau * self.pre_out + self.ts * data) / (self.tau + self.ts)
        self.pre_out = out
        return out


def get_pose_stamped(transform_matrix, time):
    position_matrix = transform_matrix[:3, 3]
    rotation_matrix = transform_matrix[:3, :3]

    x = position_matrix[0]
    y = position_matrix[1]
    z = position_matrix[2]

    r, p, y = QuaternionAngle.euler_from_rotation_matrix(rotation_matrix)
    quat = QuaternionAngle.quaternion_from_euler(r, p, y)

    orientation = Quaternion(x=quat[0], y=quat[1], z=quat[2], w=quat[3])

    return PoseStamped(
        header=Header(
            frame_id="camera1_link",
            stamp=time,
        ),
        pose=Pose(
            position=Point(x=x, y=y, z=z),
            orientation=orientation,
        ),
    )


# Temperal Function
def get_marker(transform_matrix, time, lpfs: list):
    position_matrix = transform_matrix[:3, 3]
    rotation_matrix = transform_matrix[:3, :3]

    x = lpfs[0].filter(position_matrix[0])
    y = lpfs[1].filter(position_matrix[1])
    z = lpfs[2].filter(position_matrix[2])

    point = Point(x=x, y=y, z=z)

    r, p, y = QuaternionAngle.euler_from_rotation_matrix(rotation_matrix)

    r = lpfs[3].filter(r)
    p = lpfs[4].filter(p)
    y = lpfs[5].filter(y)

    quat = QuaternionAngle.quaternion_from_euler(r, p, y)

    # orientation = Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)
    orientation = Quaternion(x=quat[0], y=quat[1], z=quat[2], w=quat[3])

    return Marker(
        header=Header(
            frame_id="camera1_link",
            stamp=time,
        ),
        type=Marker.CUBE,
        action=Marker.ADD,
        id=999,
        pose=Pose(
            position=point,
            orientation=orientation,
        ),
        scale=Vector3(x=0.1, y=0.1, z=0.2),
        color=ColorRGBA(r=0.0, g=1.0, b=0.0, a=0.5),
    )


class ScanMatchingNode(Node):
    def __init__(self):
        super().__init__("scan_matching_node")

        pcd_file_path = "/home/min/7cmdehdrb/ros2_ws/coca_cola_transformed.ply"

        # Resize matrix. mm -> m
        self.resize_matrix = np.array(
            [
                [0.001, 0.0, 0.0, 0.0],
                [0.0, 0.001, 0.0, 0.0],
                [0.0, 0.0, 0.001, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )

        points = np.asarray(o3d.io.read_point_cloud(pcd_file_path).points)
        resized_points = PointCloudTransformer.transform_pointcloud(
            points, self.resize_matrix
        )
        self.source_points = resized_points[::5]

        self.subscription = self.create_subscription(
            PointCloud2,
            "/camera/camera1/depth/color/points",
            self.callback_pointcloud,
            qos_profile_system_default,
        )

        self.sliced_points_pub = self.create_publisher(
            PointCloud2,
            f"/scan_matching/sliced_points",
            qos_profile_system_default,
        )

        self.transformed_points_pub = self.create_publisher(
            PointCloud2,
            f"/scan_matching/transformed_points",
            qos_profile_system_default,
        )

        self.transform_matrix_pub = self.create_publisher(
            Float32MultiArray,
            f"/scan_matching/transform_matrix",
            qos_profile_system_default,
        )

        self.test_pub = self.create_publisher(
            Marker,
            f"/scan_matching/test",
            qos_profile_system_default,
        )

        self.test_pub2 = self.create_publisher(
            PoseStamped,
            f"/scan_matching/test2",
            qos_profile_system_default,
        )

        self.points = np.empty((0, 3))

        # self.transform_matrix = np.array(
        #     [
        #         [1.0, 0.0, 0.0, 0.7],
        #         [0.0, 1.0, 0.0, -0.1],
        #         [0.0, 0.0, 1.0, -0.2],
        #         [0.0, 0.0, 0.0, 1.0],
        #     ]
        # )

        self.transform_matrix = np.eye(4)
        self.stacked_transform_matrix = np.eye(4)

        self.lpfs = [LowPassFilter(ts=0.1, cutoff_freq=0.1) for _ in range(6)]

        # ICP parameters
        self.min_threshold = 0.03
        self.threshold = 0.1
        self.scan_matching = (
            o3d.t.pipelines.registration.TransformationEstimationPointToPoint()
        )
        self.criteria = o3d.t.pipelines.registration.ICPConvergenceCriteria(
            max_iteration=200,  # 최대 반복 횟수
            relative_fitness=1e-6,  # Fitness 변화 임계값
            relative_rmse=1e-6,  # RMSE 변화 임계값
        )

        # Main loop
        hz = 10
        self.timer = self.create_timer(float(1 / hz), self.try_scan_matching)
        self.current_time = self.get_clock().now()

    def callback_pointcloud(self, msg):
        points = PointCloudTransformer.pointcloud2_to_numpy(msg, rgb=False)
        transform_matrix = QuaternionAngle.transform_realsense_to_ros(np.eye(4))

        points = PointCloudTransformer.transform_pointcloud(
            points=points, transform_matrix=transform_matrix
        )

        self.points = PointCloudTransformer.ROI_Color_filter(
            points,
            ROI=True,
            x_range=(0.5, 1.0),
            y_range=(0.0, 0.5),
            z_range=(-0.5, 0.0),
            rgb=False,
        )

        publish_sliced_points = True

        if publish_sliced_points:

            sliced_points_msg = PointCloudTransformer.numpy_to_pointcloud2(
                points=self.points,
                frame_id="camera1_link",
                stamp=self.get_clock().now().to_msg(),
                rgb=False,
            )

            self.sliced_points_pub.publish(sliced_points_msg)

    def try_scan_matching(self):
        if self.source_points.shape[0] == 0 or self.points.shape[0] == 0:
            self.get_logger().info("No pointcloud data")
            self.get_logger().info(
                "source_points: %d, points: %d"
                % (self.source_points.shape[0], self.points.shape[0])
            )
            # time.sleep(1)
            return None

        self.get_logger().info(
            "source_points: %d, points: %d, threshold: %f"
            % (
                self.source_points.shape[0],
                self.points.shape[0],
                round(self.threshold, 3),
            )
        )

        current_time = self.get_clock().now()
        dt = (current_time - self.current_time).nanoseconds / 1e9

        print(f"dt: {dt}, hz: {1/dt}")

        self.current_time = current_time

        source = self.pointcloud_to_gpu(self.source_points)  # Pointcloud of Can
        target = self.pointcloud_to_gpu(self.points)  # Real-Time Pointcloud

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
        self.stacked_transform_matrix = np.dot(
            self.stacked_transform_matrix, self.transform_matrix
        )

        if np.allclose(self.transform_matrix, np.eye(4)):
            self.threshold += 0.01
            self.get_logger().warn(
                "No transformation. Increasing threshold to %f" % self.threshold
            )
            return None

        else:
            self.threshold -= 0.01

        # Clamp the rotation matrix
        self.threshold = np.clip(self.threshold, self.min_threshold, 0.2)

        # print(f"Transform Matrix: {self.transform_matrix}")
        translation = self.stacked_transform_matrix[:3, 3]
        rotation = self.stacked_transform_matrix[:3, :3]

        r, p, y = QuaternionAngle.euler_from_rotation_matrix(rotation)

        self.get_logger().info(
            f"Translation - X: {round(translation[0], 5)}, Y: {round(translation[1], 5)}, Z: {round(translation[2], 5)}"
        )
        self.get_logger().info(
            f"Rotation - R: {round(np.rad2deg(r), 5)}, P: {round(np.rad2deg(p), 5)}, Y: {round(np.rad2deg(y), 5)}"
        )

        # Publish the transformation matrix
        transformed_points = PointCloudTransformer.transform_pointcloud(
            points=self.source_points, transform_matrix=self.transform_matrix
        )

        transformed_points_msg = PointCloudTransformer.numpy_to_pointcloud2(
            points=transformed_points,
            frame_id="camera1_link",
            stamp=self.get_clock().now().to_msg(),
            rgb=False,
        )

        self.transformed_points_pub.publish(transformed_points_msg)
        self.test_pub.publish(
            get_marker(
                self.stacked_transform_matrix,
                self.get_clock().now().to_msg(),
                self.lpfs,
            )
        )
        self.test_pub2.publish(
            get_pose_stamped(
                self.stacked_transform_matrix, self.get_clock().now().to_msg()
            )
        )

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


def main():

    rclpy.init(args=None)

    node = ScanMatchingNode()

    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
