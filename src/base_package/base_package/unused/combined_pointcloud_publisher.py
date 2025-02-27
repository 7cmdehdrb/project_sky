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
from header import QuaternionAngle, PointCloudTransformer


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

            # TODO: Check This Matrix is valid
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

        hz = 30
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

            camera_points = PointCloudTransformer.pointcloud2_to_numpy(
                camera.msg, rgb=rgb
            )

            # Outlier removal. Realsense Axis
            camera_points = PointCloudTransformer.ROI_Color_filter(
                points=camera_points,
                ROI=True,
                x_range=(-3.0, 3.0),  # y
                y_range=(-3.0, 3.0),  # z
                z_range=(-0.1, 3.0),  # x
                rgb=False if rgb else False,
            )[::2]

            camera_transform_matrix = camera.transform_matrix

            transformed_camera1_points = PointCloudTransformer.transform_pointcloud(
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
        ROI_filtered_points = PointCloudTransformer.ROI_Color_filter(
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

            filtered_msg = PointCloudTransformer.numpy_to_pointcloud2(
                ROI_filtered_points,
                frame_id="base_link",
                stamp=self.get_clock().now().to_msg(),
                rgb=rgb,
            )

        except Exception as e:
            self.get_logger().error(f"Exception: {e}")
            return None

        return filtered_msg


def main():
    rclpy.init(args=None)

    node = PCDSubscriber()

    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
