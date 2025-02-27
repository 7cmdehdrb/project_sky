# ROS2
import rclpy
import rclpy.logging
from rclpy.node import Node
from rclpy.time import Time
from rclpy.duration import Duration
from rclpy.qos import QoSProfile, qos_profile_system_default

# Message
import rclpy.time
import rclpy.time
from std_msgs.msg import *
from geometry_msgs.msg import *
from sensor_msgs.msg import *
from nav_msgs.msg import *
from visualization_msgs.msg import *

# TF
from tf2_ros import *

# Python
import numpy as np
import cv2
import cv_bridge
import pupil_apriltags as apriltag
from header import QuaternionAngle


class AprilDetector(Node):
    class ImageSubscriber:
        def __init__(self, node: Node, camera_id: str, callback: callable):
            self.node = node

            self.camera_id = camera_id
            self.image_topic = f"/camera/{camera_id}/color/image_raw"
            self.info_topic = f"/camera/{camera_id}/color/camera_info"
            self.frame_id = f"{camera_id}_link"

            self.image_sub = self.node.create_subscription(
                Image,
                self.image_topic,
                self.image_callback,
                qos_profile=qos_profile_system_default,
            )

            self.info_sub = self.node.create_subscription(
                CameraInfo,
                self.info_topic,
                self.info_callback,
                qos_profile=qos_profile_system_default,
            )

            self.accerleration_sub = self.node.create_subscription(
                Imu,
                f"/camera/{camera_id}/accel/sample",
                self.accel_callback,
                qos_profile=qos_profile_system_default,
            )

            self.transform_matrix_pub = self.node.create_publisher(
                Float32MultiArray,
                f"/{camera_id}_transform_matrix",
                qos_profile_system_default,
            )

            self.callback = callback
            self.camera_info = CameraInfo()
            self.image = Image()

            self.transform = None
            self.transform_matrix = None
            self.orientation_imu = Vector3()

        def image_callback(self, msg: Image):
            self.image = msg
            transform, transform_matrix = self.callback(
                self.image, self.camera_info, self.frame_id, self.orientation_imu
            )

            if transform is not None and transform_matrix is not None:
                self.transform = transform
                self.transform_matrix = transform_matrix

        def info_callback(self, msg: CameraInfo):
            self.camera_info = msg

        def accel_callback(self, msg: Imu):
            linear_acceleration = msg.linear_acceleration

            ax = linear_acceleration.x
            ay = linear_acceleration.y
            az = linear_acceleration.z

            g = np.array([ax, ay, az])

            g_norm = np.linalg.norm(g)

            if np.isclose(g_norm, 0.0):
                raise ValueError(
                    "Acceleration vector magnitude is zero, cannot calculate orientation."
                )

            g = g / g_norm

            # Calculate roll and pitch
            roll = -(np.pi / 2.0) - np.arctan2(g[1], g[2])  # y-z plane determines roll
            pitch = -np.arcsin(-g[0])  # x-axis influences pitch

            yaw = 0.0

            self.orientation_imu = Vector3(x=roll, y=pitch, z=yaw)

    def __init__(self, tag_size: float = 0.04, tag_id: int = 0):
        super().__init__("april_detector_node")

        # Initialize the AprilTag detector
        self.bridge = cv_bridge.CvBridge()
        self.tag_detector = apriltag.Detector(families="tag36h11")

        self.tag_size = tag_size
        self.tag_id = tag_id

        self.camera_rotate_matrix = np.array(
            [
                [0, -1, 0, 0],
                [0, 0, -1, 0],
                [1, 0, 0, 0],
                [0, 0, 0, 1],
            ]
        )

        self.tag_rotate_matrix = np.array(
            [
                [-1, 0, 0, 0],
                [0, 0, -1, 0],
                [0, -1, 0, 0],
                [0, 0, 0, 1],
            ]
        )

        # To add more cameras, add the camera topic and frame_id here.
        # Each camera automatically subscribes to the image and camera_info topics, and update the transform and transform_matrix.
        self.camera1 = self.ImageSubscriber(self, "camera1", self.update_tf)
        self.camera2 = self.ImageSubscriber(self, "camera2", self.update_tf)
        self.camera3 = self.ImageSubscriber(self, "camera3", self.update_tf)
        # self.camera4 = self.ImageSubscriber(self, "camera4", self.update_tf)

        self.cameras = [
            self.camera1,
            self.camera2,
            self.camera3,
            # self.camera4,
        ]

        # TF
        self.buffer = Buffer(node=self, cache_time=Duration(seconds=0.1))
        self.tf_listener = TransformListener(
            self.buffer, self, qos=qos_profile_system_default
        )
        self.tf_publisher = TransformBroadcaster(self, qos=qos_profile_system_default)

        hz = 30
        self.create_timer(float(1 / hz), self.run)

    def update_tf(
        self,
        image: Image,
        camera_info: CameraInfo,
        frame_id: str,
        imu_orientation: Vector3,
    ):
        # Callback function for ImageSubscriber
        try:
            tag_transform, transform_matrix = self.detect_tags(
                image, camera_info, imu_orientation
            )
        except Exception as e:
            self.get_logger().error(f"Error in detect_tags: {e}")
            return None, None

        if tag_transform is None or transform_matrix is None:
            return None, None

        tf_msg = TransformStamped(
            header=Header(frame_id="base_link", stamp=self.get_clock().now().to_msg()),
            child_frame_id=frame_id,
            transform=tag_transform,
        )

        data = transform_matrix.flatten().tolist()

        matrix_msg = Float32MultiArray(
            data=data,
        )

        return tf_msg, matrix_msg

    def detect_tags(
        self, image: Image, camera_info: CameraInfo, imu_orientation: Vector3
    ):
        """Function to detect the AprilTag which has the specified tag_id and return the transform object(TF) and transform matrix."""
        weight = 0.7

        if camera_info is None:
            self.get_logger().error("Camera info is not available.")
            return None, None

        cv_image = self.bridge.imgmsg_to_cv2(image, desired_encoding="bgr8")
        cv_image_gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)

        camera_params = [
            camera_info.k[0],
            camera_info.k[4],
            camera_info.k[2],
            camera_info.k[5],
        ]

        detections = self.tag_detector.detect(
            cv_image_gray,
            estimate_tag_pose=True,
            tag_size=self.tag_size,
            camera_params=camera_params,
        )

        #  detections type is apriltag.Detection()
        for detection in detections:
            detection: apriltag.Detection

            tag_id = detection.tag_id

            # Check if the detected tag is the one we are looking for
            if tag_id != self.tag_id:
                self.get_logger().warn(f"Tag ID: {tag_id}")
                continue

            # Translation: Cam -> Tag
            translation_vector = detection.pose_t.flatten() + np.array([0.06, 0.0, 0.0])

            # Rotation: Cam -> Tag
            rotation_matrix = detection.pose_R
            r, p, y = QuaternionAngle.euler_from_rotation_matrix(rotation_matrix)

            # Realsense Coordinate. Rotation > Roll, Pitch, Yaw
            r = (imu_orientation.x * weight) + (r * (1 - weight))
            p = (imu_orientation.z * 0.0) + (p * (1 - 0.0))  # IMU z axis is not used.
            y = (imu_orientation.y * weight) + (y * (1 - weight))

            # Roll, Pitch, Yaw to Rotation Matrix
            rotation_matrix = QuaternionAngle.rotation_matrix_from_euler(r, p, y)

            # Create Transform Matrix
            transform_matrix = QuaternionAngle.create_transform_matrix(
                translation=translation_vector, rotation=rotation_matrix
            )

            inverse_transform_matrix = QuaternionAngle.invert_transformation(
                transform_matrix
            )

            ros_transform_matrix = (
                self.tag_rotate_matrix
                @ inverse_transform_matrix
                @ self.camera_rotate_matrix
            )

            ros_translation_vector = ros_transform_matrix[:3, 3]
            ros_rotation_matrix = ros_transform_matrix[:3, :3]

            # Ratation Matrix to Quaternion
            r, p, y = QuaternionAngle.euler_from_rotation_matrix(ros_rotation_matrix)
            ros_transformed_quaternion = QuaternionAngle.quaternion_from_euler(r, p, y)

            # Create Transform message
            transform = Transform(
                translation=Vector3(
                    x=ros_translation_vector[0],
                    y=ros_translation_vector[1],
                    z=ros_translation_vector[2],
                ),
                rotation=Quaternion(
                    x=ros_transformed_quaternion[0],
                    y=ros_transformed_quaternion[1],
                    z=ros_transformed_quaternion[2],
                    w=ros_transformed_quaternion[3],
                ),
            )

            return transform, ros_transform_matrix

        return None, None

    def run(self):
        for camera in self.cameras:
            camera: AprilDetector.ImageSubscriber

            tf_msg = camera.transform
            matrix_msg = camera.transform_matrix

            if tf_msg is not None and matrix_msg is not None:
                tf_msg: TransformStamped
                tf_msg.header.stamp = self.get_clock().now().to_msg()

                self.tf_publisher.sendTransform(tf_msg)
                camera.transform_matrix_pub.publish(matrix_msg)


def main():
    rclpy.init(args=None)

    node = AprilDetector(tag_size=0.038, tag_id=122)

    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
