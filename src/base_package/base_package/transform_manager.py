
# ROS2
import rclpy
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.publisher import Publisher
from rclpy.qos import QoSProfile, qos_profile_system_default
from rclpy.time import Time

# ROS2 Messages
from geometry_msgs.msg import *
from nav_msgs.msg import *
from sensor_msgs.msg import *
from std_msgs.msg import *
from visualization_msgs.msg import *
from custom_msgs.msg import *
from tf2_geometry_msgs.tf2_geometry_msgs import PoseStamped as TF2PoseStamped
from builtin_interfaces.msg import Duration as BuiltinDuration

# TF
import numpy as np
from tf2_ros import *
from rotutils import *


class TransformManager:
    def __init__(self, node: Node, *args, **kwargs):
        self._node = node

        self._tf_buffer = Buffer(node=self._node, cache_time=Duration(seconds=2))
        self._tf_listener = TransformListener(node=self._node, buffer=self._tf_buffer)
        self._tf_broadcaster = TransformBroadcaster(self._node)

    def check_transform_valid(self, target_frame: str, source_frame: str):
        try:
            valid = self._tf_buffer.can_transform(
                target_frame,
                source_frame,
                self._node.get_clock().now().to_msg(),
                timeout=Duration(seconds=0.1),
            )

            if not valid:
                raise Exception("Transform is not valid")

            return valid
        except Exception as e:
            self._node.get_logger().warn(
                f"Cannot Lookup Transform Between {target_frame} and {source_frame}"
            )
            self._node.get_logger().warn(e)
            return False

    def transform_pose(
        self,
        pose: Union[Pose, PoseStamped],
        target_frame: str,
        source_frame: str,
    ) -> PoseStamped:
        """
        Transform a pose from the source frame to the target frame.
        """
        if not isinstance(pose, (Pose, PoseStamped)):
            self._node.get_logger().warn("Input must be of type Pose or PoseStamped.")
            return None

        if self.check_transform_valid(target_frame, source_frame):
            try:
                transformed_pose_stamped = PoseStamped()

                if isinstance(pose, Pose):
                    pose: Pose
                    pose_stamped = TF2PoseStamped(
                        header=Header(
                            stamp=self._node.get_clock().now().to_msg(),
                            frame_id=source_frame,
                        ),
                        pose=pose,
                    )
                elif isinstance(pose, PoseStamped):
                    pose: PoseStamped
                    pose_stamped = TF2PoseStamped(
                        header=Header(
                            stamp=self._node.get_clock().now().to_msg(),
                            frame_id=source_frame,
                        ),
                        pose=pose.pose,
                    )
                else:
                    raise TypeError("Input must be of type Pose or PoseStamped.")

                transformed_data = self._tf_buffer.transform(
                    object_stamped=pose_stamped,
                    target_frame=target_frame,
                    timeout=Duration(seconds=1),
                )

                transformed_pose_stamped.header = transformed_data.header
                transformed_pose_stamped.pose = transformed_data.pose

                return transformed_pose_stamped

            except Exception as e:
                self._node.get_logger().warn(
                    f"Cannot Transform Pose from {source_frame} to {target_frame}"
                )
                self._node.get_logger().warn(e)
                return None

        return None

    def get_transform_matrix(self, target_frame: str, source_frame: str) -> Optional[np.ndarray]:
        """
        Get the transformation matrix from the source frame to the target frame.
        """
        if self.check_transform_valid(target_frame, source_frame):
            try:
                transform = self._tf_buffer.lookup_transform(
                    target_frame,
                    source_frame,
                    self._node.get_clock().now().to_msg(),
                    timeout=Duration(seconds=1),
                )
                
                x, y, z = transform.transform.translation.x, transform.transform.translation.y, transform.transform.translation.z
                translation_vector = np.array([x, y, z])

                qx, qy, qz, qw = (
                    transform.transform.rotation.x,
                    transform.transform.rotation.y,
                    transform.transform.rotation.z,
                    transform.transform.rotation.w,
                )
                roll, pitch, yaw = euler_from_quaternion([qx, qy, qz, qw])
                rotation_matrix = rotation_matrix_from_euler(roll=roll, pitch=pitch, yaw=yaw)

                mat = compose_transform(translation=translation_vector, rotation=rotation_matrix)

                return mat
            except Exception as e:
                self._node.get_logger().warn(
                    f"Cannot Lookup Transform Between {target_frame} and {source_frame}"
                )
                self._node.get_logger().warn(e)
                return None

        return None