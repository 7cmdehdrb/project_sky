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
from custom_msgs.msg import *
from moveit_msgs.msg import *

# TF
from tf2_ros import *

# Python
import numpy as np

# custom
from base_package.header import QuaternionAngle
from robot_control.ur5e_control import MoveitClient
from object_tracker.object_pose_estimator_client import MegaPoseClient


class MainControlNode(object):
    def __init__(self, node: Node):
        self._node = node

        self._moveit_client = MoveitClient(node=self._node)
        self._megapose_client = MegaPoseClient(node=self._node)

    def initialize(self):
        self._moveit_client.initialize_world()

    def run(self):
        # Get All Object Poses
        object_poses: BoundingBox3DMultiArray = (
            self._megapose_client.send_megapose_request()
        )

        if len(object_poses.data) == 0:
            self._node.get_logger().warn("No object detected.")
            return None

        # Update Planning Scene
        collision_objects = self.collision_object_from_bbox_3d(bbox_3d=object_poses)
        self._moveit_client.apply_planning_scene(collision_objects=collision_objects)

        # Plan to Target
        target: BoundingBox3D = object_poses.data[0]
        target_pose = Pose(
            position=Point(
                x=target.pose.position.x - (target.scale.x / 2.0) * 1.5,
                y=target.pose.position.y,
                z=target.pose.position.z,
            ),
            orientation=Quaternion(x=0.5, y=0.5, z=0.5, w=0.5),
        )

        header = Header(
            stamp=self._node.get_clock().now().to_msg(), frame_id="base_link"
        )

        # self._moveit_client.compute_cartesian_path(
        #     header=header,
        #     waypoints=[target_pose],
        # )

        goal_constrain = self._moveit_client.get_goal_constraint(
            pose_stamped=PoseStamped(header=header, pose=target_pose),
            orientation_constraints=None,
        )
        if goal_constrain is not None:
            kinematic_path_response = self._moveit_client.plan_kinematic_path(
                goal_constraints=[goal_constrain]
            )
            self._moveit_client.handle_plan_kinematic_path_response(
                header=header, response=kinematic_path_response
            )

    def collision_object_from_bbox_3d(self, bbox_3d: BoundingBox3DMultiArray) -> list:
        header = Header(
            stamp=self._node.get_clock().now().to_msg(),
            frame_id="base_link",
        )
        collision_objects = []

        for bbox in bbox_3d.data:
            bbox: BoundingBox3D

            collision_object = MoveitClient.create_collision_object(
                id=bbox.cls,
                header=header,
                pose=bbox.pose,
                scale=bbox.scale,
                operation=CollisionObject.ADD,
            )
            collision_objects.append(collision_object)

        return collision_objects


def main():
    rclpy.init(args=None)

    node = Node("main_control_node")
    main_node = MainControlNode(node=node)

    # Spin in a separate thread
    thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    thread.start()

    hz = 0.2
    rate = node.create_rate(hz)

    main_node.initialize()

    try:
        while rclpy.ok():
            main_node.run()
            rate.sleep()
    except KeyboardInterrupt:
        pass

    node.destroy_node()

    rclpy.shutdown()
    thread.join()


if __name__ == "__main__":
    main()
