from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import ThisLaunchFileDir
from launch_ros.substitutions import FindPackageShare
from launch_ros.actions import Node
import os


def generate_launch_description():
    # 경로 정의
    ur_bringup_dir = os.path.join(
        FindPackageShare("ur_bringup").find("ur_bringup"), "launch"
    )
    ur_gripper_dir = os.path.join(
        FindPackageShare("ur_gripper_enabled").find("ur_gripper_enabled"), "launch"
    )
    robotiq_dir = os.path.join(
        FindPackageShare("robotiq_description").find("robotiq_description"), "launch"
    )

    # Static TF publisher node
    static_tf_node = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name="static_tf_camera1",
        arguments=[
            "-0.04",
            "-0.39",
            "0.45",
            "0.0",
            "0.0",
            "0.7071",
            "0.7071",
            "world",
            "camera1_link",
        ],
        output="screen",
    )

    integrated_joint_states_broadcaster_node = Node(
        package="robot_control",
        executable="integrated_joint_states_broadcaster",
        name="integrated_joint_states_broadcaster",
        output="screen",
    )

    return LaunchDescription(
        [
            # UR5e 제어 런치
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(
                    os.path.join(ur_bringup_dir, "ur_control.launch.py")
                ),
                launch_arguments={
                    "ur_type": "ur5e",
                    "robot_ip": "192.168.2.2",
                    "launch_rviz": "false",
                }.items(),
            ),
            # move_group.launch.py
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(
                    os.path.join(ur_gripper_dir, "move_group.launch.py")
                )
            ),
            # rsp.launch.py
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(
                    os.path.join(ur_gripper_dir, "rsp.launch.py")
                )
            ),
            # robotiq_control.launch.py
            # IncludeLaunchDescription(
            #     PythonLaunchDescriptionSource(
            #         os.path.join(robotiq_dir, "robotiq_control.launch.py")
            #     ),
            #     launch_arguments={"launch_rviz": "false"}.items(),
            # ),
            static_tf_node,
            integrated_joint_states_broadcaster_node,
        ]
    )
