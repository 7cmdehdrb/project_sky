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
    test_dir = os.path.join(FindPackageShare("test").find("test"), "launch")
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

    return LaunchDescription(
        [
            # UR5e 제어 런치
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(
                    os.path.join(ur_bringup_dir, "ur_control.launch.py")
                ),
                launch_arguments={
                    "ur_type": "ur5e",
                    "robot_ip": "192.168.56.101",
                    "launch_rviz": "false",
                }.items(),
            ),
            # move_group.launch.py
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(
                    os.path.join(test_dir, "move_group.launch.py")
                )
            ),
            # rsp.launch.py
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(os.path.join(test_dir, "rsp.launch.py"))
            ),
            # robotiq_control.launch.py
            # IncludeLaunchDescription(
            #     PythonLaunchDescriptionSource(
            #         os.path.join(robotiq_dir, "robotiq_control.launch.py")
            #     ),
            #     launch_arguments={"launch_rviz": "false"}.items(),
            # ),
            static_tf_node,
        ]
    )
