from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.substitutions import FindPackageShare
from launch.substitutions import PathJoinSubstitution


def generate_launch_description():

    ur_control = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [FindPackageShare("ur_robot_driver"), "launch", "ur_control.launch.py"]
            )
        ),
        launch_arguments={
            "robot_ip": "192.168.2.2",
            "ur_type": "ur5e",
            "launch_rviz": "false",
            "use_tool_communication": "true",
            "tool_voltage": "24",
        }.items(),
    )

    move_group = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [
                    FindPackageShare("ur5e_robotiq_config"),
                    "launch",
                    "move_group.launch.py",
                ]
            )
        )
    )

    rsp = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [FindPackageShare("ur5e_robotiq_config"), "launch", "rsp.launch.py"]
            )
        )
    )

    gripper = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [
                    FindPackageShare("robotiq_description"),
                    "launch",
                    "robotiq_control.launch.py",
                ]
            )
        )
    )

    moveit_rviz = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [
                    FindPackageShare("ur5e_robotiq_config"),
                    "launch",
                    "moveit_rviz.launch.py",
                ]
            )
        )
    )

    return LaunchDescription(
        [
            ur_control,
            move_group,
            rsp,
            # gripper
            # moveit_rviz,  # RViz는 필요할 때 켜는 걸로 (시뮬레이터에서는 안 켜도 충분히 테스트 가능)
        ]
    )
