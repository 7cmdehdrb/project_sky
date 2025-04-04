from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    # 런치 인자 선언
    grid_data_file_arg = DeclareLaunchArgument(
        "grid_data_file", description="Path to the grid data file"
    )

    obj_bounds_file_arg = DeclareLaunchArgument(
        "obj_bounds_file", description="Path to the object bounds file"
    )

    is_test_arg = DeclareLaunchArgument(
        "test_bench", default_value="false", description="Test Bench Mode"
    )

    pose_estimation_node = Node(
        package="object_tracker",
        executable="pointcloud_pose_estimation_server",
        name="pointcloud_pose_estimation_server",
        output="screen",
        arguments=[
            "--grid_data_file",
            LaunchConfiguration("grid_data_file"),
            "--obj_bounds_file",
            LaunchConfiguration("obj_bounds_file"),
            "--test_bench",
            LaunchConfiguration("test_bench"),
        ],
    )

    return LaunchDescription(
        [grid_data_file_arg, obj_bounds_file_arg, is_test_arg, pose_estimation_node]
    )
