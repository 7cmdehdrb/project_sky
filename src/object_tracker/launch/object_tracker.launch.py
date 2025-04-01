from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    # 런치 인자 선언
    model_file_arg = DeclareLaunchArgument(
        "model_file", description="Path to the model file for segmentation"
    )
    obj_bounds_file_arg = DeclareLaunchArgument(
        "obj_bounds_file", description="Path to the object bounds file"
    )
    conf_threshold_arg = DeclareLaunchArgument(
        "conf_threshold", description="Confidence threshold for segmentation"
    )

    # 노드 실행 정의
    segmentation_node = Node(
        package="object_tracker",
        executable="real_time_segmentation_node",
        name="segmentation_node",
        output="screen",
        arguments=[
            "--model_file",
            LaunchConfiguration("model_file"),
            "--obj_bounds_file",
            LaunchConfiguration("obj_bounds_file"),
            "--conf_threshold",
            LaunchConfiguration("conf_threshold"),
        ],
    )

    pose_estimation_node = Node(
        package="object_tracker",
        executable="object_pose_estimation_server",
        name="pose_estimation_server",
        output="screen",
    )

    return LaunchDescription(
        [
            model_file_arg,
            obj_bounds_file_arg,
            conf_threshold_arg,
            segmentation_node,
            pose_estimation_node,
        ]
    )
