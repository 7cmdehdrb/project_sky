from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    # 런치 인자 선언
    model_file_arg = DeclareLaunchArgument(
        "model_file", description="Path to the trained FCN model"
    )
    grid_data_file_arg = DeclareLaunchArgument(
        "grid_data_file", description="Path to the grid data file"
    )
    fcn_image_transform_arg = DeclareLaunchArgument(
        "fcn_image_transform",
        default_value="true",
        description="Whether to apply image transform",
    )
    fcn_gain_arg = DeclareLaunchArgument(
        "fcn_gain", default_value="2.0", description="Gain value for post-processing"
    )
    fcn_gamma_arg = DeclareLaunchArgument(
        "fcn_gamma", default_value="0.7", description="Gamma value for post-processing"
    )

    # FCN 서버 노드 (인자 필수)
    fcn_node = Node(
        package="fcn_network",
        executable="fcn_server",
        name="fcn_server",
        output="screen",
        arguments=[
            "--model_file",
            LaunchConfiguration("model_file"),
            "--grid_data_file",
            LaunchConfiguration("grid_data_file"),
            "--fcn_image_transform",
            LaunchConfiguration("fcn_image_transform"),
            "--fcn_gain",
            LaunchConfiguration("fcn_gain"),
            "--fcn_gamma",
            LaunchConfiguration("fcn_gamma"),
        ],
    )

    # 나머지 서버들 (별도 인자 없음으로 가정)
    grid_identifier_node = Node(
        package="fcn_network",
        executable="pointcloud_grid_identifier_server",
        name="grid_identifier_server",
        output="screen",
        arguments=[
            "--grid_data_file",
            LaunchConfiguration("grid_data_file"),
        ],
    )

    integration_node = Node(
        package="fcn_network",
        executable="fcn_integration_server",
        name="fcn_integration_server",
        output="screen",
    )

    return LaunchDescription(
        [
            model_file_arg,
            grid_data_file_arg,
            fcn_image_transform_arg,
            fcn_gain_arg,
            fcn_gamma_arg,
            fcn_node,
            grid_identifier_node,
            integration_node,
        ]
    )


from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    # 런치 인자 선언
    model_file_arg = DeclareLaunchArgument(
        "model_file", description="Path to the trained FCN model"
    )
    grid_data_file_arg = DeclareLaunchArgument(
        "grid_data_file", description="Path to the grid data file"
    )
    fcn_image_transform_arg = DeclareLaunchArgument(
        "fcn_image_transform",
        default_value="true",
        description="Whether to apply image transform",
    )
    fcn_gain_arg = DeclareLaunchArgument(
        "fcn_gain", default_value="2.0", description="Gain value for post-processing"
    )
    fcn_gamma_arg = DeclareLaunchArgument(
        "fcn_gamma", default_value="0.7", description="Gamma value for post-processing"
    )

    # FCN 서버 노드 (인자 필수)
    fcn_node = Node(
        package="fcn_network",
        executable="fcn_server",
        name="fcn_server",
        output="screen",
        arguments=[
            "--model_file",
            LaunchConfiguration("model_file"),
            "--grid_data_file",
            LaunchConfiguration("grid_data_file"),
            "--fcn_image_transform",
            LaunchConfiguration("fcn_image_transform"),
            "--fcn_gain",
            LaunchConfiguration("fcn_gain"),
            "--fcn_gamma",
            LaunchConfiguration("fcn_gamma"),
        ],
    )

    # 나머지 서버들 (별도 인자 없음으로 가정)
    grid_identifier_node = Node(
        package="fcn_network",
        executable="pointcloud_grid_identifier_server",
        name="grid_identifier_server",
        output="screen",
        arguments=[
            "--grid_data_file",
            LaunchConfiguration("grid_data_file"),
        ],
    )

    integration_node = Node(
        package="fcn_network",
        executable="fcn_integration_server",
        name="fcn_integration_server",
        output="screen",
    )

    return LaunchDescription(
        [
            model_file_arg,
            grid_data_file_arg,
            fcn_image_transform_arg,
            fcn_gain_arg,
            fcn_gamma_arg,
            fcn_node,
            grid_identifier_node,
            integration_node,
        ]
    )
