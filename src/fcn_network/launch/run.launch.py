from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    # Declare arguments
    fcn_server_arg = DeclareLaunchArgument(
        "fcn_server_arg",
        default_value="default_value1",
        description="Argument for fcn_server node",
    )
    pointcloud_grid_identifier_server_arg = DeclareLaunchArgument(
        "pointcloud_grid_identifier_server_arg",
        default_value="default_value2",
        description="Argument for pointcloud_grid_identifier_server node",
    )
    fcn_integration_server_arg = DeclareLaunchArgument(
        "fcn_integration_server_arg",
        default_value="default_value3",
        description="Argument for fcn_integration_server node",
    )

    pointcloud_grid_identifier_server_node = Node(
        package="fcn_network",
        executable="pointcloud_grid_identifier_server",
        name="pointcloud_grid_identifier_server",
        output="screen",
        parameters=[
            {"arg": LaunchConfiguration("pointcloud_grid_identifier_server_arg")}
        ],
    )

    fcn_server_node = Node(
        package="fcn_network",
        executable="fcn_server",
        name="fcn_server",
        output="screen",
        parameters=[
            {
                "model_file": LaunchConfiguration("model_file"),
                "grid_data_file": LaunchConfiguration("grid_data_file"),
                "fcn_image_transform": LaunchConfiguration("fcn_image_transform"),
                "fcn_gain": LaunchConfiguration("fcn_gain"),
            }
        ],
    )

    fcn_integration_server_node = Node(
        package="fcn_network",
        executable="fcn_integration_server",
        name="fcn_integration_server",
        output="screen",
        parameters=[{"arg": LaunchConfiguration("fcn_integration_server_arg")}],
    )

    return LaunchDescription(
        [
            fcn_server_arg,
            pointcloud_grid_identifier_server_arg,
            fcn_integration_server_arg,
            fcn_server_node,
            pointcloud_grid_identifier_server_node,
            fcn_integration_server_node,
        ]
    )
