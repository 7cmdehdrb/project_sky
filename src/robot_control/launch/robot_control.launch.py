from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    # 런치 인자 선언
    use_depth_arg = DeclareLaunchArgument(
        "use_depth", description="Whether to use depth information"
    )
    refiner_iterations_arg = DeclareLaunchArgument(
        "refiner_iterations", description="Number of iterations for the refiner"
    )
    obj_bounds_file_arg = DeclareLaunchArgument(
        "obj_bounds_file", description="Path to the object bounds file"
    )
    host_arg = DeclareLaunchArgument(
        "host", description="Host address for the robot control"
    )
    port_arg = DeclareLaunchArgument(
        "port", description="Port number for the robot control"
    )
    target_cls_arg = DeclareLaunchArgument(
        "target_cls", description="Target class for the robot control"
    )

    main_node = Node(
        package="robot_control",
        executable="main",
        name="main_control_node",
        output="screen",
        arguments=[
            "--use_depth",
            LaunchConfiguration("use_depth"),
            "--refiner_iterations",
            LaunchConfiguration("refiner_iterations"),
            "--obj_bounds_file",
            LaunchConfiguration("obj_bounds_file"),
            "--host",
            LaunchConfiguration("host"),
            "--port",
            LaunchConfiguration("port"),
            "--target_cls",
            LaunchConfiguration("target_cls"),
        ],
    )

    return LaunchDescription(
        [
            use_depth_arg,
            refiner_iterations_arg,
            obj_bounds_file_arg,
            host_arg,
            port_arg,
            target_cls_arg,
            main_node,
        ]
    )
