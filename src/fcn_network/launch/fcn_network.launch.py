from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    drl_node = Node(
        package="fcn_network",
        executable="drl_node",
        name="policy_service_node",
        output="screen",
        parameters=[
            {
                "model_path": "/home/min/7cmdehdrb/project_sky/src/fcn_network/resource/exported_45/policy.onnx",
            }
        ],
    )

    drop_grid_node = Node(
        package="fcn_network",
        executable="drop_grid_node",
        name="drop_grid_node",
        output="screen",
        parameters=[
            {
                "drop_grid_json_path": "/home/min/7cmdehdrb/project_sky/src/fcn_network/resource/drop_grid_data.json",
            }
        ],
    )

    fcn_server = Node(
        package="fcn_network",
        executable="fcn_node",
        name="fcn_service_node",
        output="screen",
        parameters=[
            {
                "fcn_gain": 2.0,
                "fcn_gamma": 0.7,
                "model_path": "/home/min/7cmdehdrb/project_sky/src/fcn_network/resource/best_model.pth",
                "fcn_image_transform": True,
                "peak_boundaries": [0, 128, 256, 384, 512, 640],
            }
        ],
    )

    grid_node = Node(
        package="fcn_network",
        executable="grid_node",
        name="grid_distance_publisher_node",
        output="screen",
        parameters=[
            {
                "grid_json_path": "/home/min/7cmdehdrb/project_sky/src/fcn_network/resource/grid_data34.json",
            }
        ],
    )

    return LaunchDescription(
        [
            drl_node,
            drop_grid_node,
            fcn_server,
            grid_node,
        ]
    )
