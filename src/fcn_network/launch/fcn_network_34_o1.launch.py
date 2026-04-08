from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    root_dir = "/home/irol/DRL-Occluded-Object-Search"

    drl_node = Node(
        package="fcn_network",
        executable="drl_node",
        name="policy_service_node",
        output="screen",
        parameters=[
            {
                "model_path": f"{root_dir}/src/fcn_network/resource/exported/policy.onnx",
                # "weight_fcn": [0.0, 0.0, 0.0, 0.0, 0.0],
                "weight_fcn": [1.0, 1.0, 1.0, 1.0, 1.0],
            }
        ],
    )


    # 4 col : src/fcn_network/resource/exported/policy.onnx
    # 5 col : 250506_ver1/exported/policy.onnx

    drop_grid_node = Node(
        package="fcn_network",
        executable="drop_grid_node",
        name="drop_grid_node",
        output="screen",
        parameters=[
            {
                "drop_grid_json_path": f"{root_dir}/src/fcn_network/resource/drop_grid_data.json",
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
                "model_path": f"{root_dir}/src/fcn_network/resource/3x4/best_model_3x4_ratio,O10,S0.pth",
                "fcn_image_transform": True,
                # "peak_boundaries": [0, 170, 270, 384, 480, 640],
                "peak_boundaries": [0, 170, 300, 460, 640],
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
                "grid_json_path": f"{root_dir}/src/fcn_network/resource/grid_data.json",
                # "grid_json_path": f"{root_dir}/src/fcn_network/resource/grid_data45.json",
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
