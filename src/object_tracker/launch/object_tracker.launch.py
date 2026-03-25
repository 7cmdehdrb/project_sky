from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    root_dir = "/home/min/7cmdehdrb/project_sky"

    closest_object_node = Node(
        package="object_tracker",
        executable="closest_object_node",
        name="closest_object_classifier",
        output="screen",
        parameters=[
            {
                # "boundary": [128, 256, 384, 512],
                "boundary": [170, 300, 460],
            }
        ],
    )

    yolo_node = Node(
        package="object_tracker",
        executable="yolo_node",
        name="real_time_segmentation_node",
        output="screen",
        parameters=[
            {
                "model_file": f"{root_dir}/src/object_tracker/resource/best_hg.pt",
                "obj_bounds_file": f"{root_dir}/src/object_tracker/resource/obj_bounds.json",
                "conf_threshold": 0.7,
            }
        ],
    )

    integration_image_node = Node(
        package="object_tracker",
        executable="integration_image_node",
        name="integration_image_node",
        output="screen",
    )

    return LaunchDescription(
        [
            closest_object_node,
            yolo_node,
            integration_image_node,
        ]
    )
