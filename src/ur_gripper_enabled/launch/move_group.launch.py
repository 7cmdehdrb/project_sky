from moveit_configs_utils import MoveItConfigsBuilder
from moveit_configs_utils.launches import generate_move_group_launch
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.actions import Node


def generate_launch_description():
    # moveit_config = MoveItConfigsBuilder(
    #     "ur5e", package_name="ur_gripper_enabled"
    # ).to_moveit_configs()

    moveit_config = (
        MoveItConfigsBuilder("ur5e", package_name="ur_gripper_enabled")
        .planning_pipelines(
            "ompl",
            ["ompl", "chomp"],
        )
        .to_moveit_configs()
    )

    return generate_move_group_launch(moveit_config)
