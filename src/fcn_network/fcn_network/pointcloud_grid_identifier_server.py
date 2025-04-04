# Python
import os
import sys
import json
import numpy as np
import argparse
import array

# ROS2
import rclpy
from rclpy.node import Node
from rclpy.time import Time
from rclpy.duration import Duration
from rclpy.qos import QoSProfile, qos_profile_system_default

# Message
from std_msgs.msg import *
from geometry_msgs.msg import *
from sensor_msgs.msg import *
from nav_msgs.msg import *
from visualization_msgs.msg import *
from custom_msgs.srv import FCNOccupiedRequest

# TF
from tf2_ros import *

# Custom
from base_package.header import PointCloudTransformer, QuaternionAngle
from base_package.manager import ObjectManager
from fcn_network.fcn_manager import GridManager


class PointCloudGridIdentifier(Node):
    def __init__(self, *args, **kwargs):
        super().__init__("pointcloud_grid_identifier_node")

        # >>> Grid Manager >>>
        self._grid_manager = GridManager(self, *args, **kwargs)
        self._grid_data = self._grid_manager.get_grid_data()

        self._rows = self._grid_data["rows"]  # ["A", "B", "C"]
        self._cols = self._grid_data["columns"]  # [0, 1, 2, 3]
        # <<< Grid Manager <<<

        # >>> ROS >>>
        self._pointcloud_subscriber = self.create_subscription(
            PointCloud2,
            "/camera/camera1/depth/color/points",  # TODO: Change the topic
            self.pointcloud_callback,
            qos_profile_system_default,
        )
        self._grid_marker_publisher = self.create_publisher(
            MarkerArray,
            self.get_name() + "/grids",
            qos_profile_system_default,
        )
        self._srv = self.create_service(
            FCNOccupiedRequest,
            "/fcn_occupied_request",
            self.fcn_occupied_request_callback,
        )
        # <<< ROS <<<

        # >>> Data >>>
        self._pointcloud_msg: PointCloud2 = None
        # <<< Data <<<

        self.get_logger().info("Pointcloud Grid Identifier Node has been initialized.")

        # >>> Main
        hz = 10
        self._timer = self.create_timer(float(1.0 / hz), self.publish_grid_marker)
        # <<< Main

    def get_side_columns(self, column_id: int):
        side_columns = []
        for column in self._grid_manager.cols:
            column: GridManager.Line

            if int(column.id) == column_id + 1 or int(column.id) == column_id - 1:
                side_columns.append(column)

        return side_columns

    def fcn_occupied_request_callback(
        self, request: FCNOccupiedRequest.Request, response: FCNOccupiedRequest.Response
    ):
        """
        Input target column and empty columns, and return the row and columns to move.

        Request:
            str: target_col
            int[]: empty_cols
        Response:
            str: moving_row
            int[] moving_cols
            bool: action
        """
        self.get_logger().info(
            f"Request received: {request.target_col}, {request.empty_cols}"
        )

        target_column_id: int = request.target_col
        target_column: GridManager.Line = None

        # >>> STEP 1. Get the target colums >>>
        for column in self._grid_manager.cols:
            column: GridManager.Line
            if int(column.id) == target_column_id:
                target_column = column
                break

        if target_column is None:
            raise ValueError(f"Invalid target column: '{target_column_id}'.")

        # >>> STEP 2. Get the first occupied row >>>
        first_occupied_grid: GridManager.Grid = None
        for grid in target_column.grids:
            grid: GridManager.Grid

            if grid.is_occupied:
                first_occupied_grid = grid
                break

        if first_occupied_grid is None:
            raise ValueError(f"No occupied grid in column: '{target_column_id}'.")

        # >>> STEP 3. Get the first occupied row >>>
        side_columns = self.get_side_columns(target_column_id)

        results = []
        for side_column in side_columns:
            side_column: GridManager.Line

            for grid in side_column.grids:
                grid: GridManager.Grid

                if grid.is_occupied:
                    id = grid.row  # the row id of first occupied grid

                    if ord(id) > ord(first_occupied_grid.row):
                        results.append(grid.col)
                    break

        response.action = len(results) != 0  # True : Sweaping, False : Grasping
        response.moving_row = first_occupied_grid.row
        response.moving_cols = results

        action = "Sweaping" if response.action else "Grasping"

        self.get_logger().info(
            f"Response: {action} from {response.moving_row}{request.target_col} to {response.moving_row}{response.moving_cols.tolist()}"
        )

        return response

    def pointcloud_callback(self, msg: PointCloud2):
        self._pointcloud_msg = msg

    def publish_grid_marker(self):
        if self._pointcloud_msg is None:
            self.get_logger().warn("No pointcloud message to process")
            return None

        header = Header(frame_id="camera1_link", stamp=self.get_clock().now().to_msg())

        points = PointCloudTransformer.pointcloud2_to_numpy(
            msg=self._pointcloud_msg, rgb=False
        )
        transform_matrix = QuaternionAngle.transform_realsense_to_ros(np.eye(4))
        transformed_points = PointCloudTransformer.transform_pointcloud(
            points, transform_matrix
        )

        marker_array = MarkerArray()
        for grid in self._grid_manager.grids:
            grid: GridManager.Grid

            grid.slice(transformed_points)

            marker = grid.get_marker(header)
            marker_array.markers.append(marker)

            text_marker = grid.get_text_marker(header)
            marker_array.markers.append(text_marker)

        self._grid_marker_publisher.publish(marker_array)


def main():
    rclpy.init(args=None)

    from rclpy.utilities import remove_ros_args

    # Remove ROS2 arguments
    argv = remove_ros_args(sys.argv)

    parser = argparse.ArgumentParser(description="FCN Server Node")
    parser.add_argument(
        "--grid_data_file",
        type=str,
        required=True,
        default="grid_data.json",
        help="Path or file name of the grid data. If input is a file name, the file should be located in the 'resource' directory. Required",
    )

    args = parser.parse_args(argv[1:])
    kagrs = vars(args)

    node = PointCloudGridIdentifier(**kagrs)

    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
