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
        self._grids, self._grids_dict = self._grid_manager.create_grid()
        # <<< Grid Manager <<<

        # >>> Grid Parameters >>>
        self._rows = self._grid_data["rows"]  # ["A", "B", "C"]
        self._cols = self._grid_data["columns"]  # [0, 1, 2, 3]

        grid_identifier = self._grid_data["grid_identifier"]

        self._grid_size = Vector3(
            x=grid_identifier["grid_size"]["x"],
            y=grid_identifier["grid_size"]["y"],
            z=grid_identifier["grid_size"]["z"],
        )
        self._start_center_coord = Point(
            x=grid_identifier["start_center_coord"]["x"],
            y=grid_identifier["start_center_coord"]["y"],
            z=grid_identifier["start_center_coord"]["z"],
        )
        self._point_threshold = grid_identifier["point_threshold"]
        # <<< Grid Parameters <<<

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

        # Update All Grids
        for col in self._cols:
            is_front_grid_occupied = False
            for row in self._rows:
                grid_id = f"{row}{col}"

                grid = self._grids_dict[grid_id]
                grid: GridManager.Grid

                is_grid_occupied = grid.get_state()

                # If the front grid is occupied, the back grid should be occupied
                if is_grid_occupied:
                    grid.set_state(state=True)
                    is_front_grid_occupied = True

                elif not is_grid_occupied and is_front_grid_occupied:
                    grid.set_state(state=True)
                    is_front_grid_occupied = True

                else:
                    grid.set_state(state=False)
                    is_front_grid_occupied = False

        # Get Target col's grid
        target_grids_id = [f"{row}{request.target_col}" for row in self._rows]
        target_grids = [self._grids_dict[grid_id] for grid_id in target_grids_id]

        # In target grids, check if there is any occupied grid
        occupied_rows = []
        for grid in target_grids:
            grid: GridManager.Grid

            if grid.is_occupied:
                occupied_rows.append(grid.row_id)

        # Exception: If there is no occupied grid in target grids
        if len(occupied_rows) == 0:
            self.get_logger().warn("No occupied grid.")
            response.moving_row = "Z"
            return response

        # Get Minimum row of the occupied grids
        first_occupied_row = min(occupied_rows)  # For example, 'A'

        # Check side grids
        side_grids_id = [f"{first_occupied_row}{col}" for col in request.empty_cols]

        # Check if the side grids are empty
        result = []
        for grid_id in side_grids_id:
            grid: GridManager.Grid = self._grids_dict[grid_id]

            # Occupied: False, Empty: True
            result.append((grid, not grid.is_occupied))

        # Set the response
        moving_cols = []
        for grid, is_empty in result:
            grid: GridManager.Grid
            if is_empty:
                response.action = True
                moving_cols.append(grid.col_id)

        response.action = False
        response.moving_row = first_occupied_row
        response.moving_cols = moving_cols

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
        for grid in self._grids:
            grid: GridManager.Grid

            grid.slice(transformed_points)

            marker = grid.get_marker(header)
            marker_array.markers.append(marker)

            text_marker = grid.get_text_marker(header)
            marker_array.markers.append(text_marker)

        self._grid_marker_publisher.publish(marker_array)


def main():
    rclpy.init(args=None)

    parser = argparse.ArgumentParser(description="FCN Server Node")
    parser.add_argument(
        "--grid_data_file",
        type=str,
        required=True,
        default="grid_data.json",
        help="Path or file name of the grid data. If input is a file name, the file should be located in the 'resource' directory. Required",
    )

    args = parser.parse_args()
    kagrs = vars(args)

    node = PointCloudGridIdentifier(**kagrs)

    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
