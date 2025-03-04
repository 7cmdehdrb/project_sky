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

# Python
import numpy as np
from base_package.header import PointCloudTransformer, QuaternionAngle


class PointCloudGridIdentifier(Node):
    class Grid(object):
        def __init__(
            self,
            row_id: str,
            col_id: int,
            center_coord: Point,
            size: Vector3,
            threshold: int = 1000,
        ):
            self.row_id = row_id
            self.col_id = col_id
            self.center_coord = center_coord
            self.size = size
            self.threshold = threshold

            self.points = 0
            self.is_occupied = False

        def set_state(self, state: bool = None):
            if state is not None:
                self.is_occupied = state
                return self.is_occupied

            self.is_occupied = self.points > self.threshold
            return self.is_occupied

        def get_state(self):
            return self.points > self.threshold

        def slice(self, points: np.array):
            xrange = (
                self.center_coord.x - (self.size.x / 2) * 0.9,
                self.center_coord.x + (self.size.x / 2) * 0.9,
            )
            yrange = (
                self.center_coord.y - (self.size.y / 2) * 0.9,
                self.center_coord.y + (self.size.y / 2) * 0.9,
            )
            zrange = (
                self.center_coord.z - (self.size.z / 2) * 0.9,
                self.center_coord.z + (self.size.z / 2) * 0.9,
            )

            points_in_grid: np.ndarray = PointCloudTransformer.ROI_Color_filter(
                points,
                ROI=True,
                x_range=xrange,
                y_range=yrange,
                z_range=zrange,
                rgb=False,
            )

            self.points = points_in_grid.shape[0]

        def get_marker(self, header: Header):
            temp_center_coord = Point(
                x=self.center_coord.x, y=self.center_coord.y, z=self.center_coord.z
            )
            temp_center_coord.z = -0.15

            marker = Marker(
                header=header,
                ns=f"{self.row_id}{self.col_id}",
                id=((ord(self.row_id) - 64) * 10) + self.col_id,
                type=Marker.CUBE,
                action=Marker.ADD,
                pose=Pose(
                    position=temp_center_coord,
                    orientation=Quaternion(x=0.0, y=0.0, z=0.0, w=1.0),
                ),
                scale=Vector3(
                    x=self.size.x * 0.9,
                    y=self.size.y * 0.8,
                    z=0.01,
                    # z=self.size.z,
                ),
                color=ColorRGBA(
                    r=1.0 if self.points > self.threshold else 0.0,
                    g=0.0 if self.points > self.threshold else 1.0,
                    b=0.0,
                    a=0.5,
                ),
            )

            return marker

        def get_text_marker(self, header: Header):
            temp_center_coord = Point(
                x=self.center_coord.x, y=self.center_coord.y, z=self.center_coord.z
            )
            temp_center_coord.z = -0.15

            marker = Marker(
                header=header,
                ns=f"{self.row_id}{self.col_id}_text",
                id=((ord(self.row_id) - 64) * 10) + self.col_id + 100,
                type=Marker.TEXT_VIEW_FACING,
                action=Marker.ADD,
                pose=Pose(
                    position=temp_center_coord,
                    orientation=Quaternion(x=0.0, y=0.0, z=0.0, w=1.0),
                ),
                scale=Vector3(
                    x=0.0,
                    y=0.0,
                    z=0.01,
                ),
                color=ColorRGBA(r=1.0, g=1.0, b=1.0, a=0.7),
                text=f"{self.row_id}{self.col_id}\t{int(self.points * 0.001)}k",
                # text=f"{int(self.points)}",
            )

            return marker

    def __init__(self):
        super().__init__("pointcloud_grid_identifier_node")

        # Grid Parameters
        self.rows = ["A", "B", "C"]  # A -> Front(-), C -> Back(+)
        self.cols = [0, 1, 2, 3]  # Left(+) -> Right(-)
        self.grid_size = Vector3(x=0.13, y=0.20, z=0.33)
        self.start_center_coord = Point(x=0.75, y=0.3, z=0.0)
        self.point_threshold = 100

        # self.rows = [
        #     chr(i) for i in range(65, 65 + 15)
        # ]  # A -> Front(-), C -> Back(+)
        # self.cols = [i for i in range(32)]  # 1 -> Right(-), 5 -> Left(+)
        # self.grid_size = Vector3(x=0.025, y=0.025, z=0.25)
        # self.start_center_coord = Point(x=0.7, y=-0.4, z=0.0)
        # self.point_threshold = 10

        self.grids, self.grids_dict = self.create_grid(
            rows=self.rows,
            cols=self.cols,
            grid_size=self.grid_size,
            start_center_coord=self.start_center_coord,
        )

        # ROS
        self.pointcloud_subscriber = self.create_subscription(
            PointCloud2,
            "/camera/camera1/depth/color/points",  # TODO: Change the topic
            self.pointcloud_callback,
            qos_profile_system_default,
        )

        self.grid_marker_publisher = self.create_publisher(
            MarkerArray,
            "/pointcloud_grid_identifier_node/grids",
            qos_profile_system_default,
        )

        self.srv = self.create_service(
            FCNOccupiedRequest, "fcn_occupied_request", self.fcn_request_callback
        )

        hz = 10
        self.timer = self.create_timer(float(1.0 / hz), self.publish_grid_marker)

    def fcn_request_callback(
        self, request: FCNOccupiedRequest.Request, response: FCNOccupiedRequest.Response
    ):
        # Update All Grids
        for col in self.cols:
            is_front_grid_occupied = False
            for row in self.rows:
                grid_id = f"{row}{col}"

                grid = self.grids_dict[grid_id]
                grid: PointCloudGridIdentifier.Grid

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
        target_grids_id = [f"{row}{request.target_col}" for row in self.rows]
        target_grids = [self.grids_dict[grid_id] for grid_id in target_grids_id]

        # In target grids, check if there is any occupied grid
        occupied_rows = []
        for grid in target_grids:
            grid: PointCloudGridIdentifier.Grid

            if grid.is_occupied:
                occupied_rows.append(grid.row_id)

        # Get Minimum row of the occupied grids
        first_occupied_row = min(occupied_rows)  # For example, 'A'

        # Check side grids
        side_grids_id = [f"{first_occupied_row}{col}" for col in request.empty_cols]

        # Check if the side grids are empty
        result = []
        for grid_id in side_grids_id:
            grid: PointCloudGridIdentifier.Grid = self.grids_dict[grid_id]

            # Occupied: False, Empty: True
            result.append((grid, not grid.is_occupied))

        # Set the response
        response.action = "grasping"
        for grid, is_empty in result:
            if is_empty:
                response.action = "sweeping"
                response.moving_cols.append(grid.col_id)

        return response

    def pointcloud_callback(self, msg: PointCloud2):
        points = PointCloudTransformer.pointcloud2_to_numpy(msg=msg, rgb=False)
        transform_matrix = QuaternionAngle.transform_realsense_to_ros(np.eye(4))
        transformed_points = PointCloudTransformer.transform_pointcloud(
            points, transform_matrix
        )

        for grid in self.grids:
            grid: PointCloudGridIdentifier.Grid
            grid.slice(transformed_points)

    def create_grid(self, rows, cols, grid_size: Vector3, start_center_coord: Point):
        grids = []
        grids_dict = {}

        for r, row in enumerate(rows):
            for c, col in enumerate(cols):

                center_coord = Point(
                    x=start_center_coord.x + grid_size.x * r,
                    y=start_center_coord.y - grid_size.y * c,
                    z=start_center_coord.z,
                )

                grid = self.Grid(
                    row_id=row,
                    col_id=col,
                    center_coord=center_coord,
                    size=grid_size,
                    threshold=self.point_threshold,
                )

                grids.append(grid)
                grids_dict[f"{row}{col}"] = grid

        return grids, grids_dict

    def publish_grid_marker(self):
        header = Header(frame_id="camera1_link", stamp=self.get_clock().now().to_msg())

        marker_array = MarkerArray()
        for grid in self.grids:
            grid: PointCloudGridIdentifier.Grid

            marker = grid.get_marker(header)
            marker_array.markers.append(marker)

            text_marker = grid.get_text_marker(header)
            marker_array.markers.append(text_marker)

        if len(marker_array.markers) == 0:
            self.get_logger().warn("No grid markers to publish")
            return None

        self.grid_marker_publisher.publish(marker_array)


def main():

    rclpy.init(args=None)

    node = PointCloudGridIdentifier()

    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
