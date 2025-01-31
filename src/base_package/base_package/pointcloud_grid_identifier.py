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

# TF
from tf2_ros import *

# Python
import numpy as np
from header import PointCloudTransformer, QuaternionAngle


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

        def slice(self, points: np.array):
            xrange = (
                self.center_coord.x - self.size.x / 2,
                self.center_coord.x + self.size.x / 2,
            )
            yrange = (
                self.center_coord.y - self.size.y / 2,
                self.center_coord.y + self.size.y / 2,
            )
            zrange = (
                self.center_coord.z - self.size.z / 2,
                self.center_coord.z + self.size.z / 2,
            )

            points_in_grid = PointCloudTransformer.ROI_Color_filter(
                points,
                ROI=True,
                x_range=xrange,
                y_range=yrange,
                z_range=zrange,
                rgb=False,
            )

            self.points = points_in_grid.shape[0]

        def get_marker(self, header: Header):
            marker = Marker(
                header=header,
                ns=f"{self.row_id}{self.col_id}",
                id=((ord(self.row_id) - 64) * 10) + self.col_id,
                type=Marker.CUBE,
                action=Marker.ADD,
                pose=Pose(
                    position=self.center_coord,
                    orientation=Quaternion(x=0.0, y=0.0, z=0.0, w=1.0),
                ),
                scale=Vector3(
                    x=self.size.x * 0.9,
                    y=self.size.y * 0.9,
                    z=0.05,
                ),
                color=ColorRGBA(
                    r=1.0 if self.points < self.threshold else 0.0,
                    g=0.0 if self.points < self.threshold else 1.0,
                    b=0.0,
                    a=0.5,
                ),
            )

            return marker

        def get_text_marker(self, header: Header):
            marker = Marker(
                header=header,
                ns=f"{self.row_id}{self.col_id}_text",
                id=((ord(self.row_id) - 64) * 10) + self.col_id + 100,
                type=Marker.TEXT_VIEW_FACING,
                action=Marker.ADD,
                pose=Pose(
                    position=self.center_coord,
                    orientation=Quaternion(x=0.0, y=0.0, z=0.0, w=1.0),
                ),
                scale=Vector3(
                    x=0.0,
                    y=0.0,
                    z=0.05,
                ),
                color=ColorRGBA(r=1.0, g=1.0, b=1.0, a=0.7),
                text=f"{self.row_id}{self.col_id}\t{int(self.points * 0.001)}k",
            )

            return marker

    def __init__(self):
        super().__init__("pointcloud_grid_identifier_node")

        self.rows = ["A", "B", "C"]  # A -> Front(-), C -> Back(+)
        self.cols = [1, 2, 3, 4, 5]  # 1 -> Right(-), 5 -> Left(+)
        self.grid_size = Vector3(x=0.5, y=0.5, z=10.0)
        self.start_center_coord = Point(x=0.5, y=-1.0, z=0.0)
        self.point_threshold = 10000

        self.grids = self.create_grid(
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

        self.timer = self.create_timer(1.0, self.publish_grid_marker)

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
        for r, row in enumerate(rows):
            for c, col in enumerate(cols):

                center_coord = Point(
                    x=start_center_coord.x + grid_size.x * r,
                    y=start_center_coord.y + grid_size.y * c,
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

        return grids

    def publish_grid_marker(self):
        header = Header(frame_id="camera1_link", stamp=self.get_clock().now().to_msg())

        marker_array = MarkerArray()
        for grid in self.grids:
            grid: PointCloudGridIdentifier.Grid

            marker = grid.get_marker(header)
            text_marker = grid.get_text_marker(header)

            marker_array.markers.append(marker)
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
