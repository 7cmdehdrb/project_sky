# ROS2
import rclpy
from rclpy.node import Node
from rclpy.time import Time
from rclpy.duration import Duration
from rclpy.qos import QoSProfile, qos_profile_system_default

# ROS2 Messages
from std_msgs.msg import *
from geometry_msgs.msg import *
from sensor_msgs.msg import *
from nav_msgs.msg import *
from visualization_msgs.msg import *
from custom_msgs.srv import FCNRequest, FCNOccupiedRequest, FCNIntegratedRequest

# ROS2 TF
from tf2_ros import *

# Python Standard Libraries
import os
import sys
import json
from enum import Enum
import argparse

# Third-party Libraries
import numpy as np
import cv2
import cv_bridge
from PIL import Image as PILImage
from matplotlib import pyplot as plt
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d

# PyTorch
import torch
from torch import nn, Tensor
from torchvision.transforms import Normalize
from torchvision.models.segmentation import fcn_resnet50

# Custom Modules
from base_package.header import PointCloudTransformer, QuaternionAngle
from base_package.manager import Manager, ImageManager, ObjectManager
from object_tracker.real_time_segmentation import RealTimeSegmentationNode
from ament_index_python.packages import get_package_share_directory


# FCNModel class
class FCNModel(nn.Module):
    def __init__(self):
        super(FCNModel, self).__init__()
        self.model = fcn_resnet50(
            weights=None
        )  # pretrained=False는 weights=None으로 대체됨
        self.model.classifier[4] = nn.Conv2d(512, 12, kernel_size=1)

    def forward(self, x):
        return self.model(x)["out"]


class FCNManager(Manager):
    def __init__(self, node: Node, *args, **kwargs):
        super().__init__(node, *args, **kwargs)

        self._last_results_data: np.ndarray = None

        # >>> Initialize the FCN Model >>>
        self._model = FCNModel()
        self._device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # <<< Initialize the FCN Model <<<

        # >>> Load Files >>>
        fcn_package_path = get_package_share_directory("fcn_network")

        resource_path = os.path.join(
            fcn_package_path, "../ament_index/resource_index/packages"
        )

        model_path = kwargs["model_file"]
        if not os.path.isfile(model_path):
            model_path = os.path.join(resource_path, model_path)

        # <<< Load Files <<<

        # >>> FCN Parameters >>>
        self._do_transform = kwargs["fcn_image_transform"]
        self._gain = kwargs["fcn_gain"]
        self._gamma = kwargs["fcn_gamma"]
        self._transformer = Normalize(
            mean=[0.6501676617516412, 0.6430160918638441, 0.6165616299396091],
            std=[0.16873769158467197, 0.17505241356408263, 0.1989546266815883],
        )
        self._state_dict: dict = torch.load(
            model_path,
            map_location=self._device,
        )
        self._filtered_state_dict = {
            k: v for k, v in self._state_dict.items() if "aux_classifier" not in k
        }
        # <<< FCN Model Parameters <<<

        # >>> Initialize the FCN Model
        self._model.eval()
        self._model.load_state_dict(self._filtered_state_dict, strict=False)
        self._model = self._model.to(self._device)
        # <<< Initialize the FCN Model

    def post_process_raw_image(self, img: np.ndarray) -> Tensor:
        if img.ndim == 2:
            img = np.stack([img] * 3, axis=-1)
        elif img.shape[-1] == 4:
            img = img[..., :3]

        img = img.astype(np.float32) / 255.0
        img = img.transpose(2, 0, 1)

        if self._do_transform:
            img = torch.tensor(img, dtype=torch.float32)
            img = self._transformer(img)
        else:
            img = torch.tensor(img, dtype=torch.float32)

        return img

    def predict(self, np_image: np.ndarray):
        self._model.eval()

        np_image = self.post_process_raw_image(np_image)

        tensor_img: Tensor = Tensor(np_image).to(self._device)
        tensor_img = tensor_img.unsqueeze(0)  # Add batch dimension

        outputs: Tensor = self._model(tensor_img)
        outputs = outputs.squeeze(0)

        np_outputs = outputs.detach().cpu().numpy()

        return np_outputs

    def post_process_results(self, results: np.ndarray, weights: list):
        """
        Input FCN results and weights, and return the target column and empty columns.
        Returns:
            [int: max_peak_idx, int[]: res(Available side columns), np.ndarray: data]
        """

        # results = np.clip(results, 0.0, 1.0)

        normalized_results = results * np.exp(
            -self._gain * (1 - results)
        )  # 지수 함수로 가중치 적용

        data = np.sum(normalized_results, axis=0)
        if self._last_results_data is not None:
            data = data * self._gamma + (1 - self._gamma) * self._last_results_data

        num_peaks = len(weights)

        # Find the top peaks and apply weights
        top_peak_idx, top_peak_datas = self.find_top_peaks(
            data, num_peaks=num_peaks, smooth_sigma=10, min_distance=10
        )
        if len(top_peak_idx) != 4:
            raise ValueError(
                f"Number of peaks found: {len(top_peak_idx)}. Expected: {num_peaks}."
            )

        top_peak_datas = top_peak_datas * weights

        # Sort the top_peak_idx and top_peak_datas in ascending order of top_peak_idx
        sorted_indices = np.argsort(top_peak_idx)
        top_peak_idx = np.array(top_peak_idx)[sorted_indices]
        top_peak_datas = np.array(top_peak_datas)[sorted_indices]

        max_peak_idx = int(np.argmax(top_peak_datas))
        res = [
            idx
            for idx in range(max_peak_idx - 1, max_peak_idx + 2)
            if 0 <= idx < num_peaks and idx != max_peak_idx
        ]

        # target_col, empty_cols, post_processed_data
        return max_peak_idx, res, data

    def find_top_peaks(self, data, num_peaks=4, smooth_sigma=5, min_distance=10):
        """
        데이터에서 상위 num_peaks개의 주요 피크를 찾는 함수.

        :param data: 1차원 배열 (측정 데이터)
        :param num_peaks: 찾을 피크 개수
        :param smooth_sigma: 가우시안 필터의 표준편차 (노이즈 제거)
        :param min_distance: 피크 간 최소 거리 (작은 봉우리를 무시)
        :return: 상위 num_peaks개의 피크 인덱스와 해당 값
        """

        # 1. 스무딩 적용 (노이즈 완화)
        smoothed_data = gaussian_filter1d(data, sigma=smooth_sigma)

        # 2. 피크 찾기 (높이 및 거리 조건 적용)
        peaks, _ = find_peaks(smoothed_data, distance=min_distance)

        # 3. 상위 num_peaks개의 피크 선택
        top_peaks = sorted(peaks, key=lambda x: data[x], reverse=True)[:num_peaks]

        return top_peaks, data[top_peaks]


class FCNClientManager(Manager):
    def __init__(self, node: Node, *args, **kwargs):
        super().__init__(node, *args, **kwargs)

        # >>> Managers >>>
        self._object_manager = ObjectManager(self._node, *args, **kwargs)
        # <<< Managers >>>

        # >>> Data >>>
        self._cls: str = None
        # <<< Data >>>

        # >>> ROS2 >>>
        self._cls_subscriber = self._node.create_subscription(
            String,
            self._node.get_name() + "/fcn_target_cls",
            self.cls_callback,
            qos_profile=qos_profile_system_default,
        )
        self._client = self._node.create_client(
            FCNIntegratedRequest, "/fcn_integrated_request"
        )
        # <<< ROS2 <<<

        # No main loop. This class is used as a callback

    def cls_callback(self, msg: String):
        if msg.data in self._object_manager.names.keys():
            self._cls = msg.data

    def send_fcn_integrated_request(self):
        if self._cls is None:
            return None

        request = FCNIntegratedRequest.Request()
        request.target_cls = self._cls
        response: FCNIntegratedRequest.Response = self._client.call(request)

        self._cls = None

        return response


class GridManager(Manager):
    class Grid(object):
        def __init__(
            self,
            row_id: str,
            col_id: int,
            center_coord: Point,
            size: Vector3,
            threshold: int = 1000,
        ):
            self._row_id = row_id
            self._col_id = col_id

            self._size = size
            self._center_coord = center_coord

            self._threshold = threshold

            self._points = 0

        @property
        def row(self):
            return self._row_id

        @property
        def col(self):
            return self._col_id

        @property
        def center_coord(self):
            return self._center_coord

        @property
        def is_occupied(self):
            return self._points > self._threshold

        def slice(self, points: np.array):
            xrange = (
                self._center_coord.x - (self._size.x / 2) * 0.9,
                self._center_coord.x + (self._size.x / 2) * 0.9,
            )
            yrange = (
                self._center_coord.y - (self._size.y / 2) * 0.9,
                self._center_coord.y + (self._size.y / 2) * 0.9,
            )
            zrange = (
                self._center_coord.z - (self._size.z / 2) * 0.9,
                self._center_coord.z + (self._size.z / 2) * 0.9,
            )

            points_in_grid: np.ndarray = PointCloudTransformer.ROI_Color_filter(
                points,
                ROI=True,
                x_range=xrange,
                y_range=yrange,
                z_range=zrange,
                rgb=False,
            )

            self._points = points_in_grid.shape[0]

        def get_marker(self, header: Header):
            marker = Marker(
                header=header,
                ns=f"{self._row_id}{self._col_id}",
                id=((ord(self._row_id) - 64) * 10) + self._col_id,
                type=Marker.CUBE,
                action=Marker.ADD,
                pose=Pose(
                    position=self._center_coord,
                    orientation=Quaternion(x=0.0, y=0.0, z=0.0, w=1.0),
                ),
                scale=Vector3(
                    x=self._size.x * 0.9,
                    y=self._size.y * 0.9,
                    z=self._size.z * 0.9,
                ),
                color=ColorRGBA(
                    r=1.0 if self._points > self._threshold else 0.0,
                    g=0.0 if self._points > self._threshold else 1.0,
                    b=0.0,
                    a=0.5,
                ),
            )

            return marker

        def get_text_marker(self, header: Header):
            marker = Marker(
                header=header,
                ns=f"{self._row_id}{self._col_id}_text",
                id=((ord(self._row_id) - 64) * 10) + self._col_id + 100,
                type=Marker.TEXT_VIEW_FACING,
                action=Marker.ADD,
                pose=Pose(
                    position=self._center_coord,
                    orientation=Quaternion(x=0.0, y=0.0, z=0.0, w=1.0),
                ),
                scale=Vector3(
                    x=0.0,
                    y=0.0,
                    z=0.01,
                ),
                color=ColorRGBA(r=1.0, g=1.0, b=1.0, a=0.7),
                text=f"{self._row_id}{self._col_id}\t{int(self._points)}",
            )

            return marker

        def slice_and_get_points(self, points: np.array):
            xrange = (
                self._center_coord.x - (self._size.x / 2) * 0.9,
                self._center_coord.x + (self._size.x / 2) * 0.9,
            )
            yrange = (
                self._center_coord.y - (self._size.y / 2) * 0.9,
                self._center_coord.y + (self._size.y / 2) * 0.9,
            )
            zrange = (
                self._center_coord.z - (self._size.z / 2) * 0.9,
                self._center_coord.z + (self._size.z / 2) * 0.9,
            )

            points_in_grid: np.ndarray = PointCloudTransformer.ROI_Color_filter(
                points,
                ROI=True,
                x_range=xrange,
                y_range=yrange,
                z_range=zrange,
                rgb=False,
            )

            return points_in_grid

    class Line(object):
        def __init__(self, id: str):
            self._id = id
            self._grids = []

        def append(self, grid, by: str):
            grid: GridManager.Grid
            if not isinstance(grid, GridManager.Grid):
                raise TypeError("grid must be an instance of GridManager.Grid")

            self._grids.append(grid)

            self._grids = self.sort(by=by)

            return self._grids

        def set(self, grids, by: str):
            grids: List[GridManager.Grid]

            if not all(isinstance(grid, GridManager.Grid) for grid in grids):
                raise TypeError("grids must be a list of GridManager.Grid instances")

            self._grids = grids

            self._grids = self.sort(by=by)

            return self._grids

        def sort(self, by: str):
            """
            :param by: "row" or "col"
            """

            if by not in ["row", "col"]:
                raise ValueError("Sort parameter 'by' must be either 'row' or 'col'")

            self._grids.sort(key=lambda grid: getattr(grid, by))
            return self._grids

        @property
        def id(self):
            return self._id

        @property
        def grids(self):
            return self._grids

        def __lt__(self, other):
            if not isinstance(other, GridManager.Line):
                return NotImplemented
            return ord(self._id) < ord(other._id)

        def __gt__(self, other):
            if not isinstance(other, GridManager.Line):
                return NotImplemented
            return ord(self._id) > ord(other._id)

        def __le__(self, other):
            if not isinstance(other, GridManager.Line):
                return NotImplemented
            return ord(self._id) <= ord(other._id)

        def __ge__(self, other):
            if not isinstance(other, GridManager.Line):
                return NotImplemented
            return ord(self._id) >= ord(other._id)

        def __eq__(self, other):
            if not isinstance(other, GridManager.Line):
                return NotImplemented
            return ord(self._id) == ord(other._id)

        def __ne__(self, other):
            if not isinstance(other, GridManager.Line):
                return NotImplemented
            return ord(self._id) != ord(other._id)

    def __init__(self, node: Node, *args, **kwargs):
        super().__init__(node, *args, **kwargs)

        # >>> Load Files >>>
        fcn_package_path = get_package_share_directory("fcn_network")

        resource_path = os.path.join(
            fcn_package_path, "../ament_index/resource_index/packages"
        )

        grid_data_path = kwargs["grid_data_file"]
        if not os.path.isfile(grid_data_path):
            grid_data_path = os.path.join(resource_path, grid_data_path)

        with open(grid_data_path, "r") as f:
            self._grid_data = json.load(f)
        # <<< Load Files

        # >>> Initialize the Grid Manager >>>
        self._grids, self._rows, self._cols = self.create_grid()

    def create_grid(self) -> Tuple[List[Grid], List[Line], List[Line]]:
        rows = self._grid_data["rows"]
        cols = self._grid_data["columns"]

        grid_identifier = self._grid_data["grid_identifier"]

        grid_size = Vector3(
            x=grid_identifier["grid_size"]["x"],
            y=grid_identifier["grid_size"]["y"],
            z=grid_identifier["grid_size"]["z"],
        )
        start_center_coord = Point(
            x=grid_identifier["start_center_coord"]["x"],
            y=grid_identifier["start_center_coord"]["y"],
            z=grid_identifier["start_center_coord"]["z"],
        )
        point_threshold = grid_identifier["point_threshold"]

        grids = []
        cols_line = [GridManager.Line(id=str(col)) for col in cols]  # e.g. '0'
        rows_line = [GridManager.Line(id=str(row)) for row in rows]  # e.g. 'A'

        for r, row_line, row in zip(range(len(rows)), rows_line, rows):
            for c, col_line, col in zip(range(len(cols)), cols_line, cols):
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
                    threshold=point_threshold,
                )

                self._node.get_logger().info(
                    f"Append grid {row}{col} at {center_coord.x:.3f}, {center_coord.y:.3f}, {center_coord.z:.3f}"
                )

                # Append grid to row and column, and grids
                row_line.append(grid, by="col")
                col_line.append(grid, by="row")
                grids.append(grid)

        return grids, rows_line, cols_line

    @property
    def rows(self):
        return self._rows

    @property
    def cols(self):
        return self._cols

    @property
    def grids(self):
        return self._grids

    def get_grid(self, row: str, col: int):
        """
        :param row: Row ID (e.g. 'A')
        :param col: Column ID (e.g. 0)
        :return: Grid object
        """

        for grid in self._grids:
            if grid.row == row and grid.col == col:
                return grid

        raise ValueError(f"Grid {row}{col} not found")

    def get_grid_data(self):
        return self._grid_data

    def get_colums_length(self):
        return len(self._grid_data["columns"])


class FCN_Integration_Manager(Manager):
    def __init__(self, node: Node, *args, **kwargs):
        super().__init__(node, *args, **kwargs)

        # >>> Service Clients
        self._fcn_client = self._node.create_client(FCNRequest, "/fcn_request")
        self._fcn_occupied_client = self._node.create_client(
            FCNOccupiedRequest, "/fcn_occupied_request"
        )
        # <<< Service Clients

    def run(
        self, target_cls: str, last_target_col: int = -1
    ) -> Tuple[FCNRequest.Response, FCNOccupiedRequest.Response]:
        """
        :param target_cls: Target class to be detected. e.g. 'bottle_1'
        """

        # >>> STEP 1. FCN Request >>>
        fcn_request = FCNRequest.Request()
        fcn_request.target_cls = target_cls
        fcn_request.last_target_col = last_target_col
        fcn_response: FCNRequest.Response = self._fcn_client.call(fcn_request)

        if fcn_response is None:
            self._node.get_logger().warn("FCN response is None.")
            return None, None

        if fcn_response.target_col == -1:
            self._node.get_logger().warn("FCN response target_col is -1.")
            return fcn_response, None

        # >>> STEP 2. FCN Occupied Request >>>
        fcn_occupied_request = FCNOccupiedRequest.Request()
        fcn_occupied_request.empty_cols = fcn_response.empty_cols.tolist()
        fcn_occupied_request.target_col = fcn_response.target_col

        fcn_occupied_response: FCNOccupiedRequest.Response = (
            self._fcn_occupied_client.call(fcn_occupied_request)
        )

        if fcn_occupied_response is None:
            self._node.get_logger().warn("FCN occupied response is None.")
            return fcn_response, None

        if fcn_occupied_response.moving_row == "Z":
            self._node.get_logger().warn("FCN occupied response moving_row is Z.")
            return fcn_response, None

        return fcn_response, fcn_occupied_response
