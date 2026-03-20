# Python
import os
import sys
import json
import time
import threading
import numpy as np
from enum import Enum
from typing import Optional, Dict, List, Tuple

# ROS2
import rclpy
from rclpy.node import Node
from rclpy.time import Time
from rclpy.duration import Duration
from rclpy.qos import QoSProfile, qos_profile_system_default

# ROS2 Messages
from std_msgs.msg import Header, ColorRGBA
from geometry_msgs.msg import Point, Vector3, Pose, Quaternion
from sensor_msgs.msg import *
from nav_msgs.msg import *
from visualization_msgs.msg import Marker, MarkerArray
from builtin_interfaces.msg import Duration as BuiltinDuration

# TF
from tf2_ros import *


class GridCell:
    """개별 그리드 셀의 정보와 상태, 시각화 마커 생성을 담당하는 클래스"""

    def __init__(
        self,
        row_id: str,
        col_id: int,
        center_coord: Point,
        size: Vector3,
        threshold: int,
    ):
        self._row_id = row_id
        self._col_id = col_id
        self._center_coord = center_coord
        self._size = size
        self._threshold = threshold

        self._points_count = 0
        self._is_occupied = False

        self._mean = np.zeros(3)
        self._cov = np.zeros((3, 3))
        self._scale = np.zeros(3)

    # --- Getter (조건 3) ---
    @property
    def row(self) -> str:
        return self._row_id

    @property
    def col(self) -> int:
        return self._col_id

    @property
    def id(self) -> str:
        """통합 ID 리턴 (예: 'A' + 0 -> 'A0')"""
        return f"{self._row_id}{self._col_id}"

    @property
    def position(self) -> Point:
        return self._center_coord

    @property
    def size(self) -> Vector3:
        return self._size

    @property
    def is_occupied(self) -> bool:
        return self._is_occupied

    @property
    def point_with_covariance(self) -> Tuple[np.ndarray, np.ndarray]:
        """현재 셀의 평균 위치와 공분산을 넘파이 배열로 리턴합니다."""
        return self._mean, self._cov

    # --- 핵심 로직 ---
    def update_occupancy(self, points: np.ndarray):
        """넘파이 배열을 받아 현재 셀 영역 내의 점 개수를 파악하고 점유 여부를 갱신합니다."""
        # 셀 크기의 90% 영역만 관심 영역(ROI)으로 설정 (기존 로직 유지)
        x_min = self._center_coord.x - (self._size.x / 2) * 0.9
        x_max = self._center_coord.x + (self._size.x / 2) * 0.9
        y_min = self._center_coord.y - (self._size.y / 2) * 0.9
        y_max = self._center_coord.y + (self._size.y / 2) * 0.9
        z_min = self._center_coord.z - (self._size.z / 2) * 0.9
        z_max = self._center_coord.z + (self._size.z / 2) * 0.9

        # Numpy Boolean Masking을 이용한 초고속 필터링
        mask = (
            (points[:, 0] >= x_min)
            & (points[:, 0] <= x_max)
            & (points[:, 1] >= y_min)
            & (points[:, 1] <= y_max)
            & (points[:, 2] >= z_min)
            & (points[:, 2] <= z_max)
        )

        self._points_count = np.sum(mask)
        self._is_occupied = self._points_count > self._threshold

        # --- 추가된 로직: 통계량 (Mean, Covariance) 계산 ---
        if self._is_occupied and self._points_count > 0:
            # 1. 마스크를 적용하여 ROI 내의 포인트만 추출
            filtered_points = points[mask]

            # 2. XYZ/XYZI/XYZRGB 범용성 처리: 앞의 3개 컬럼(X, Y, Z)만 추출
            xyz_points = filtered_points[:, :3]

            # 3. 평균 (Mean) 계산 -> 형태: (3,)
            self._mean = np.mean(xyz_points, axis=0)

            self._scale = self.get_volume(xyz_points)

            # 4. 공분산 (Covariance) 계산 -> 형태: (3, 3)
            if self._points_count > 1:
                # rowvar=False: 행(row)을 관측치로, 열(column)을 변수(x, y, z)로 취급
                self._cov = np.cov(xyz_points, rowvar=False)
            else:
                # 점이 1개뿐이면 분산을 구할 수 없으므로 0 행렬 처리
                self._cov = np.zeros((3, 3))
        else:
            # ROI 내에 점이 전혀 없을 경우 필드 초기화
            self._mean = np.zeros(3)
            self._cov = np.zeros((3, 3))
            self._scale = np.zeros(3)

    def get_volume(self, np_points: np.ndarray) -> Marker:
        # 1. Q1(25%)과 Q3(75%) 계산
        Q1 = np.percentile(np_points, 25, axis=0)
        Q3 = np.percentile(np_points, 75, axis=0)

        # 2. IQR (Interquartile Range) 계산
        IQR = Q3 - Q1

        # 3. 정상 범위 설정 (보통 1.5를 곱하지만, PCD 특성에 따라 2.0 등으로 조절 가능)
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        # 4. 각 축별로 정상 범위 안에 있는 포인트만 마스킹
        # (세 축 모두 정상 범위 안에 있는 포인트만 살림)
        mask = np.all((np_points >= lower_bound) & (np_points <= upper_bound), axis=1)
        filtered_pcd = np_points[mask]

        # 5. 필터링된 데이터에서 실제 사이즈 계산
        scale = np.max(filtered_pcd, axis=0) - np.min(filtered_pcd, axis=0)

        return scale

    def get_marker(self, header: Header) -> Marker:
        """현재 상태에 맞는 ROS Marker 객체를 리턴합니다."""
        # ID 생성 로직 (예: 'A' -> 65. ((65-64)*10) + 0 = 10)
        marker_id = ((ord(self._row_id) - 64) * 10) + self._col_id

        # 차있으면 빨간색(r=1.0), 비어있으면 초록색(g=1.0)
        r_color = 1.0 if self._is_occupied else 0.0
        g_color = 0.0 if self._is_occupied else 1.0

        marker = Marker(
            header=header,
            ns="grid_cells",
            id=marker_id,
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
            color=ColorRGBA(r=r_color, g=g_color, b=0.0, a=0.5),
        )
        return marker

    def get_text_marker(self, header: Header) -> Marker:
        """셀 ID를 표시하는 텍스트 마커를 리턴합니다."""
        marker_id = (
            ((ord(self._row_id) - 64) * 10) + self._col_id + 1000
        )  # 텍스트 마커는 ID offset

        text_marker = Marker(
            header=header,
            ns="grid_cell_labels",
            id=marker_id,
            type=Marker.TEXT_VIEW_FACING,
            action=Marker.ADD,
            pose=Pose(
                position=Point(
                    x=self._center_coord.x,
                    y=self._center_coord.y,
                    z=self._center_coord.z
                    + (self._size.z / 2)
                    + 0.03,  # 셀 위에 약간 띄워서 표시
                ),
                orientation=Quaternion(x=0.0, y=0.0, z=0.0, w=1.0),
            ),
            scale=Vector3(x=0.05, y=0.05, z=0.05),  # 텍스트 크기
            color=ColorRGBA(r=1.0, g=1.0, b=1.0, a=1.0),  # 흰색
        )

        text_marker.text = f"{self.id}"
        # self.id + " " + str(cnt_k)  # 예: 'A0 15' (셀 ID + 점 개수)
        return text_marker

    def get_volume_marker(self, header: Header) -> Marker:
        """셀의 점유 상태에 따른 볼륨 마커를 리턴합니다."""
        marker_id = (
            ((ord(self._row_id) - 64) * 10) + self._col_id + 2000
        )  # 볼륨 마커는 ID offset

        volume_marker = Marker(
            header=header,
            ns="grid_volume",
            id=marker_id,
            type=Marker.CUBE,
            action=Marker.ADD,
            pose=Pose(
                position=Point(
                    x=float(self._mean[0]),
                    y=float(self._mean[1]),
                    z=float(self._mean[2]),
                ),
                orientation=Quaternion(x=0.0, y=0.0, z=0.0, w=1.0),
            ),
            scale=Vector3(
                x=float(self._scale[0]),
                y=float(self._scale[1]),
                z=float(self._scale[2]),
            ),
            color=ColorRGBA(r=0.0, g=0.0, b=1.0, a=0.5),  # 파란색 (볼륨 마커는 항상 파란색으로 표시, 점유 여부와 무관하게)
        )
        
        return volume_marker


class GridManager:
    """전체 그리드 셀을 생성하고 관리하는 총괄 클래스"""

    def __init__(self, resource_path: str):
        # 조건 1: 무조건 절대경로만 받음 / 파일 없을 시 예외 발생
        if not os.path.isabs(resource_path):
            raise ValueError(
                f"[Error] resource_path는 반드시 절대경로여야 합니다. 입력된 경로: {resource_path}"
            )

        if not os.path.isfile(resource_path):
            raise FileNotFoundError(
                f"[Error] 해당 경로에 JSON 파일이 존재하지 않습니다. 시스템을 셧다운합니다. 경로: {resource_path}"
            )

        # JSON 읽기
        with open(resource_path, "r") as f:
            self._config = json.load(f)

        self._rows: List[str] = self._config["rows"]
        self._cols: List[int] = self._config["columns"]

        # 조건 2: 셀들을 튜플 키 (row_id, col_id) 딕셔너리에 저장하여 O(1) 접근 확보
        self._cells: Dict[Tuple[str, int], GridCell] = {}
        self._initialize_cells()

    def _initialize_cells(self):
        """JSON 데이터를 바탕으로 모든 GridCell 인스턴스를 생성하여 딕셔너리에 저장합니다."""
        grid_info = self._config["grid_identifier"]

        size = Vector3(**grid_info["grid_size"])
        start = Point(**grid_info["start_center_coord"])
        threshold = grid_info["point_threshold"]

        for r_idx, row_id in enumerate(self._rows):
            for c_idx, col_id in enumerate(self._cols):
                # 기준 위치로부터 각 셀의 중심 좌표 계산 (기존 로직 유지)
                center_coord = Point(
                    x=start.x + (size.x * r_idx),
                    y=start.y - (size.y * c_idx),
                    z=start.z,
                )

                cell = GridCell(
                    row_id=row_id,
                    col_id=col_id,
                    center_coord=center_coord,
                    size=size,
                    threshold=threshold,
                )

                self._cells[(row_id, col_id)] = cell

    # --- 추가 조건 2: 특정 셀 객체 가져오기 ---
    def get_cell(self, row_id: str, col_id: int) -> GridCell:
        """ROW, COL 아이디를 받아 해당하는 GridCell 인스턴스를 리턴합니다."""
        if (row_id, col_id) not in self._cells:
            raise KeyError(f"존재하지 않는 셀입니다. 요청한 셀: {row_id}{col_id}")
        return self._cells[(row_id, col_id)]

    # --- 조건 3: MarkerArray 리턴 및 배열 검증 ---
    def get_marker_array(
        self, header: Header, points: Optional[np.ndarray] = None
    ) -> MarkerArray:
        """
        포인트 클라우드 데이터를 받아 각 셀의 점유 상태를 갱신하고 MarkerArray를 리턴합니다.
        points가 None일 경우 갱신 없이 현재(기본) 상태의 마커들만 리턴합니다.
        """
        if points is not None:
            # Type 및 Shape 깐깐하게 검증 (N x 3 배열 형태 확인)
            assert isinstance(
                points, np.ndarray
            ), "points 인자는 반드시 numpy.ndarray 타입이거나 None이어야 합니다."
            assert (
                len(points.shape) == 2 and points.shape[1] >= 3
            ), "points는 최소 XYZ (N, 3) 이상의 형태여야 합니다."

            # 각 셀마다 점유 상태 갱신
            for cell in self._cells.values():
                cell.update_occupancy(points)

        # 전체 마커 취합 후 리턴
        marker_array = MarkerArray()
        for cell in self._cells.values():
            marker_array.markers.append(cell.get_marker(header))
            marker_array.markers.append(cell.get_text_marker(header))
            if cell.is_occupied:
                marker_array.markers.append(cell.get_volume_marker(header))

        return marker_array

    # --- 조건 4: 탐색 함수 ---
    def get_frontmost_col_by_row(self) -> Dict[str, Optional[int]]:
        """ROW 기준으로 가장 먼저(0부터) 차 있는 COL 번호를 딕셔너리로 리턴합니다."""
        result = {}
        for row in self._rows:
            result[row] = None
            for col in self._cols:
                if self._cells[(row, col)].is_occupied:
                    result[row] = col
                    break  # 가장 앞의 하나를 찾았으므로 해당 행은 탐색 종료
        return result

    def get_frontmost_row_by_col(self) -> Dict[int, Optional[str]]:
        """COL 기준으로 가장 먼저('A'부터) 차 있는 ROW 문자를 딕셔너리로 리턴합니다."""
        result = {}
        for col in self._cols:
            result[col] = None
            for row in self._rows:
                if self._cells[(row, col)].is_occupied:
                    result[col] = row
                    break  # 가장 앞의 하나를 찾았으므로 해당 열은 탐색 종료
        return result

    def update_occupancy(self, points: np.ndarray):
        """외부에서 포인트 클라우드 데이터를 받아 모든 셀의 점유 상태를 갱신하는 함수입니다."""
        for cell in self._cells.values():
            cell.update_occupancy(points)
