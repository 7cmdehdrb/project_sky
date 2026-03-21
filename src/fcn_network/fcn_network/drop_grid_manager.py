import os
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_system_default

from enum import Enum
from typing import Optional, List, Tuple
import numpy as np

# ROS2 Messages
from std_msgs.msg import Header
from sensor_msgs.msg import PointCloud2
from visualization_msgs.msg import MarkerArray
from std_msgs.msg import Float32MultiArray
from geometry_msgs.msg import Point, Vector3
from custom_msgs.srv import GetNextDropCell

# 기존 모듈에서 GridManager와 GridCell이 임포트되어 있다고 가정합니다.
from fcn_network.grid_manager import GridManager, GridCell


class DropPriority(Enum):
    ROW_FIRST = 1  # ROW 고정, COL 순회 (예: A1, A2, A3 -> B1, B2, B3)
    COL_FIRST = 2  # COL 고정, ROW 순회 (예: A1, B1, C1 -> A2, B2, C2)


class DropDirection(Enum):
    FORWARD = 1  # 정순 (예: A -> Z, 1 -> 10)
    REVERSE = 2  # 역순 (예: Z -> A, 10 -> 1)


class DropGridManager(GridManager):
    """
    포인트 클라우드가 아닌, 지정된 순서 규칙에 따라 순차적으로
    그리드 셀에 Drop(점유)을 수행하는 매니저 클래스.
    """

    def __init__(
        self,
        resource_path: str,
        priority: DropPriority = DropPriority.ROW_FIRST,
        row_dir: DropDirection = DropDirection.FORWARD,
        col_dir: DropDirection = DropDirection.FORWARD,
    ):
        super().__init__(resource_path)

        # 독립적인 Drop 규칙 설정
        self._priority = priority
        self._row_dir = row_dir
        self._col_dir = col_dir

        self._drop_sequence: List[Tuple[str, int]] = []
        self._current_drop_index = 0

        self._generate_drop_sequence()

    def set_drop_rule(
        self, priority: DropPriority, row_dir: DropDirection, col_dir: DropDirection
    ):
        """ROW와 COL의 방향 및 우선순위를 각각 별도로 재설정하고 시퀀스를 갱신합니다."""
        self._priority = priority
        self._row_dir = row_dir
        self._col_dir = col_dir

        self._generate_drop_sequence()
        self._current_drop_index = 0

    def _generate_drop_sequence(self):
        """
        ROW와 COL에 대해 각각 독립적으로 정순/역순을 적용하여 전체 탐색 순서를 생성합니다.
        """
        # 1. ROW와 COL 각각의 독립적인 방향 리스트 생성
        rows = (
            self._rows
            if self._row_dir == DropDirection.FORWARD
            else list(reversed(self._rows))
        )
        cols = (
            self._cols
            if self._col_dir == DropDirection.FORWARD
            else list(reversed(self._cols))
        )

        self._drop_sequence.clear()

        # 2. 우선순위에 따른 시퀀스 병합
        if self._priority == DropPriority.ROW_FIRST:
            for row in rows:
                for col in cols:
                    self._drop_sequence.append((row, col))
        else:  # COL_FIRST
            for col in cols:
                for row in rows:
                    self._drop_sequence.append((row, col))

    def get_next_drop_cell(self):
        """
        (3) 다음에 Drop해야 하는 Cell 객체를 리턴합니다.
        가져온 셀 객체에서 .row, .col, .position, .id 속성에 접근하여 사용할 수 있습니다.
        """
        for i in range(self._current_drop_index, len(self._drop_sequence)):
            row, col = self._drop_sequence[i]
            cell = self._cells[(row, col)]

            # 이미 점유된 상태가 아니라면 해당 셀 리턴
            if not cell.is_occupied:
                self._current_drop_index = i
                return cell

        return None  # 순회할 셀이 남아있지 않음

    def drop(self) -> bool:
        """
        (2) Drop 함수 콜 시, 결정된 순서에 의해 _is_occupied 필드를 True로 변경합니다.
        """
        target_cell = self.get_next_drop_cell()

        if target_cell is not None:
            # 부모 클래스의 _is_occupied 필드를 True로 강제 변경
            target_cell._is_occupied = True
            self._current_drop_index += 1  # 다음 타겟으로 인덱스 이동
            return True

        return False

    def update_occupancy(self, points=None):
        """
        [오버라이드] PointCloud 기반의 점유율 업데이트 기능을 차단합니다.
        """
        pass

    def get_marker_array(self, header, points=None):
        """
        [오버라이드] points 인자를 무시하고, 현재 수동으로 설정된 상태 기준으로만 마커를 리턴합니다.
        """
        return super().get_marker_array(header, points=None)

    # --- 부피 조건을 덮어 씌우기 위한 오버라이딩 ---
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

        return marker_array
