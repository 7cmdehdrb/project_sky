import sys
import os
import numpy as np
import time
from scipy.spatial.transform import Rotation as R

import asyncio
import websockets

# Isaac Sim Python API
from omni.isaac.core.prims import XFormPrim


class QuaternionAngle:
    @staticmethod
    def euler_from_quaternion(quaternion):
        """
        In: [x, y, z, w], Out: roll, pitch, yaw
        """
        x = quaternion[0]
        y = quaternion[1]
        z = quaternion[2]
        w = quaternion[3]

        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = np.arctan2(sinr_cosp, cosr_cosp)

        sinp = 2 * (w * y - z * x)
        pitch = np.arcsin(sinp)

        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = np.arctan2(siny_cosp, cosy_cosp)

        return roll, pitch, yaw

    @staticmethod
    def quaternion_from_euler(roll, pitch, yaw):
        """
        In: roll, pitch, yaw, Out: x, y, z, w
        """
        qx = np.sin(roll / 2) * np.cos(pitch / 2) * np.cos(yaw / 2) - np.cos(
            roll / 2
        ) * np.sin(pitch / 2) * np.sin(yaw / 2)
        qy = np.cos(roll / 2) * np.sin(pitch / 2) * np.cos(yaw / 2) + np.sin(
            roll / 2
        ) * np.cos(pitch / 2) * np.sin(yaw / 2)
        qz = np.cos(roll / 2) * np.cos(pitch / 2) * np.sin(yaw / 2) - np.sin(
            roll / 2
        ) * np.sin(pitch / 2) * np.cos(yaw / 2)
        qw = np.cos(roll / 2) * np.cos(pitch / 2) * np.cos(yaw / 2) + np.sin(
            roll / 2
        ) * np.sin(pitch / 2) * np.sin(yaw / 2)

        return qx, qy, qz, qw

    @staticmethod
    def euler_from_rotation_matrix(rotation_matrix):
        """
        In: rotation_matrix, Out: roll, pitch, yaw
        """
        # 회전 행렬을 scipy의 Rotation 객체로 변환
        rotation = R.from_matrix(rotation_matrix)

        # Roll, Pitch, Yaw 추출 (XYZ 순서)
        roll, pitch, yaw = rotation.as_euler("xyz", degrees=False)

        return roll, pitch, yaw

    @staticmethod
    def rotation_matrix_from_euler(roll, pitch, yaw):
        """
        In: roll, pitch, yaw, Out: rotation_matrix
        """
        rotation = R.from_euler("xyz", [roll, pitch, yaw], degrees=False)
        return rotation.as_matrix()

    @staticmethod
    def create_transform_matrix(translation, rotation):
        """
        In: translation, rotation, Out: transform_matrix
        """
        transform_matrix = np.eye(4)
        transform_matrix[:3, :3] = rotation
        transform_matrix[:3, 3] = translation
        return transform_matrix

    @staticmethod
    def transform_realsense_to_ros(transform_matrix: np.array) -> np.array:
        """
        Realsense 좌표계를 ROS 좌표계로 변환합니다.

        Realsense 좌표계:
            X축: 이미지의 가로 방향 (오른쪽으로 증가).
            Y축: 이미지의 세로 방향 (아래쪽으로 증가).
            Z축: 카메라 렌즈가 바라보는 방향 (깊이 방향).

        ROS 좌표계:
            X축: 앞으로 나아가는 방향.
            Y축: 왼쪽으로 이동하는 방향.
            Z축: 위로 이동하는 방향.

        Args:
            transform_matrix (np.ndarray): 4x4 변환 행렬.

        Returns:
            np.ndarray: 변환된 4x4 변환 행렬 (ROS 좌표계 기준).
        """
        if transform_matrix.shape != (4, 4):
            raise ValueError("Input transformation matrix must be a 4x4 matrix.")

        # Realsense에서 ROS로 좌표계를 변환하는 회전 행렬
        realsense_to_ros_rotation = np.array(
            [[0, 0, 1], [-1, 0, 0], [0, -1, 0]]  # X -> Z  # Y -> -X  # Z -> -Y
        )

        # 변환 행렬의 분해
        rotation = transform_matrix[:3, :3]  # 3x3 회전 행렬
        translation = transform_matrix[:3, 3]  # 3x1 평행 이동 벡터

        # 좌표계 변환
        rotation_ros = realsense_to_ros_rotation @ rotation
        translation_ros = realsense_to_ros_rotation @ translation

        # 새로운 변환 행렬 구성
        transform_matrix_ros = np.eye(4)
        transform_matrix_ros[:3, :3] = rotation_ros
        transform_matrix_ros[:3, 3] = translation_ros

        return transform_matrix_ros

    @staticmethod
    def invert_transformation(matrix):
        """
        Inverts a 4x4 transformation matrix.

        Parameters:
            matrix (np.ndarray): A 4x4 transformation matrix representing A > B.

        Returns:
            np.ndarray: The inverted transformation matrix representing B > A.
        """
        if matrix.shape != (4, 4):
            raise ValueError("Input matrix must be a 4x4 matrix.")

        # Extract rotation and translation components
        rotation = matrix[:3, :3]
        translation = matrix[:3, 3]

        # Invert the rotation (transpose for orthogonal matrix)
        rotation_inv = rotation.T

        # Invert the translation
        translation_inv = -np.dot(rotation_inv, translation)

        # Construct the inverted transformation matrix
        inverted_matrix = np.eye(4)
        inverted_matrix[:3, :3] = rotation_inv
        inverted_matrix[:3, 3] = translation_inv

        return inverted_matrix


class Area:
    def __init__(self, position: np.array, id: str):
        self.id = id
        self.position = position

        self.z_data = {
            "/World/alive": 0.73773,
            "/World/coca_cola": 0.7215,
            "/World/cyder": 0.73193,
            "/World/green_tea": 0.76842,
            "/World/yello_peach_transformed": 0.7238,
            "/World/yello_smoothie_transformed": 0.75301,
            "/World/Mug_2": 0.65974,
            "/World/Mug_3": 0.66,
            "/World/Mug_4": 0.66083,
            "/World/Cup_1": 0.66101,
            "/World/Cup_2": 0.66138,
            "/World/Cup_4": 0.66195,
        }

    def get_random_orientation(self, is_scanned=False):
        x, y, z, w = QuaternionAngle.quaternion_from_euler(
            np.deg2rad(90.0) if is_scanned else 0.0,
            0.0,
            # 0.0,
            np.random.uniform(0.0, 2.0 * np.pi),
        )
        return np.array([w, x, y, z])

    def get_random_position(self, id: str):
        noise = 0.01
        random_noise = np.array(
            [
                np.random.uniform(-noise, noise),
                np.random.uniform(-noise, noise),
                0.0,
            ]
        )

        new_position = self.position + random_noise
        new_position[2] = self.z_data[id]
        return new_position

    def get_random_pose(self, id: str, is_scanned=False):
        return self.get_random_position(id=id), self.get_random_orientation(
            is_scanned=is_scanned
        )

    @staticmethod
    def select_random_area(areas: dict, num: int):
        if num > len(areas):
            raise ValueError("The number of areas should be less than the total areas.")

        return np.random.choice(list(areas.values()), num, replace=False)


class SegmentObject:
    def __init__(self, id: str):
        self.id = id
        self.prims = XFormPrim(self.id)
        self.is_scanned = not ("Mug" in self.id or "Cup" in self.id)

        self.initial_position = np.array(
            [
                np.random.uniform(100, 500),
                np.random.uniform(100, 500),
                np.random.uniform(100, 500),
            ]
        )

    def get_pose(self):
        position, orientation = self.prims.get_world_pose()
        return position, orientation

    def reset_object_pose(self):
        self.prims.set_world_pose(position=self.initial_position)

    def set_pose(self, position, orientation):
        self.prims.set_world_pose(position=position, orientation=orientation)
        return True

    @staticmethod
    def select_random_objects(objects: dict, num: int):
        if num > len(objects):
            raise ValueError(
                "The number of objects should be less than the total objects."
            )

        return np.random.choice(list(objects.values()), num, replace=False)


class ID_Publisher:
    def __init__(self):
        self.unique_id = "/World/id"
        self.prims = XFormPrim(self.unique_id)

    def set_id(self, id: float):
        self.prims.set_world_pose(position=np.array([-500.0, id, -500.0]))

    def get_id(self):
        position, _ = self.prims.get_world_pose()
        return int(position[1])


# Main
async def run(areas: dict, objects: dict, id_publisher: ID_Publisher):
    uri = "ws://localhost:8765"

    try:
        async with websockets.connect(uri) as websocket:
            await websocket.send("0")

            # Reset all objects pose
            for obj in objects.values():
                obj: SegmentObject
                obj.reset_object_pose()

            num = np.random.randint(3, 8)

            # reset all object pose
            for obj in objects.values():
                obj: SegmentObject
                obj.reset_object_pose()

            # Select random areas
            selected_areas = Area.select_random_area(areas, num=num)

            # Select random objects
            selected_objects = SegmentObject.select_random_objects(objects, num=num)

            # Set random pose for selected objects
            for area, obj in zip(selected_areas, selected_objects):
                area: Area
                obj: SegmentObject

                position, orientation = area.get_random_pose(
                    id=obj.id, is_scanned=obj.is_scanned
                )
                obj.set_pose(position, orientation)

            await websocket.send("1")

    except Exception as e:
        print(f"Error: {e}")
        await websocket.close()


if __name__ == "__main__":
    # Area Setting (A1 ~ E3)

    # Define areas variables
    rows = ["A", "B", "C", "D", "E"]
    cols = ["0", "1", "2"]
    areas = {}

    # Create areas with ids
    area_ids = []
    for row in rows:
        for col in cols:
            area_ids.append(f"{row}{col}")

    # Create area objects
    for area_id in area_ids:
        # Isaac Sim Prim Object
        prim_path = f"/World/{area_id}"
        prim = XFormPrim(prim_path)

        # World Frame에서 위치(Translation)와 회전(Rotation) 가져오기
        position, _ = prim.get_world_pose()
        areas[area_id] = Area(position, area_id)

    # Object Setting

    object_ids = [
        "/World/alive",
        "/World/coca_cola",
        "/World/cyder",
        "/World/green_tea",
        "/World/yello_peach_transformed",
        "/World/yello_smoothie_transformed",
        "/World/Mug_2",
        "/World/Mug_3",
        "/World/Mug_4",
        "/World/Cup_1",
        "/World/Cup_2",
        "/World/Cup_4",
    ]
    objects = {}

    for idx, object_id in enumerate(object_ids):
        # Isaac Sim Prim Object
        objects[object_id] = SegmentObject(object_id)

    id_publisher = ID_Publisher()

    asyncio.run(run(areas=areas, objects=objects, id_publisher=id_publisher))
    # run(areas=areas, objects=objects, id_publisher=id_publisher)
