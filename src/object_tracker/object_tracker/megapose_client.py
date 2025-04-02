# Python
import io
import json
import socket
import struct
import time

# OpenCV
import cv2

# NumPy
import numpy as np

# ROS2
import rclpy
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import QoSProfile, qos_profile_system_default
from rclpy.time import Time

# ROS2 Messages
from custom_msgs.msg import BoundingBox, BoundingBoxMultiArray
from geometry_msgs.msg import *
from nav_msgs.msg import *
from sensor_msgs.msg import *
from std_msgs.msg import *
from visualization_msgs.msg import *

# TF
from tf2_ros import *

# Custom Packages
from base_package.manager import ObjectManager


class MegaPoseClient(object):
    """Socket client for MegaPose server."""

    class ServerMessage:
        GET_POSE = "GETP"
        RET_POSE = "RETP"
        GET_VIZ = "GETV"
        RET_VIZ = "RETV"
        SET_INTR = "INTR"
        GET_SCORE = "GSCO"
        RET_SCORE = "RSCO"
        SET_SO3_GRID_SIZE = "SO3G"
        GET_LIST_OBJECTS = "GLSO"
        RET_LIST_OBJECTS = "RLSO"
        ERR = "RERR"
        OK = "OKOK"

    def __init__(self, node: Node, *args, **kwargs):
        """
        kwargs:
            --host: str
            --port: int
            --use_depth: bool
            --score_threshold: float (Score threshold for segmentation)
            --refiner_iterations: int
        """
        self._node = node

        # >>> Socket >>>
        self._SERVER_HOST = kwargs.get("host", "127.0.0.1")
        self._SERVER_PORT = kwargs.get("port", 5555)
        self._SERVER_OPERATION_CODE_LENGTH = 4

        self._socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._socket.connect((self._SERVER_HOST, self._SERVER_PORT))
        # >>> Socket >>>

        # >>> ROS2 >>>
        self._camera_info_subscriber = self._node.create_subscription(
            CameraInfo,
            "/camera/camera1/color/camera_info",
            self.camera_info_callback,
            qos_profile=qos_profile_system_default,
        )
        # <<< ROS2 <<<

        # >>> Data >>>
        self._is_configured = False
        self._desired_image_size = (640, 480)
        self._last_time = self._node.get_clock().now()
        self._avilable_objects = self.send_list_objects_request(self._socket)
        # <<< Data <<<
        
    @property
    def available_objects(self):
        return self._avilable_objects

    def camera_info_callback(self, msg: CameraInfo):
        if self._is_configured is True:
            return None

        K = np.array(msg.k).reshape(3, 3)

        if (
            msg.width == self._desired_image_size[0]
            and msg.height == self._desired_image_size[1]
        ):
            self._node.get_logger().info(
                f"Image size is same as desired size: {msg.width}, {msg.height}"
            )
            self._node.get_logger().info(f"\nOriginal K: {K}")

        elif msg.width == 1280 and msg.height == 720:
            self._node.get_logger().info(
                f"Image size is same as server size: {msg.width}, {msg.height}"
            )
            self._node.get_logger().info(f"Calibrating K: {K}")
            self._node.get_logger().info("")

            K = K.copy()

            h, w = 720, 1280
            crop_w, crop_h = 640, 480
            start_x = int((w - crop_w) // 2.05)
            start_y = int((h - crop_h) // 2.7)

            K[0, 2] -= start_x
            K[1, 2] -= start_y

        else:
            # Prevent setting intrinsics
            self._node.get_logger().warn(
                f"Invalid image size. Cannot set intrinsics: {msg.width}, {msg.height}"
            )
            return None

        self.set_intrinsics(
            K=K,
            image_size=(480, 640),
        )

        self._node.get_logger().info("Set intrinsics successfully.")
        self._is_configured = True

    def set_intrinsics(self, K: np.ndarray, image_size: tuple):
        if not self._is_configured:
            response = self.send_intrinsics_request(
                self._socket,
                K=K,
                image_size=image_size,
            )
            self._is_configured = response

    def send_message(self, sock: socket.socket, code: str, data: bytes):
        msg_length = struct.pack(">I", len(data))
        sock.sendall(msg_length + code.encode("UTF-8") + data)

    def receive_message(self, sock: socket.socket):
        msg_length = sock.recv(4)
        length = struct.unpack(">I", msg_length)[0]
        code = sock.recv(self._SERVER_OPERATION_CODE_LENGTH).decode("UTF-8")
        data = sock.recv(length)
        return code, io.BytesIO(data)

    def pack_string(self, data: str) -> bytes:
        encoded = data.encode("utf-8")
        length = struct.pack(">I", len(encoded))
        return length + encoded

    def read_string(self, buffer: io.BytesIO) -> str:
        length = struct.unpack(">I", buffer.read(4))[0]
        return buffer.read(length).decode("utf-8")

    def send_pose_request_rgbd(
        self, image: np.ndarray, depth: np.ndarray, json_data: dict
    ):
        if image.shape[-1] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # RGB image
        height, width, channels = image.shape
        img_shape_bytes = struct.pack(">3I", height, width, channels)
        img_bytes = image.tobytes()

        # JSON
        json_str = json.dumps(json_data)
        json_bytes = self.pack_string(json_str)

        # Depth image
        assert depth.dtype == np.uint16
        assert depth.shape == (height, width)
        depth_shape_bytes = struct.pack(">2I", height, width)  # only height & width
        endianness_byte = struct.pack("c", b">")  # big endian
        depth_bytes = depth.tobytes()

        # 최종 데이터 조합
        data = (
            img_shape_bytes
            + img_bytes
            + json_bytes
            + depth_shape_bytes
            + endianness_byte
            + depth_bytes
        )

        # Send and receive
        self.send_message(self._socket, MegaPoseClient.ServerMessage.GET_POSE, data)
        code, response_buffer = self.receive_message(self._socket)

        if code == MegaPoseClient.ServerMessage.RET_POSE:
            return json.loads(self.read_string(response_buffer))
        elif code == MegaPoseClient.ServerMessage.ERR:
            self._node.get_logger().warn(
                "Error from server:", self.read_string(response_buffer)
            )
        else:
            self._node.get_logger().error("Unknown response code:", code)
        return None

    def send_pose_request(self, image: np.ndarray, json_data: dict):
        # **(1) RGB 이미지를 전송할 수 있도록 BGR → RGB 변환**
        if image.shape[-1] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # **(2) 서버의 read_image 형식에 맞춰 (height, width, channels) 전송**
        height, width, channels = image.shape
        img_shape_bytes = struct.pack(">3I", height, width, channels)
        img_bytes = image.tobytes()

        # **(3) JSON 데이터를 직렬화**
        json_str = json.dumps(json_data)
        json_bytes = self.pack_string(json_str)

        # **(4) 최종 데이터 생성 (크기 + 이미지 + JSON)**
        data = img_shape_bytes + img_bytes + json_bytes

        # **(5) 서버에 데이터 전송**
        self.send_message(self._socket, MegaPoseClient.ServerMessage.GET_POSE, data)

        # **(6) 서버 응답 수신**
        code, response_buffer = self.receive_message(self._socket)
        if code == MegaPoseClient.ServerMessage.RET_POSE:
            json_str = self.read_string(response_buffer)
            decoded_json = json.loads(json_str)

            if len(decoded_json) > 0:
                return decoded_json

        elif code == MegaPoseClient.ServerMessage.ERR:
            print("Error from server:", self.read_string(response_buffer))
        else:
            print("Unknown response code:", code)

        return None

    def send_intrinsics_request(
        self, sock: socket.socket, K: np.ndarray, image_size: tuple
    ):
        """
        서버에 카메라의 내부 파라미터(K 행렬)와 이미지 크기를 설정하는 요청을 보낸다.

        :param sock: 열린 소켓 객체
        :param K: 3x3 카메라 내부 파라미터 행렬
        :param image_size: (height, width) 이미지 크기
        """
        # K 행렬에서 필요한 파라미터 추출
        px, py = K[0, 0], K[1, 1]  # 초점 거리 (f_x, f_y)
        u0, v0 = K[0, 2], K[1, 2]  # 주점 (principal point)
        h, w = image_size  # 이미지 높이, 너비

        # JSON 데이터 생성
        intrinsics_data = {"px": px, "py": py, "u0": u0, "v0": v0, "h": h, "w": w}

        # JSON 직렬화
        json_str = json.dumps(intrinsics_data)
        json_bytes = self.pack_string(json_str)

        # 메시지 전송
        self.send_message(sock, "INTR", json_bytes)

        # 응답 수신
        code, response_buffer = self.receive_message(sock)
        if code == MegaPoseClient.ServerMessage.OK:
            self._node.get_logger().info("Intrinsics successfully set on the server.")
            return True

        elif code == MegaPoseClient.ServerMessage.ERR:
            self._node.get_logger().warn(
                "Error from server:", self.read_string(response_buffer)
            )
        else:
            self._node.get_logger().warn("Unknown response code:", code)

        return False

    def send_list_objects_request(self, sock: socket.socket):
        """
        서버에 오브젝트 목록을 요청하고 응답을 받는다.

        :param sock: 열린 소켓 객체
        :return: 오브젝트 목록 (list of str) 또는 None
        """
        # 서버에 'GLSO' 요청 전송
        self.send_message(sock, "GLSO", b"")

        # 응답 수신
        code, response_buffer = self.receive_message(sock)

        if code == MegaPoseClient.ServerMessage.RET_LIST_OBJECTS:
            json_str = self.read_string(response_buffer)
            object_list = json.loads(json_str)
            return object_list

        elif code == MegaPoseClient.ServerMessage.ERR:
            self._node.get_logger().warn(
                f"Error from server: {self.read_string(response_buffer)}"
            )
        else:
            self._node.get_logger().warn(f"Unknown response code: {code}")

        return None
