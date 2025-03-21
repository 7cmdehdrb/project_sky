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
from custom_msgs.msg import BoundingBox, BoundingBoxMultiArray

# TF
from tf2_ros import *

# Python
import numpy as np
from PIL import ImageEnhance
from PIL import Image as PILImage

# OpenCV
import cv2
from cv_bridge import CvBridge


# Megapose Server
import socket
import struct
import json
import io
import time

from base_package.header import QuaternionAngle


class MegaPoseClient:
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

    def __init__(self, node: Node):
        self.SERVER_HOST = "127.0.0.1"
        self.SERVER_PORT = 5555
        self.SERVER_OPERATION_CODE_LENGTH = 4

        self.node = node
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.connect((self.SERVER_HOST, self.SERVER_PORT))

        self.segmentation_subscriber = self.node.create_subscription(
            BoundingBoxMultiArray,
            "/real_time_segmentation_node/segmented_bbox",
            self.segmentation_callback,
            qos_profile=qos_profile_system_default,
        )

        self.result_pose = PoseStamped()

        # Parameters
        self.is_configured = False
        self.is_segmentation_valid = False
        self.score_threshold = 0.5

        temp = [
            221.11175537109375,
            142.1138916015625,
            282.87127685546875,
            354.2355651855469,
        ]
        int_temp = [int(i) for i in temp]

        self.initial_data = {
            "detections": [int_temp],
            "labels": ["smoothie"],
            "use_depth": False,
        }
        self.loop_data = {
            "initial_cTos": None,
            "labels": ["smoothie"],
            "refiner_iterations": 1,
            "use_depth": False,
        }

        self.last_time = self.node.get_clock().now()

        # Run Once
        while True:
            response = self.send_list_objects_request(self.socket)
            if response is not None:
                print("Object list:", response)
                break

    def set_intrinsics(self, K: np.ndarray, image_size: tuple):
        if not self.is_configured:
            response = self.send_intrinsics_request(
                self.socket,
                K=K,
                image_size=image_size,
            )
            self.is_configured = response

    def segmentation_callback(self, msg: BoundingBoxMultiArray):
        for bbox in msg.data:
            bbox: BoundingBox

            if bbox.cls == self.initial_data["labels"][0]:
                self.initial_data["detections"] = [bbox.bbox]

    def run(self, frame: np.array):
        """Send a frame and json data to the server and receive the result."""
        time = self.node.get_clock().now()

        dt = (time - self.last_time).nanoseconds / 1e9
        self.last_time = time

        self.node.get_logger().info(f"dt: {round(dt, 3)}, FPS: {round(1 / dt, 1)}")

        cto, score, bbox = self.send_pose_request(
            frame,
            (self.initial_data if not self.is_segmentation_valid else self.loop_data),
        )

        if cto is None or score is None or bbox is None:
            self.node.get_logger().warn("No response from server.")
            return None, self.initial_data["detections"][0]

        self.is_segmentation_valid = score > self.score_threshold

        if not self.is_segmentation_valid:
            self.node.get_logger().warn("Score too low. Requesting new bounding box.")
            return None, self.initial_data["detections"][0]

        # Update cTo
        self.loop_data["initial_cTos"] = [cto.tolist()]

        # Get PoseStamped message
        cto_matrix = np.array(cto).reshape(4, 4)
        cto_matrix_ros_frame = QuaternionAngle.transform_realsense_to_ros(cto_matrix)
        print(f"BBox: {bbox}")
        print(f"Score: {score}")

        translation_matrix = cto_matrix_ros_frame[:3, 3]
        rotation_matrix = cto_matrix_ros_frame[:3, :3]

        r, p, y = QuaternionAngle.euler_from_rotation_matrix(rotation_matrix)
        quat = QuaternionAngle.quaternion_from_euler(r, p, y)

        pose_msg = PoseStamped(
            header=Header(
                stamp=self.node.get_clock().now().to_msg(),
                frame_id="camera1_link",
            ),
            pose=Pose(
                position=Point(
                    x=translation_matrix[0],
                    y=translation_matrix[1] - 0.06,
                    z=translation_matrix[2],
                ),
                orientation=Quaternion(x=quat[0], y=quat[1], z=quat[2], w=quat[3]),
            ),
        )

        self.result_pose = pose_msg
        return pose_msg, bbox

    def send_message(self, sock: socket.socket, code: str, data: bytes):
        msg_length = struct.pack(">I", len(data))
        sock.sendall(msg_length + code.encode("UTF-8") + data)

    def receive_message(self, sock: socket.socket):
        msg_length = sock.recv(4)
        length = struct.unpack(">I", msg_length)[0]
        code = sock.recv(self.SERVER_OPERATION_CODE_LENGTH).decode("UTF-8")
        data = sock.recv(length)
        return code, io.BytesIO(data)

    def pack_string(self, data: str) -> bytes:
        encoded = data.encode("utf-8")
        length = struct.pack(">I", len(encoded))
        return length + encoded

    def read_string(self, buffer: io.BytesIO) -> str:
        length = struct.unpack(">I", buffer.read(4))[0]
        return buffer.read(length).decode("utf-8")

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
        self.send_message(self.socket, MegaPoseClient.ServerMessage.GET_POSE, data)

        # **(6) 서버 응답 수신**
        code, response_buffer = self.receive_message(self.socket)
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
            self.node.get_logger().info("Intrinsics successfully set on the server.")
            return True

        elif code == MegaPoseClient.ServerMessage.ERR:
            self.node.get_logger().warn(
                "Error from server:", self.read_string(response_buffer)
            )
        else:
            self.node.get_logger().warn("Unknown response code:", code)

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
            self.node.get_logger().warn(
                f"Error from server: {self.read_string(response_buffer)}"
            )
        else:
            self.node.get_logger().warn(f"Unknown response code: {code}")

        return None


class RealTimeTrackingClientNode(Node):
    def __init__(self):
        super().__init__("real_time_tracking_client_node")

        self.megapose_client = MegaPoseClient(node=self)
        self.bridge = CvBridge()

        self.do_publish_image = True
        self.width, self.height = 640, 480

        # ROS
        self.image_subscriber = self.create_subscription(
            Image,
            "/camera/camera1/color/image_raw",
            self.image_callback,
            qos_profile=qos_profile_system_default,
        )
        self.camera_info_subscriber = self.create_subscription(
            CameraInfo,
            "/camera/camera1/color/camera_info",
            self.camera_info_callback,
            qos_profile=qos_profile_system_default,
        )
        self.pose_publisher = self.create_publisher(
            PoseStamped,
            self.get_name() + "/megapose",
            qos_profile=qos_profile_system_default,
        )
        self.image_publisher = self.create_publisher(
            Image,
            self.get_name() + "/megapose_image",
            qos_profile=qos_profile_system_default,
        )

        self.frame = None

        self.timer = self.create_timer(0.05, self.run)

    def image_callback(self, msg: Image):
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")

        width = frame.shape[1]
        height = frame.shape[0]
        print(f"Image shape: {frame.shape[:2]}")

        if width == 1280 and height == 720:
            self.frame = self.crop_and_resize_image(frame)

        elif width == self.width and height == self.height:
            self.frame = frame

        else:
            self.get_logger().warn("Invalid image size. Cannot set frame.")

    def camera_info_callback(self, msg: CameraInfo):
        K = np.array(msg.k).reshape(3, 3)
        image_size = (msg.height, msg.width)

        if msg.height == 720 and msg.width == 1280:
            image_size = (self.height, self.width)

            offset = int((msg.width - self.width) // 2)

            K[0, 0] = K[0, 0] * (self.width / msg.width)
            K[1, 1] = K[1, 1] * (self.height / msg.height)
            K[0, 2] = (K[0, 2] - offset) * (self.width / msg.width)
            K[1, 2] = K[1, 2] * (self.height / msg.height)

        elif msg.height == self.height and msg.width == self.width:
            # Do nothing
            pass

        else:
            # Prevent setting intrinsics
            self.get_logger().warn("Invalid image size. Cannot set intrinsics.")
            return None

        self.megapose_client.set_intrinsics(
            K=K,
            image_size=image_size,
        )

    def run(self):
        # while True:
        if self.frame is None:
            self.get_logger().warn("Frame does not exist.")
            return None

        pose_msg, bbox = self.megapose_client.run(frame=self.frame)

        if pose_msg is not None:
            self.pose_publisher.publish(pose_msg)

        if bbox is not None and self.do_publish_image:
            cv2.rectangle(
                self.frame,
                (int(bbox[0]), int(bbox[1])),
                (int(bbox[2]), int(bbox[3])),
                (0, 255, 0),
                2,
            )

            image_msg = self.bridge.cv2_to_imgmsg(self.frame, encoding="bgr8")
            self.image_publisher.publish(image_msg)

    def crop_and_resize_image(self, image: np.array):
        """Crop 1280x720 image to 640x480."""
        image_height, image_width = image.shape[:2]

        assert image_height == 720 and image_width == 1280

        crop_ratio = self.width / self.height

        new_width = int(image_height * crop_ratio)
        offset = (image_width - new_width) // 2
        cropped_image = image[:, offset : offset + new_width]

        return cv2.resize(cropped_image, (self.width, self.height))


def main(args=None):
    rclpy.init(args=args)

    node = RealTimeTrackingClientNode()

    rclpy.spin(node=node)

    cv2.destroyAllWindows()

    node.destroy_node()


if __name__ == "__main__":
    main()
