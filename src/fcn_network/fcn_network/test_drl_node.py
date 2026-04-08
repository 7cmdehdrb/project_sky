import sys
import os
import time
import threading

import numpy as np
import cv2
import rclpy
from enum import Enum
import copy
from rclpy.node import Node
from rclpy.time import Time
from rclpy.duration import Duration
from rclpy.qos import QoSProfile, qos_profile_system_default

from std_msgs.msg import *
from geometry_msgs.msg import *
from sensor_msgs.msg import *
from nav_msgs.msg import *
from visualization_msgs.msg import *
from builtin_interfaces.msg import Duration as BuiltinDuration

from tf2_ros import *

from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup

import datetime
from loguru import logger
from custom_msgs.srv import GetPolicyAction
from base_package.image_manager import ImageManager
from enum import Enum

class Mode(Enum):
    Occusion = 0
    Simmilarity = 1
    Nofcn = 2
    MCTS = 3
    CHAOS = 4
    


class ImageLogger:
    def __init__(self, node: Node, col_num: int = 4, exp_num: int = -1, mode: Mode = Mode.Occusion):
        # 기본 인자
        self._node = node
        self._col_num = col_num
        self._mode = mode
        self._exp_num = exp_num

        self._file_name_dict = {
            Mode.Occusion: "DRL_10,0",
            Mode.Simmilarity: "DRL_0,10",
            Mode.Nofcn: "NF",
            Mode.MCTS: "MCTS",
            Mode.CHAOS: "CHAOS",
        }


        # >>> 로그용 인자 >>>
        ROOT_DIR = "/home/irol/DRL-Occluded-Object-Search/src/fcn_network/log"
        # existing_dirs = [
        #     d
        #     for d in os.listdir(ROOT_DIR)
        #     if d.startswith("exp_") and os.path.isdir(os.path.join(ROOT_DIR, d))
        # ]
        # existing_nums = sorted(
        #     [int(d.split("_")[1]) for d in existing_dirs if d.split("_")[1].isdigit()]
        # )
        # next_num = (existing_nums[-1] if existing_nums else -1) + 1
        self._log_dir = os.path.join(ROOT_DIR, f"{self._file_name_dict[self._mode]}_{self._exp_num}")
        # <<< 로그용 인자 <<<

        # >>> 로깅 시작 >>>
        os.makedirs(self._log_dir, exist_ok=True)
        logger.add(
            os.path.join(self._log_dir, "image_log_{time}.log"),
            format="{message}",
            level="INFO",
        )
        logger.info(f"step,target_id,action,target_column,1d_pdm")
        # <<< 로깅 시작 <<<

        # >>> ROS Subscriber & Publisher 초기화 >>>

        self._raw_image: Image = None
        self._closest_image: Image = None
        self._segmentation_image: Image = None
        self._top_view_image: Image = None

        self._image_manager = ImageManager(
            node=self._node,
            subscribed_topics=[
                {
                    "topic_name": "/camera/camera1/color/image_raw",
                    "callback": self._callback_raw_image,
                },
                {
                    "topic_name": "/closest_object_classifier/closest_object_overlay",
                    "callback": self._callback_closest_image,
                },
                {
                    "topic_name": "/real_time_segmentation_node/segmented_image",
                    "callback": self._callback_segmentation_image,
                },
                {
                    "topic_name": "/action_cam_node/top_view_image",
                    "callback": self._callback_top_view_image,
                },
                # {
                #     "topic_name": "/fcn_service_node/pdm_visualization",
                #     "callback": self._callback_1d_fcn_processed_image,
                # },
                # {
                #     "topic_name": "/fcn_service_node/target_map_visualization",
                #     "callback": self._callback_2d_fcn_processed_image,
                # },
            ],
            published_topics=[],
        )

        # self._1d_pdm_sub = self._node.create_subscription(
        #     Float32MultiArray,
        #     "/fcn_service_node/one_d_pdm",
        #     qos_profile=qos_profile_system_default,
        #     callback=self._callback_1d_pdm,
        # )
        # <<< ROS Subscriber & Publisher 초기화 <<<

        # >>> Tlqkf >>>
        # self._cnt_sub = self._node.create_subscription(
        #     Int32,
        #     "/fcn_service_node/cnt",
        #     qos_profile=qos_profile_system_default,
        #     callback=self._callback_cnt,
        # )

        # 외부에서 값을 부여할 것
        self.step = 0
        self.target_id = 0
        self.action = 0
        self.target_column = 0
        self.one_d_fcn_processed_image: Image = None
        self.two_d_fcn_processed_image: Image = None
        self.one_d_pdm_value: Float32MultiArray = None

        # 디버깅용 카운트 및 트리거 플래그
        self._cnt = 0
        self._trigger = False
        self._trigger_time = None
        # <<< Tlqkf <<<

        # HZ: 2

    # def _reset(self):
    #     self._raw_image = None
    #     self._closest_image = None
    #     self._segmentation_image = None
    #     self._1d_fcn_processed_image = None
    #     self._2d_fcn_processed_image = None
    #     self._top_view_image = None
    #     self._1d_pdm_value = None

    # def run(self):
    #     # timer 를 써서 주기적으로 회전 시킬 함수
    #     # self._trigger가 True + trigger time 과 3초 이상 차이날 때 로그를 기록하고 _trigger는 False로 바꿔주는 함수
    #     # print(self._trigger)
    #     # print(self._trigger_time, (time.time() - self._trigger_time) if self._trigger_time else None)

    #     if self._trigger_time is None:
    #         return

    #     if self._trigger and (time.time() - self._trigger_time) > 2.0:
    #         self.log()
    #         self._trigger = False
    #         self._trigger_time = None

    # def _callback_cnt(self, msg: Int32):
    #     data = msg.data
    #     if data != self._cnt:
    #         # self._node.get_logger().info(f"카운트 변경 감지: {self._cnt} -> {data}")
    #         # 카운트가 변경될 때마다 로그에 기록
    #         self._cnt = data
    #         self._trigger = True
    #         self._trigger_time = time.time()

    def _callback_raw_image(self, msg: Image):
        self._raw_image = msg

    def _callback_closest_image(self, msg: Image):
        self._closest_image = msg

    def _callback_segmentation_image(self, msg: Image):
        self._segmentation_image = msg

    def _callback_top_view_image(self, msg: Image):
        self._top_view_image = msg

    # def _callback_1d_fcn_processed_image(self, msg: Image):
    #     self._1d_fcn_processed_image = msg

    # def _callback_2d_fcn_processed_image(self, msg: Image):
    #     self._2d_fcn_processed_image = msg

    # def _callback_1d_pdm(self, msg: Float32MultiArray):
    #     self._1d_pdm_value = msg

    def _post_process_images(self, msg: Image, ignore_none: bool = False) -> np.ndarray:

        if msg is None:
            if ignore_none is True:
                self._node.get_logger().warn("Received None image, but ignore_none=True, so returning blank image.")
                return np.zeros((480, 640, 3), dtype=np.uint8)
            else:
                self._node.get_logger().warn("Received None image, returning None.")
                return None
            
        if msg.width == 0 or msg.height == 0:
            self._node.get_logger().warn("Received image with zero width or height, returning blank image.")
            return np.zeros((480, 640, 3), dtype=np.uint8)

        np_image = self._image_manager.decode_message(
            image_msg=msg, desired_encoding="bgr8"
        )
        if np_image.shape[0] != 480 or np_image.shape[1] != 640:
            np_image = self._image_manager.crop_image(img=np_image)

        return np_image

    def log(self):
        # Images from Subscribers
        raw_image = self._post_process_images(self._raw_image)
        closest_image = self._post_process_images(self._closest_image)
        segmentation_image = self._post_process_images(self._segmentation_image)
        top_view_image = self._post_process_images(
            self._top_view_image, ignore_none=True
        )

        # Images from FCN Service (복사본 생성)
        # print(type(self.one_d_fcn_processed_image))
        # print(type(self.two_d_fcn_processed_image))
        fcn_1d_image = self._post_process_images(self.one_d_fcn_processed_image, ignore_none=True)
        fcn_2d_image = self._post_process_images(self.two_d_fcn_processed_image, ignore_none=True)
        
        processed_1d_pdm = (
            "None"
            if self.one_d_pdm_value is None
            else f"{'; '.join(f'{v:.2f}' for v in self.one_d_pdm_value.data)}"
        )

        images_to_save = [
            (raw_image, "raw"),
            (closest_image, "closest"),
            (segmentation_image, "segmentation"),
            (fcn_1d_image, "fcn_1d"),
            (fcn_2d_image, "fcn_2d"),
            (top_view_image, "top_view"),
        ]

        for image, name in images_to_save:
            if image is None:
                self._node.get_logger().warn(f"Invalid {name} image for step {self.step}...")
                continue

            cv2.imwrite(os.path.join(self._log_dir, f"{self.step}_{name}.png"), image)

        logger.info(
            f"{self.step},{self.target_id},{self.action},{self.target_column},{processed_1d_pdm}"
        )


class MockMainNode(Node):
    
    def __init__(self, target_id: int, num_columns: int = 4, mode: Mode = Mode.Occusion, exp_num: int = -1):
        super().__init__("mock_main_node")
        self._mode = mode
        self._exp_num = exp_num

        self._image_logger = ImageLogger(node=self, mode=self._mode, exp_num=self._exp_num)
        self._target_id = target_id

        if num_columns not in (4, 5):
            raise ValueError("num_columns must be 4 or 5")
        self.num_columns = num_columns

        self._action_descriptions = {
            0: "잡기",
            1: "오른쪽 밀기",
            2: "왼쪽 밀기",
        }

        self.service_cb_group = MutuallyExclusiveCallbackGroup()
        self.client_a = self.create_client(
            GetPolicyAction, "get_policy_action", callback_group=self.service_cb_group
        )

        self.get_logger().info(
            f"Node A (Policy Server) 대기 중... (열 개수: {num_columns})"
        )

        while not self.client_a.wait_for_service(timeout_sec=1.0):
            self.get_logger().info("Node A가 켜질 때까지 기다리는 중...")

        self.get_logger().info("🟢 Node A 확인 완료!")
        self.get_logger().info(
            "✨ 엔터를 눌러 제어 요청을 보내세요. 종료하려면 'q' + 엔터를 입력하세요."
        )

        self.request_count = 0

        # self._timer = self.create_timer(0.5, self._image_logger.run)

    def send_request(self):
        self.request_count += 1
        # self._image_logger._reset()

        req = GetPolicyAction.Request()
        req.target_id = self._target_id

        self.get_logger().info(
            f"▶️ [Main] {self.request_count}번째 요청 발송 (Target ID: {self._target_id}).."
        )

        future = self.client_a.call_async(req)
        future.add_done_callback(
            lambda fut, req_num=self.request_count: self.response_callback(fut, req_num)
        )

    def response_callback(self, future: rclpy.Future, req_num: int):
        try:
            result: GetPolicyAction.Response = future.result()

            action = result.action_type
            col = result.target_column
            one_d_pdm_msg = Float32MultiArray(data=result.one_d_pdm)
            one_d_image: Image = result.one_d_image
            two_d_image: Image = result.two_d_image

            action_str = self._action_descriptions.get(action, f"알 수 없음 ({action})")

            target_visual = ["□"] * self.num_columns
            if action == 0:
                target_visual[col] = "■"
            elif action == 1:
                target_visual[col] = "▶"
            elif action == 2:
                target_visual[col] = "◀"

            visual_str = "".join(target_visual)

            self.get_logger().info(
                f"✅ [Main] {req_num}번째 응답 수신 성공! Action: {action} | Target Column: {col}\n"
                f"{action_str} -> {visual_str}"
            )

            self._image_logger.step = req_num
            self._image_logger.target_id = self._target_id
            self._image_logger.action = action
            self._image_logger.target_column = col
            self._image_logger.one_d_pdm_value = one_d_pdm_msg
            self._image_logger.one_d_fcn_processed_image = one_d_image
            self._image_logger.two_d_fcn_processed_image = two_d_image


            self._image_logger.log()

        except Exception as e:
            self.get_logger().error(f"❌ [Main] {req_num}번째 호출 실패: {e}")


def main(args=None):

    TARGET_ID = 4
    NUM_COLUMNS = 5

    EXP_NUM = 107
    MODE = Mode.CHAOS

    rclpy.init(args=args)
    try:
        node = MockMainNode(target_id=TARGET_ID, num_columns=NUM_COLUMNS, mode=MODE, exp_num=EXP_NUM)
    except ValueError as e:
        print(f"[ERROR] Failed to initialize node: {e}")
        return

    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)

    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()

    try:
        while rclpy.ok():
            user_input = (
                input(" Press Enter to send request, 'q' to quit: ").strip().lower()
            )
            if user_input == "q":
                break
            elif user_input == "":
                node.send_request()
            else:
                node.get_logger().info("Press Enter or 'q' only.")

    except KeyboardInterrupt:
        node.get_logger().info("Interrupted by user (Ctrl+C)")
    finally:
        node.get_logger().info("Shutting down...")
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
