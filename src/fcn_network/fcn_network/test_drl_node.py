import sys
import os
import time
import threading

import numpy as np
import cv2
import rclpy
from enum import Enum
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


class ImageLogger:
    def __init__(self, node: Node, col_num: int = 4):
        self._node = node
        self._col_num = col_num

        ROOT_DIR = "/home/min/7cmdehdrb/project_sky/src/fcn_network/log"

        existing_dirs = [
            d
            for d in os.listdir(ROOT_DIR)
            if d.startswith("exp_") and os.path.isdir(os.path.join(ROOT_DIR, d))
        ]
        existing_nums = sorted(
            [int(d.split("_")[1]) for d in existing_dirs if d.split("_")[1].isdigit()]
        )
        next_num = (existing_nums[-1] if existing_nums else -1) + 1

        self._log_dir = os.path.join(ROOT_DIR, f"exp_{next_num:03d}")
        os.makedirs(self._log_dir, exist_ok=True)

        logger.add(
            os.path.join(self._log_dir, "image_log_{time}.log"),
            format="{message}",
            level="INFO",
        )
        logger.info(f"step,target_id,action,target_column,1d_pdm")

        self._raw_image: Image = None
        self._closest_image: Image = None
        self._segmentation_image: Image = None
        self._1d_fcn_processed_image: Image = None
        self._2d_fcn_processed_image: Image = None
        self._top_view_image: Image = None
        self._1d_pdm_value: Float32MultiArray = None

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
                    "topic_name": "/fcn_service_node/pdm_visualization",
                    "callback": self._callback_1d_fcn_processed_image,
                },
                {
                    "topic_name": "/fcn_service_node/target_map_visualization",
                    "callback": self._callback_2d_fcn_processed_image,
                },
                {
                    "topic_name": "/action_cam_node/top_view_image",
                    "callback": self._callback_top_view_image,
                },
            ],
            published_topics=[],
        )
        self._1d_pdm_sub = self._node.create_subscription(
            Float32MultiArray,
            "/fcn_service_node/one_d_pdm",
            qos_profile=qos_profile_system_default,
            callback=self._callback_1d_pdm,
        )

    def _reset(self):
        self._raw_image = None
        self._closest_image = None
        self._segmentation_image = None
        self._1d_fcn_processed_image = None
        self._2d_fcn_processed_image = None
        self._top_view_image = None
        self._1d_pdm_value = None

    def _callback_raw_image(self, msg: Image):
        self._raw_image = msg

    def _callback_closest_image(self, msg: Image):
        self._closest_image = msg

    def _callback_segmentation_image(self, msg: Image):
        self._segmentation_image = msg

    def _callback_1d_fcn_processed_image(self, msg: Image):
        self._1d_fcn_processed_image = msg

    def _callback_2d_fcn_processed_image(self, msg: Image):
        self._2d_fcn_processed_image = msg

    def _callback_top_view_image(self, msg: Image):
        self._top_view_image = msg

    def _callback_1d_pdm(self, msg: Float32MultiArray):
        self._1d_pdm_value = msg

    def _post_process_images(self, msg: Image, ignore_none: bool = False) -> np.ndarray:

        if msg is None:
            if ignore_none is True:
                return np.zeros((480, 640, 3), dtype=np.uint8)
            else:
                return None

        np_image = self._image_manager.decode_message(
            image_msg=msg, desired_encoding="bgr8"
        )
        if np_image.shape[0] != 480 or np_image.shape[1] != 640:
            np_image = self._image_manager.crop_image(img=np_image)

        return np_image

    def log(self, step: int, target_id: int, action: int, target_column: int):
        start_time = time.time()
        timeout_sec = 3.0  # 최대 5초까지만 대기

        # ✅ 타임아웃 및 누락 토픽 체크 로직 추가
        while True:
            raw_image = self._post_process_images(self._raw_image)
            closest_image = self._post_process_images(self._closest_image)
            segmentation_image = self._post_process_images(self._segmentation_image)
            fcn_1d_image = self._post_process_images(self._1d_fcn_processed_image)
            fcn_2d_image = self._post_process_images(self._2d_fcn_processed_image)
            top_view_image = self._post_process_images(
                self._top_view_image, ignore_none=True
            )

            missing_topics = []
            if raw_image is None:
                missing_topics.append("raw")
            if closest_image is None:
                missing_topics.append("closest")
            if segmentation_image is None:
                missing_topics.append("segmentation")
            if fcn_1d_image is None:
                missing_topics.append("fcn_1d")
            if fcn_2d_image is None:
                missing_topics.append("fcn_2d")
            if top_view_image is None:
                missing_topics.append("top_view")
            if self._1d_pdm_value is None:
                missing_topics.append("1d_pdm")

            # 모든 이미지가 다 들어왔으면 탈출
            if not missing_topics:
                self._node.get_logger().info(
                    f"📸 [{step}번째] 모든 이미지 수신 완료! 저장 진행."
                )
                break

            # 지정된 시간을 초과하면 경고를 띄우고 탈출 (들어온 것만이라도 저장)
            if time.time() - start_time > timeout_sec:
                self._node.get_logger().error(
                    f"⚠️ [{step}번째] 이미지 수신 타임아웃! 누락된 토픽: {missing_topics}. 수신된 이미지만 저장합니다."
                )
                break

            time.sleep(0.1)

        if raw_image is not None:
            cv2.imwrite(os.path.join(self._log_dir, f"{step}_raw.png"), raw_image)

        if closest_image is not None:
            cv2.imwrite(
                os.path.join(self._log_dir, f"{step}_closest.png"), closest_image
            )

        if segmentation_image is not None:
            cv2.imwrite(
                os.path.join(self._log_dir, f"{step}_segmentation.png"),
                segmentation_image,
            )

        if fcn_1d_image is not None:
            cv2.imwrite(os.path.join(self._log_dir, f"{step}_fcn_1d.png"), fcn_1d_image)

        if fcn_2d_image is not None:
            cv2.imwrite(os.path.join(self._log_dir, f"{step}_fcn_2d.png"), fcn_2d_image)

        if top_view_image is not None:
            cv2.imwrite(
                os.path.join(self._log_dir, f"{step}_top_view.png"), top_view_image
            )

        if self._1d_pdm_value is not None:
            processed_1d_pdm = (
                f"{'; '.join(f'{v:.2f}' for v in self._1d_pdm_value.data)}"
            )
        else:
            processed_1d_pdm = f"{'; '.join(f'{v:.2f}' for v in [0.0] * self._col_num)}"

        logger.info(f"{step},{target_id},{action},{target_column},{processed_1d_pdm}")

        self._reset()


class MockMainNode(Node):
    def __init__(self, target_id: int, num_columns: int = 4):
        super().__init__("mock_main_node")

        self._image_logger = ImageLogger(node=self)
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

    def send_request(self):
        self.request_count += 1
        self._image_logger._reset()

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
            result = future.result()

            action = result.action_type
            col = result.target_column

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

            # ✅ ROS Executor의 스레드 점유를 완전히 피하기 위해 Python 스레드로 분리합니다.
            log_thread = threading.Thread(
                target=self._image_logger.log,
                args=(req_num, self._target_id, action, col),
                daemon=True,
            )
            log_thread.start()

        except Exception as e:
            self.get_logger().error(f"❌ [Main] {req_num}번째 호출 실패: {e}")


def main(args=None):
    NUM_COLUMNS = 4

    rclpy.init(args=args)
    try:
        node = MockMainNode(target_id=0, num_columns=NUM_COLUMNS)
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
