import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup, MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.qos import qos_profile_system_default

import numpy as np
import cv2
from cv_bridge import CvBridge

import matplotlib

matplotlib.use("Agg")  # 백그라운드 렌더링을 위해 GUI 백엔드 비활성화
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg

from sensor_msgs.msg import Image
from std_msgs.msg import *

# 임의의 커스텀 서비스 (환경에 맞게 수정하세요)
from custom_msgs.srv import GetFCNResult

# 작성하신 FCNManager 임포트 (경로는 상황에 맞게 수정)
# from your_custom_module import FCNManager
from fcn_network.fcn_manager import FCNManager
from base_package.image_manager import ImageManager


class FCNServiceNode(Node):
    def __init__(self):
        super().__init__("fcn_service_node")

        self.declare_parameters(
            namespace="",
            parameters=[
                ("fcn_gain", 2.0),
                ("fcn_gamma", 0.7),
                (
                    "model_path",
                    "/home/min/7cmdehdrb/project_sky/src/fcn_network/resource/best_model.pth",
                ),
                ("fcn_image_transform", True),
                ("peak_boundaries", [0, 128, 256, 384, 512, 640]),
            ],
        )

        # 비동기 서비스 처리를 위한 콜백 그룹
        self.srv_cb_group = ReentrantCallbackGroup()
        self.timer_cb_group = MutuallyExclusiveCallbackGroup()

        self.bridge = CvBridge()

        # 메모리에 상주시킬 최신 이미지 및 PDM 데이터
        self._latest_image: np.ndarray = None
        self._latest_1d_pdm: np.ndarray = None
        self._latest_weighted_peak_data: np.ndarray = None
        self._target_map: np.ndarray = None  # 원본 타겟 클래스 맵 (시각화용)

        # --- 1. FCN Manager 초기화 ---
        self.get_logger().info("FCN 모델을 로드합니다...")

        fcn_gain = self.get_parameter("fcn_gain").get_parameter_value().double_value
        fcn_gamma = self.get_parameter("fcn_gamma").get_parameter_value().double_value
        model_path = self.get_parameter("model_path").get_parameter_value().string_value
        fcn_image_transform = (
            self.get_parameter("fcn_image_transform").get_parameter_value().bool_value
        )
        peak_boundaries = list(
            self.get_parameter("peak_boundaries")
            .get_parameter_value()
            .integer_array_value
        )

        if len(peak_boundaries) == 6:
            layer_cnt = 16  # 5개 구간 + 1개 여분
        elif len(peak_boundaries) == 5:
            layer_cnt = 12  # 4개 구간 + 1개 여분
        else:
            self.get_logger().error(
                "Peak boundaries should have 5 or 6 values. Check the parameter configuration."
            )
            raise ValueError("Invalid peak boundaries length")

        self.fcn_manager = FCNManager(
            node=self,
            fcn_gain=fcn_gain,
            fcn_gamma=fcn_gamma,
            model_path=model_path,
            fcn_image_transform=fcn_image_transform,
            layer_cnt=layer_cnt,
        )
        self.fcn_manager.peak_boundaries = peak_boundaries

        # 구역 설정 (필요시 동적 변경 가능)

        # --- 2. Image Manager 초기화 및 상시 구독 ---
        self._image_manager = ImageManager(
            self,
            subscribed_topics=[
                {
                    "topic_name": "/camera/camera1/color/image_raw",
                    "callback": self.image_callback,
                }
            ],
            published_topics=[
                {"topic_name": self.get_name() + "/pdm_visualization"},
                {
                    "topic_name": self.get_name() + "/target_map_visualization",
                },
            ],
        )

        # self._1d_pdm_publisher = self.create_publisher(
        #     Float32MultiArray,
        #     self.get_name() + "/one_d_pdm",
        #     qos_profile=qos_profile_system_default,
        # )
        # self._cnt_publisher = self.create_publisher(
        #     Int32, self.get_name() + "/cnt", qos_profile=qos_profile_system_default
        # )

        # --- 3. 서비스 서버 오픈 (요청 대기) ---
        self.srv = self.create_service(
            GetFCNResult,
            "get_fcn_prediction",
            self.handle_get_fcn_prediction,
            callback_group=self.srv_cb_group,
        )

        # --- 4. 시각화 퍼블리싱 타이머 (1Hz) ---
        HZ = 10.0
        self._one_d_image: Image = None
        self._two_d_image: Image = None
        self.timer = self.create_timer(
            1.0 / HZ, self._publish_image, callback_group=self.timer_cb_group
        )

        # self._cnt = 0
        # self.timer2 = self.create_timer(
        #     1.0 / HZ, self.publish_1d_pdm, callback_group=self.timer_cb_group
        # )
        # self.timer3 = self.create_timer(
        #     1.0 / HZ, self.publish_cnt, callback_group=self.timer_cb_group
        # )

        self.get_logger().info(
            "🟢 노드 B (FCN Service) 준비 완료. 이미지 수신 및 요청 대기 중..."
        )

        self.get_logger().info("🟢 FCN Manager 초기화 완료.")
        self.get_logger().info(
            f"Model Path: {model_path}\n"
            f"Gain: {fcn_gain}\n"
            f"Gamma: {fcn_gamma}\n"
            f"Image Transform: {fcn_image_transform}"
            f"Peak Boundaries: {peak_boundaries}"
        )

    def _publish_image(self):
        """타이머 콜백: 시각화된 이미지가 있으면 주기적으로 발행"""
        if self._latest_1d_pdm is not None:
            if self._one_d_image is not None:
                self._image_manager.get_publisher(
                    self.get_name() + "/pdm_visualization"
                ).publish(self._one_d_image)

        if self._target_map is not None:
            if self._two_d_image is not None:
                self._image_manager.get_publisher(
                    self.get_name() + "/target_map_visualization"
                ).publish(self._two_d_image)

    def image_callback(self, msg: Image):
        """카메라로부터 이미지를 상시 수신하여 최신 상태로 유지합니다."""
        try:
            np_image = self._image_manager.decode_message(
                image_msg=msg, desired_encoding="rgb8"
            )
            # 필요에 따라 crop 로직 적용
            cropped_image = self._image_manager.crop_image(img=np_image)

            # 스레드 간 데이터 덮어쓰기 (Python의 GIL 덕분에 얕은 복사 대입은 thread-safe에 가깝지만,
            # 추론 중 이미지가 바뀌는 것을 방지하려면 lock을 걸어도 좋습니다)
            self._latest_image = cropped_image
        except Exception as e:
            self.get_logger().error(f"이미지 수신 에러: {e}")

    def handle_get_fcn_prediction(
        self, request: GetFCNResult.Request, response: GetFCNResult.Response
    ):
        """A 노드로부터 요청이 들어왔을 때 실행되는 메인 콜백"""
        self.get_logger().info("[B] FCN 추론 요청 수신. 연산 시작...")

        if self._latest_image is None:
            self.get_logger().warn("[B] 아직 수신된 이미지가 없습니다!")
            response.data = []  # 혹은 에러 플래그
            return response

        # 1. 최신 이미지 캡처본으로 추론 진행
        target_image = self._latest_image.copy()

        # 2. 모델 예측 수행 (2D 결과 맵)
        result_2d = self.fcn_manager.predict(target_image)

        # 3. 후처리 및 1D PDM, 가중치 적용 데이터 획득
        weights = request.weight  # 요청에서 가중치 배열과 타겟 클래스 인덱스 받기
        target_class_idx = request.target_class_idx

        # TODO: target_class_idx 및 가중치 등, request에서 필요한 정보를 받아서 처리하도록 개선 필요
        one_d_pdm, _, weighted_peak_data, _, target_map = (
            self.fcn_manager.post_process_results(
                result_2d, weights, target_class_idx=target_class_idx
            )
        )

        # 시각화 타이머가 사용할 수 있도록 1D 데이터 저장
        self._latest_1d_pdm = one_d_pdm
        self._target_map = target_map  # 시각화용 원본 타겟 클래스 맵 저장
        self._latest_weighted_peak_data = (
            weighted_peak_data  # 시각화용 가중치 적용 데이터 저장
        )

        # 4. 길이 N(4)의 float 배열 응답 생성
        # NumPy 배열을 Python 리스트(float)로 변환하여 할당

        self._one_d_image: Image = (
            self._get_pdm_visualization()
        )  # 시각화 퍼블리싱 (옵션)
        self._two_d_image: Image = (
            self._get_target_map_visualization()
        )  # 시각화 퍼블리싱 (옵션)

        response.data = weighted_peak_data.tolist()

        response.one_d_image = self._one_d_image
        response.two_d_image = self._two_d_image

        self.get_logger().info(f"[B] 추론 완료. 결과: {response.data}")
        return response

    def _get_pdm_visualization(self):
        """1초 주기로 1D PDM 그래프를 렌더링하여 ROS Image로 발행"""
        if self._latest_1d_pdm is None:
            return

        """
        1. 1D PDM 그래프 렌더링
        """
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(self._latest_1d_pdm, color="blue")
        ax.grid(True)
        ax.set_title("1D PDM Profile")
        ax.set_xlim(0, len(self._latest_1d_pdm))

        # 설정된 경계선(Boundaries) 긋기
        for b in self.fcn_manager.peak_boundaries:
            ax.axvline(x=b, color="red", linestyle="--", alpha=0.5)

        # 렌더링 후 넘파이 배열로 변환
        canvas = FigureCanvasAgg(fig)
        canvas.draw()
        buf = canvas.buffer_rgba()
        vis_image = np.asarray(buf)

        # RGBA를 BGR8 (OpenCV 표준)로 변환
        vis_image_bgr = cv2.cvtColor(vis_image, cv2.COLOR_RGBA2BGR)
        plt.close(fig)  # 메모리 누수 방지

        # ROS 메세지 변환 및 발행
        img_msg = self.bridge.cv2_to_imgmsg(vis_image_bgr, encoding="bgr8")
        img_msg.header.stamp = self.get_clock().now().to_msg()
        img_msg.header.frame_id = "camera1_color_optical_frame"

        return img_msg

    def _get_target_map_visualization(self):

        ######################################################

        """
        2. 원본 타겟 클래스 맵 시각화
        """
        # 메모리에 target_map이 아직 없으면 무시
        if self._target_map is None:
            return

        # 1. Float 배열(보통 0.0~1.0)을 시각화를 위해 0~255 범위의 8비트 이미지(uint8)로 정규화
        target_vis = cv2.normalize(
            self._target_map, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U
        )

        # 2. NumPy 배열을 ROS Image 메시지로 변환 (Grayscale 이므로 'mono8' 사용)
        img_msg = self.bridge.cv2_to_imgmsg(target_vis, encoding="mono8")

        # (선택) 헤더에 타임스탬프 추가가 필요하다면 아래 주석 해제
        img_msg.header.stamp = self.get_clock().now().to_msg()
        img_msg.header.frame_id = "camera1_color_optical_frame"

        return img_msg

    # def publish_1d_pdm(self):
    #     """1초 주기로 1D PDM 데이터를 Float32MultiArray로 발행"""
    #     if self._latest_1d_pdm is None:
    #         return

    #     # NumPy 배열을 Float32MultiArray 메시지로 변환
    #     pdm_msg = Float32MultiArray()
    #     pdm_msg.data = self._latest_weighted_peak_data.astype(np.float32).tolist()

    #     # 토픽 발행
    #     self._1d_pdm_publisher.publish(pdm_msg)

    # def publish_cnt(self):
    #     """1초 주기로 카운트 발행 (디버깅용)"""
    #     cnt_msg = Int32()
    #     cnt_msg.data = self._cnt
    #     self._cnt_publisher.publish(cnt_msg)


def main(args=None):
    rclpy.init(args=args)
    node = FCNServiceNode()

    # 비동기 서비스 처리(추론)와 타이머(시각화)가 동시에 돌아가기 위해 멀티스레드 사용
    executor = MultiThreadedExecutor()
    executor.add_node(node)

    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
