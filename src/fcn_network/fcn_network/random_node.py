import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, qos_profile_system_default
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor

import numpy as np
import onnxruntime as ort
from dataclasses import dataclass
from typing import List, Union

from std_msgs.msg import *
from sensor_msgs.msg import Image
from custom_msgs.srv import GetPolicyAction, GetFCNResult
from fcn_network.drl_manager import RLPolicyManager, PolicyAction


class PolicyServiceNode(Node):
    def __init__(self):
        super().__init__("policy_service_node")

        self.declare_parameters(
            namespace="",
            parameters=[
                (
                    "model_path",
                    "/home/min/7cmdehdrb/project_sky/src/fcn_network/resource/exported_45/policy.onnx",
                ),
            ],
        )

        self.cb_group = ReentrantCallbackGroup()

        self._closet_object_list: List[int] = None
        self.obj_sub = self.create_subscription(
            Int32MultiArray,
            "/closest_object_classifier/closest_object_ids",
            self.object_id_callback,
            qos_profile=qos_profile_system_default,
            callback_group=self.cb_group,
        )

        # self._cnt_pub = self.create_publisher(
        #     Int32,
        #     "/fcn_service_node/cnt",
        #     qos_profile=qos_profile_system_default,
        # )
        # self._cnt = 0

        # 3. Main 노드를 위한n 서비스 서버 오픈
        self.srv = self.create_service(
            GetPolicyAction,
            "get_policy_action",
            self.handle_get_policy_action,
            callback_group=self.cb_group,
        )

        self.get_logger().info("🟢 Node A (Policy Server) 준비 완료. 요청 대기 중...")
        self.get_logger().info("RL Policy Manager 초기화 완료.")

    #     self._timer = self.create_timer(
    #         0.1, self._publish_cnt, callback_group=self.cb_group
    #     )

    # def _publish_cnt(self):
    #     """현재 카운트 값을 주기적으로 퍼블리시하는 헬퍼 함수 (디버깅용)"""
    #     cnt_msg = Int32()
    #     cnt_msg.data = self._cnt
    #     self._cnt_pub.publish(cnt_msg)

    def object_id_callback(self, msg: Int32MultiArray):
        self._closet_object_list = [int(x) for x in msg.data]

    async def handle_get_policy_action(
        self, request: GetPolicyAction.Request, response: GetPolicyAction.Response
    ):

        # self._cnt += 1

        target_id = request.target_id
        self.get_logger().info(
            f"[A] Main으로부터 추론 요청 수신 (Target ID: {target_id})"
        )

        # 3. 모델 1회 추론 및 내부 상태(t-1) 자동 갱신
        self.get_logger().info(f"   -> Random 추론 진행...")

        print(f"현재 가장 가까운 객체 리스트: {self._closet_object_list}")

        if target_id in list(self._closet_object_list):
            response.action_type = 0  # Grasp
            response.target_column = int(
                np.where(np.array(self._closet_object_list) == target_id)[0][0]
            )

            self.get_logger().info(
                f"✅ [A] 추론 완료! 반환 값: Action={response.action_type}, Column={response.target_column}"
            )

        else:
            # 4. 결과 반환 (Main으로)
            response.action_type = 0  # 0 is Grasp (Fixed)
            response.target_column = int(
                np.random.choice(
                    [i for i, v in enumerate(self._closet_object_list) if v != -1]
                )
            )

            self.get_logger().info(
                f"✅ [A] 랜덤 탐색!: Action={response.action_type}, Column={response.target_column}"
            )

        response.one_d_pdm = []
        response.one_d_image = Image()
        response.two_d_image = Image()

        return response


def main(args=None):
    rclpy.init(args=args)
    node = PolicyServiceNode()

    # 여러 콜백(서버, 클라이언트, 서브스크립션)이 엉키지 않도록 멀티스레드 사용
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
