import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, qos_profile_system_default
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor

import numpy as np
import onnxruntime as ort
from dataclasses import dataclass
from typing import List, Union

from std_msgs.msg import Float32MultiArray, Int32MultiArray
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
                    "/home/irol/DRL-Occluded-Object-Search/src/fcn_network/resource/best_model_45_og.pth",
                ),
            ],
        )

        self.cb_group = ReentrantCallbackGroup()

        # 2. Node B 클라이언트 설정 및 확인
        self.fcn_client = self.create_client(
            GetFCNResult, "get_fcn_prediction", callback_group=self.cb_group
        )
        self.get_logger().info("Node B (FCN Server) 대기 중...")
        while not self.fcn_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info("Node B가 켜질 때까지 기다리는 중...")
        self.get_logger().info("🟢 Node B 확인 완료!")

        # 3. Main 노드를 위한n 서비스 서버 오픈
        self.srv = self.create_service(
            GetPolicyAction,
            "get_policy_action",
            self.handle_get_policy_action,
            callback_group=self.cb_group,
        )
        self.get_logger().info("🟢 Node A (Policy Server) 준비 완료. 요청 대기 중...")

        self.get_logger().info("RL Policy Manager 초기화 완료.")

    async def handle_get_policy_action(
        self, request: GetPolicyAction.Request, response: GetPolicyAction.Response
    ):
        target_id = request.target_id
        self.get_logger().info(
            f"[A] Main으로부터 추론 요청 수신 (Target ID: {target_id})"
        )

        # 1. Node B(FCN)에 결과 요청
        fcn_req = GetFCNResult.Request()
        fcn_req.weight = [1.0, 1.0, 1.0, 1.0, 1.0]  # 기본 가중치
        fcn_req.target_class_idx = target_id

        self.get_logger().info(f"   -> Node B에 FCN 결과 요청 중...")
        future = self.fcn_client.call_async(fcn_req)

        try:
            fcn_res: GetFCNResult.Response = await future  # B 노드 응답 대기 (비동기)
        except Exception as e:
            self.get_logger().error(f"Node B 호출 실패: {e}")
            return response

        # float32[] data
        fcn_data = fcn_res.data

        # 4. 결과 반환 (Main으로)
        response.action_type = 0  # 0 is Grasp (Fixed)
        response.target_column = int(
            np.argmax(fcn_data)
        )  # FCN 결과에서 가장 높은 값의 인덱스를 열로 사용
        response.one_d_pdm = fcn_res.data
        response.one_d_image = fcn_res.one_d_image
        response.two_d_image = fcn_res.two_d_image

        self.get_logger().info(
            f"✅ [A] 추론 완료! 반환 값: Action={response.action_type}, Column={response.target_column}"
        )
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
