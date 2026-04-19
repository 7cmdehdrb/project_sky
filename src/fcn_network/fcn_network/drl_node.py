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
                    "/home/min/7cmdehdrb/project_sky/src/fcn_network/resource/exported_45/policy.onnx",
                ),
                (
                    "weight_fcn",
                    [1.0, 1.0, 1.0, 1.0, 1.0],
                ),
            ],
        )

        self.cb_group = ReentrantCallbackGroup()

        # 모델 경로 수정 필요
        self._weight_fcn = list(
            self.get_parameter("weight_fcn").get_parameter_value().double_array_value
        )
        model_path = self.get_parameter("model_path").get_parameter_value().string_value
        self.policy_manager = RLPolicyManager(model_path)

        # 1. 구독 (Observations)
        self.dist_sub = self.create_subscription(
            Float32MultiArray,
            "/front_object_distance",  # 미정 토픽명
            self.distance_callback,
            qos_profile=qos_profile_system_default,
            callback_group=self.cb_group,
        )

        self.obj_sub = self.create_subscription(
            Int32MultiArray,
            "/closest_object_classifier/closest_object_ids",
            self.object_id_callback,
            qos_profile=qos_profile_system_default,
            callback_group=self.cb_group,
        )

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
        self.get_logger().info(f"모델 경로: {model_path}")

    def distance_callback(self, msg: Float32MultiArray):
        # Float Array -> Policy Manager 저장
        self.policy_manager.front_object_distance = msg.data

    def object_id_callback(self, msg: Int32MultiArray):
        # Int Array -> Float 캐스팅 후 Policy Manager 저장
        self.policy_manager.front_object = [float(x) for x in msg.data]

    async def handle_get_policy_action(
        self, request: GetPolicyAction.Request, response: GetPolicyAction.Response
    ):
        exp_index: int = request.index
        if exp_index == 0:
            self.get_logger().info(f"▶️ Episode {exp_index} 시작 - 상태 초기화")
            self.policy_manager.reset_states()  # Episode 시작 시 상태 초기화

        target_id = request.target_id
        self.get_logger().info(
            f"[A] Main으로부터 추론 요청 수신 (Target ID: {target_id})"
        )

        # 1. Node B(FCN)에 결과 요청
        fcn_req = GetFCNResult.Request()
        fcn_req.weight = self._weight_fcn
        fcn_req.target_class_idx = target_id

        self.get_logger().info(f"   -> Node B에 FCN 결과 요청 중...")
        future = self.fcn_client.call_async(fcn_req)

        try:
            fcn_res: GetFCNResult.Response = await future  # B 노드 응답 대기 (비동기)
        except Exception as e:
            self.get_logger().error(f"Node B 호출 실패: {e}")
            return response

        # 2. B의 응답 및 Main의 요청 데이터를 Policy Manager에 주입
        self.policy_manager.column_distribution = fcn_res.data
        self.policy_manager.target_id = target_id

        self.get_logger().info(
            f"Observation - column_distribution: {self.policy_manager.column_distribution}"
        )
        self.get_logger().info(
            f"Observation - target_id: {self.policy_manager.target_id}"
        )
        self.get_logger().info(
            f"Observation - front_object_distance: {self.policy_manager.front_object_distance}"
        )
        self.get_logger().info(
            f"Observation - front_object: {self.policy_manager.front_object}"
        )

        # 3. 모델 1회 추론 및 내부 상태(t-1) 자동 갱신
        self.get_logger().info(f"   -> RL Policy 추론 진행...")
        policy_action = self.policy_manager.request_action()

        # 4. 결과 반환 (Main으로)
        response.action_type = policy_action.action_type
        response.target_column = policy_action.target_column
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
