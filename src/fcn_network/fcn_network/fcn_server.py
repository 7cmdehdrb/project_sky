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
from custom_msgs.srv import FCNRequest

# TF
from tf2_ros import *

# Python
import cv2
import cv_bridge
from enum import Enum
import numpy as np
import torch
from torchvision.transforms import Normalize
from torchvision.models.segmentation import fcn_resnet50
from torch import nn, Tensor
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d


# FCNModel class
class FCNModel(nn.Module):
    def __init__(self):
        super(FCNModel, self).__init__()
        self.model = fcn_resnet50(
            weights=None
        )  # pretrained=False는 weights=None으로 대체됨
        self.model.classifier[4] = nn.Conv2d(512, 12, kernel_size=1)

    def forward(self, x):
        return self.model(x)["out"]


class FCNClassNames(Enum):
    CAN_1 = 0
    CAN_2 = 1
    CAN_3 = 2
    CUP_1 = 3
    CUP_2 = 4
    CUP_3 = 5
    MUG_1 = 6
    MUG_2 = 7
    MUG_3 = 8
    BOTTLE_1 = 9
    BOTTLE_2 = 10
    BOTTLE_3 = 11


# Enum에서 인덱스로 클래스 이름 가져오기
def get_class_name(index):
    return FCNClassNames(index).name.lower()


class FCNServerNode(Node):
    def __init__(self):
        super().__init__("fcn_server_node")

        # Initialize the FCN model
        self.model = FCNModel()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Transform Parameters
        self.do_transform = True
        self.transformer = Normalize(
            mean=[0.6501676617516412, 0.6430160918638441, 0.6165616299396091],
            std=[0.16873769158467197, 0.17505241356408263, 0.1989546266815883],
        )

        # Load the pre-trained model
        self.state_dict: dict = torch.load(
            "/home/min/7cmdehdrb/ros2_ws/src/fcn_network/resource/best_model.pth",  # TODO: Change the path
            map_location=self.device,
        )
        self.filtered_state_dict = {
            k: v for k, v in self.state_dict.items() if "aux_classifier" not in k
        }
        self.model.eval()
        self.model.load_state_dict(self.filtered_state_dict, strict=False)
        self.model = self.model.to(self.device)

        # Load CV Bridge
        self.bridge = cv_bridge.CvBridge()

        # ROS2 Subscribers, Publishers, and Services
        self.srv = self.create_service(
            FCNRequest, "fcn_request", self.fcn_request_callback
        )
        self.image_subscription = self.create_subscription(
            Image,
            "/camera/camera1/color/image_raw",
            self.image_callback,
            qos_profile=qos_profile_system_default,
        )
        self.image = None

    def image_callback(self, msg: Image):
        self.image = msg

    def fcn_request_callback(
        self, request: FCNRequest.Request, response: FCNRequest.Response
    ):
        # Define the target class
        target_cls = request.target_cls

        # Get the class name from the target class
        cls_names_dict = {cls.value: cls.name.lower() for cls in FCNClassNames}
        target_cls_key = None
        for key, value in cls_names_dict.items():
            if value == target_cls:
                target_cls_key = key
                break
        if target_cls_key is None:
            response.empty_cols = []
            return response

        # Exception handling
        if self.image is None:
            self.get_logger().warn("No image received.")
            response.target_col = -1
            response.empty_cols = []
            return response

        # Get the image
        img = self.bridge.imgmsg_to_cv2(self.image, "bgr8")
        img = self.post_process_raw_image(img)  # TODO: Crop the image

        # Predict the image
        outputs = self.predict(img)

        # Get the target output
        target_output = outputs[target_cls_key]

        # Post-process the results
        target_col, empty_cols = self.post_process_results(target_output)

        # Set the response
        response.empty_cols = empty_cols
        response.target_col = target_col

        return response

    def post_process_results(self, results: np.ndarray) -> np.ndarray:
        data = np.sum(results, axis=0)

        num_peaks = 4

        _, top_peak_datas = self.find_top_peaks(
            data, num_peaks=num_peaks, smooth_sigma=5, min_distance=10
        )

        max_peak_idx = np.argmax(top_peak_datas)
        # max_peak_value = top_peak_datas[max_peak_idx]

        res = []
        res1 = max_peak_idx - 1
        res2 = max_peak_idx + 1

        if not (res1 < 0):
            res.append(res1)

        if not (res2 > num_peaks):
            res.append(res2)

        return max_peak_idx, results

    def post_process_raw_image(self, img: np.ndarray) -> np.ndarray:
        if img.ndim == 2:
            img = np.stack([img] * 3, axis=-1)
        elif img.shape[-1] == 4:
            img = img[..., :3]

        img = img.astype(np.float32) / 255.0
        img = img.transpose(2, 0, 1)

        if self.do_transform:
            img = torch.tensor(img, dtype=torch.float32)
            img = self.transformer(img)
        else:
            img = torch.tensor(img, dtype=torch.float32)

        return img

    def predict(self, img: np.ndarray) -> np.ndarray:
        img = self.post_process_raw_image(img)
        img: Tensor = Tensor(img).to(self.device)
        img = img.unsqueeze(0)  # Add batch dimension

        outputs: Tensor = self.model(img)
        outputs = outputs.squeeze(0)

        np_outputs = outputs.detach().cpu().numpy()

        return np_outputs

    def find_top_peaks(data, num_peaks=4, smooth_sigma=5, min_distance=10):
        """
        데이터에서 상위 num_peaks개의 주요 피크를 찾는 함수.

        :param data: 1차원 배열 (측정 데이터)
        :param num_peaks: 찾을 피크 개수
        :param smooth_sigma: 가우시안 필터의 표준편차 (노이즈 제거)
        :param min_distance: 피크 간 최소 거리 (작은 봉우리를 무시)
        :return: 상위 num_peaks개의 피크 인덱스와 해당 값
        """

        # 1. 스무딩 적용 (노이즈 완화)
        smoothed_data = gaussian_filter1d(data, sigma=smooth_sigma)

        # 2. 피크 찾기 (높이 및 거리 조건 적용)
        peaks, _ = find_peaks(smoothed_data, distance=min_distance)

        # 3. 상위 num_peaks개의 피크 선택
        top_peaks = sorted(peaks, key=lambda x: data[x], reverse=True)[:num_peaks]

        return top_peaks, data[top_peaks]


def main():
    rclpy.init(args=None)

    node = FCNServerNode()

    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
