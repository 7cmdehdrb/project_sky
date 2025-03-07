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
from object_tracker.real_time_segmentation import (
    RealTimeSegmentationNode,
)
import os
import sys
import json
from enum import Enum
from matplotlib import pyplot as plt
import cv2
import cv_bridge
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
        package_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "../resource")
        )
        model_path = os.path.join(package_path, "best_model.pth")
        self.state_dict: dict = torch.load(
            model_path,
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

        grid_data_path = os.path.join(package_path, "grid_data.json")
        with open(grid_data_path, "r") as f:
            self.grid_data = json.load(f)

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
        self.image_publisher = self.create_publisher(
            Image,
            self.get_name() + "/processed_image",
            qos_profile=qos_profile_system_default,
        )
        self.plot_publisher = self.create_publisher(
            Image,
            self.get_name() + "/plot_image",
            qos_profile=qos_profile_system_default,
        )
        self.temp_col_subscription = self.create_subscription(
            UInt16,
            self.get_name() + "/moving_col",
            self.temp_col_callback,
            qos_profile=qos_profile_system_default,
        )
        self.image = None

        # Post-processed data
        self.target_output = None
        self.post_processed_data = None
        self.top_peak_idx = [i for i in range(len(self.grid_data["columns"]))]

        self.iteration_num = 0
        self.moving_col = -1

        self.get_logger().info("FCN Server Node has been initialized.")

        self.create_timer(1.0, self.publish_processed_image)

    def publish_processed_image(self):
        if self.target_output is None or self.post_processed_data is None:
            return

        # Convert the target output to a ROS2 Image message
        target_output_normalized = cv2.normalize(
            self.target_output, None, 0, 255, cv2.NORM_MINMAX
        ).astype(np.uint8)
        target_output_colored = cv2.applyColorMap(
            target_output_normalized, cv2.COLORMAP_JET
        )
        msg = self.bridge.cv2_to_imgmsg(target_output_colored, encoding="bgr8")

        self.image_publisher.publish(msg)

        # Convert the post-processed data to a ROS2 Image message
        fig = plt.figure(figsize=(16, 9))
        plt.plot(self.post_processed_data)

        for peak_idx in self.top_peak_idx:
            plt.axvline(x=peak_idx, color="r", linestyle="--", linewidth=5)

        plt.xlabel("Pixel")
        plt.ylabel("Intensity")
        plt.title("Post-processed Data")

        # Convert the plot to a ROS2 Image message
        fig.canvas.draw()
        plot_image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        plot_image = plot_image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plot_image_msg = self.bridge.cv2_to_imgmsg(plot_image, encoding="rgb8")

        self.plot_publisher.publish(plot_image_msg)
        plt.close(fig)

    def temp_col_callback(self, msg: UInt16):
        self.moving_col = msg.data

        self.get_logger().info(f"Force set moving column: {self.moving_col}")

    def image_callback(self, msg: Image):
        self.image = msg

    def fcn_request_callback(
        self, request: FCNRequest.Request, response: FCNRequest.Response
    ):
        self.get_logger().info(
            f"Request Received - target class is {str(request.target_cls)}"
        )

        if str(request.target_cls) == "exit":
            response.target_col = -1
            response.empty_cols = []
            self.iteration_num = 0
            return response

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
        img = RealTimeSegmentationNode.crop_image(img, 640, 480)

        # Predict the image
        outputs = self.predict(img)

        # Get the target output
        target_output = outputs[target_cls_key]

        # Set post-processed data: the target output
        self.target_output = target_output

        # Post-process the results
        weights = [1.0] * len(self.grid_data["columns"])
        if self.iteration_num != 0 and self.moving_col != -1:
            weights[self.moving_col] = 0.3

        target_col, empty_cols = self.post_process_results(
            target_output, np.array(weights)
        )

        # Set the response
        response.target_col = target_col
        response.empty_cols = empty_cols

        self.get_logger().info(
            f"Return response - {str(request.target_cls)} is located in colum called {empty_cols}.\n\
                move it to colunm  called {target_col}"
        )

        self.iteration_num += 1

        return response

    def post_process_results(self, results: np.ndarray, weights: list) -> np.ndarray:
        data = np.sum(results, axis=0)

        num_peaks = len(self.grid_data["columns"])

        # Find the top peaks and apply weights
        top_peak_idx, top_peak_datas = self.find_top_peaks(
            data, num_peaks=num_peaks, smooth_sigma=5, min_distance=10
        )
        top_peak_datas = top_peak_datas * weights

        # Sort the top_peak_idx and top_peak_datas in ascending order of top_peak_idx
        sorted_indices = np.argsort(top_peak_idx)
        top_peak_idx = np.array(top_peak_idx)[sorted_indices]
        top_peak_datas = np.array(top_peak_datas)[sorted_indices]

        # Set post-processed data: the post-processed data
        self.post_processed_data = data
        self.top_peak_idx = top_peak_idx

        max_peak_idx = int(np.argmax(top_peak_datas))

        res = []
        res1 = max_peak_idx - 1
        res2 = max_peak_idx + 1

        if not (res1 < 0):
            res.append(res1)

        if not (res2 > num_peaks):
            res.append(res2)

        # target_col, empty_cols
        return max_peak_idx, res

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

    def find_top_peaks(self, data, num_peaks=4, smooth_sigma=5, min_distance=10):
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
