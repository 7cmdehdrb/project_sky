# ROS2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from rclpy.qos import qos_profile_system_default

# OpenCV
import cv2
from cv_bridge import CvBridge

# Python
import numpy as np
import json
from PIL import ImageEnhance
from PIL import Image as PILImage

# YOLO
from ultralytics.engine.results import Results, Masks, Boxes
from ultralytics import YOLO


class CameraViewer(Node):
    def __init__(self):
        super().__init__("camera_viewer_node")

        # YOLO 모델 로드
        model_path = "/home/irol/workspace/project_sky/src/object_tracker/resource/weights/best.pt"
        model_path = "/home/irol/workspace/project_sky/src/object_tracker/resource/weights/best_hg.pt"
        self.model = YOLO(model_path)
        self.bridge = CvBridge()

        with open(
            "/home/irol/workspace/project_sky/src/object_tracker/resource/sim_stats.json",
            "r",
        ) as f:
            stats = json.load(f)
            self.stats = stats

        # ROS
        self.publisher_ = self.create_publisher(
            Image, "/processed_image", qos_profile=qos_profile_system_default
        )

        self.subscriber = self.create_subscription(
            Image,
            "/camera/camera1/color/image_raw",
            self.image_callback,
            qos_profile=qos_profile_system_default,
        )
        self.image = None

        self.color_dict = {
            0: (128, 0, 0),
            1: (0, 128, 0),
            2: (0, 0, 128),
            3: (128, 128, 0),
            4: (128, 0, 128),
            5: (0, 128, 128),
            6: (192, 192, 192),
            7: (128, 128, 128),
            8: (0, 0, 0),
            9: (255, 255, 255),
            10: (64, 64, 64),
            11: (255, 0, 0),
            12: (0, 255, 0),
            13: (0, 0, 255),
            14: (255, 255, 0),
        }

        # self.create_timer(1.0, self.image_loop)

    def image_callback(self, msg: Image):
        self.image = msg

        self.image_loop()

    def image_loop(self):
        if self.image is None:
            return None

        msg = self.image

        # Load Image
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")
        cv_image_array = np.asanyarray(cv_image)

        assert cv_image_array.shape[0] == 720 and cv_image_array.shape[1] == 1280

        # Crop Image
        h, w = cv_image_array.shape[:2]  # h=720, w=1280
        crop_w, crop_h = 640, 480
        start_x = int((w - crop_w) // 2.05)
        start_y = int((h - crop_h) // 2.7)

        # 크롭 범위 보정
        start_x = max(0, min(start_x, w - crop_w))
        start_y = max(0, min(start_y, h - crop_h))

        # 이미지 크롭 (640x480)
        cropped_image_array = cv_image_array[
            start_y : start_y + crop_h, start_x : start_x + crop_w
        ]

        modify_color = False

        if modify_color:
            # YOLO 모델 입력을 위해 RGB 변환
            cropped_pil_image = PILImage.fromarray(cropped_image_array)
            img_hsv = cv2.cvtColor(cropped_image_array, cv2.COLOR_RGB2HSV)

            # Color Statistics
            curr_brightness = np.mean(img_hsv[:, :, 2])
            curr_saturation = np.mean(img_hsv[:, :, 1])
            curr_contrast = np.std(img_hsv[:, :, 2])

            # enhancement factor 계산 (0으로 나누는 경우 방지)
            brightness_factor = self.stats["avg_brightness"] / (curr_brightness + 1e-8)
            saturation_factor = self.stats["avg_saturation"] / (curr_saturation + 1e-8)
            contrast_factor = self.stats["avg_contrast"] / (curr_contrast + 1e-8)

            # PIL ImageEnhance 모듈로 순차적으로 조정
            # 1) 명도 보정
            enhancer = ImageEnhance.Brightness(cropped_pil_image)
            cropped_pil_image = enhancer.enhance(brightness_factor)

            # 2) 채도 보정
            enhancer = ImageEnhance.Color(cropped_pil_image)
            cropped_pil_image = enhancer.enhance(saturation_factor)

            # 3) 대조 보정
            enhancer = ImageEnhance.Contrast(cropped_pil_image)
            cropped_pil_image = enhancer.enhance(contrast_factor)

        else:
            cropped_pil_image = PILImage.fromarray(cropped_image_array)

        # YOLO 세그멘테이션 수행
        results = self.model(cropped_pil_image)

        if len(results) == 0:
            self.get_logger().info("No object detected.")
            return None

        original_image_rgb = PILImage.fromarray(cropped_image_array)
        original_image_bgr = cv2.cvtColor(
            np.array(original_image_rgb), cv2.COLOR_RGB2BGR
        )
        original_image_array = np.array(original_image_bgr)

        result: Results = results[0]

        boxes: Boxes = result.boxes
        classes: dict = result.names

        np_boxes = boxes.xyxy.cpu().numpy()
        np_confs = boxes.conf.cpu().numpy()
        np_cls = boxes.cls.cpu().numpy()

        # 바운딩 박스 그리기
        if boxes is None:
            self.get_logger().info("No box detected.")
            return None

        for idx in range(len(boxes)):
            conf = np_confs[idx]  # 신뢰도
            cls = classes[int(np_cls[idx])]
            x1, y1, x2, y2 = map(int, np_boxes[idx])

            if conf < 0.7:
                continue

            label = f"{cls}, {conf:.2f}"
            cv2.rectangle(
                original_image_array,
                (x1, y1),
                (x2, y2),
                self.color_dict[int(np_cls[idx])],
                2,
            )
            cv2.putText(
                img=original_image_array,
                text=label,
                org=(x1, y1 - 10),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.5,
                color=self.color_dict[int(np_cls[idx])],
                thickness=2,
            )

        # 결과를 ROS 2 이미지 메시지로 변환 및 퍼블리시
        # img_msg = self.bridge.cv2_to_imgmsg(original_image_array, encoding="bgr8")
        # self.publisher_.publish(img_msg)

        # 결과 화면 출력
        cv2.imshow("YOLOv11 Segmentation (Cropped 640x480)", original_image_array)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            rclpy.shutdown()


def main(args=None):
    rclpy.init(args=args)

    node = CameraViewer()

    rclpy.spin(node=node)

    node.destroy_node()


if __name__ == "__main__":
    main()
