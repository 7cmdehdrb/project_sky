
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2


class ActionCamNode(Node):
	def __init__(self, device: str = '/dev/video7'):
		super().__init__('action_cam_node')
		
		self._video_device = device
		
		self.publisher_ = self.create_publisher(Image, '/action_cam_node/top_view_image', 10)
		self.bridge = CvBridge()
		self.cap = cv2.VideoCapture(device)

		if not self.cap.isOpened():
			self.get_logger().error(f'Cannot open {device}')
			raise RuntimeError(f'Cannot open {device}')
		
		self.timer = self.create_timer(0.05, self.timer_callback)  # 20Hz

	def timer_callback(self):
		ret, frame = self.cap.read()
		if not ret:
			self.get_logger().warning(f'Failed to read frame from {self._video_device}')
			return
		
		# OpenCV는 BGR, ROS는 RGB를 기대함
		frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
		frame_rgb = cv2.resize(frame_rgb, (640, 480))

		msg = self.bridge.cv2_to_imgmsg(frame_rgb, encoding='rgb8')
		self.publisher_.publish(msg)

	def destroy_node(self):
		if hasattr(self, 'cap') and self.cap.isOpened():
			self.cap.release()
		super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = ActionCamNode(device="/dev/video6")
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
	main()
