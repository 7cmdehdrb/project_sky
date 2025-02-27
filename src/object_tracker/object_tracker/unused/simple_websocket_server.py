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

# TF
from tf2_ros import *

# Python
import numpy as np

import asyncio
import websockets


class SimpleWebsocketServer(Node):
    def __init__(self):
        super().__init__("simple_websocket_server")
        self.get_logger().info("Simple Websocket Server Node has been initialized.")

        self.int_pub = self.create_publisher(
            UInt16, "/first_trigger", qos_profile=qos_profile_system_default
        )
        # self.create_timer(0.01, self.send_ros_message)
        # self.data = None

        asyncio.run(self.main())

    async def echo(self, websocket):
        async for message in websocket:
            print(f"Received: {message}")
            self.int_pub.publish(UInt16(data=int(message)))
            await websocket.send(f"Echo: {message}")

    async def main(self):
        async with websockets.serve(self.echo, "localhost", 8765):
            print("WebSocket server started on ws://localhost:8765")
            await asyncio.Future()  # 서버를 계속 실행


def main():
    rclpy.init(args=None)

    node = SimpleWebsocketServer()

    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
