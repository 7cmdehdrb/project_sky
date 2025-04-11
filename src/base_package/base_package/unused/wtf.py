import cv2
import numpy as np

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter("test_output.mp4", fourcc, 30.0, (640, 480))

if not out.isOpened():
    print("Failed to open video writer.")
else:
    for _ in range(60):
        dummy = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        out.write(dummy)
    out.release()
    print("Video saved successfully.")
