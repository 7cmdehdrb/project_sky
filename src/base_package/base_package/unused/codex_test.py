import cv2


def test_codec(codec_str, filename="test_output.mp4"):
    fourcc = cv2.VideoWriter_fourcc(*codec_str)
    out = cv2.VideoWriter(filename, fourcc, 20.0, (640, 480))
    if out.isOpened():
        print(f"[OK] {codec_str} works.")
    else:
        print(f"[FAIL] {codec_str} failed.")
    out.release()


codecs_to_test = ["XVID", "MJPG", "mp4v", "avc1", "H264", "X264"]

for codec in codecs_to_test:
    test_codec(codec, f"test_{codec}.mp4")
