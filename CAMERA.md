ros2 launch realsense2_camera rs_launch.py rgb_camera.color_profile:="1280,720,30" rgb_camera.enable_auto_exposure:=false pointcloud.enable:=false camera_name:="camera1" 


ros2 launch realsense2_camera rs_launch.py camera_name:="camera1" pointcloud.enable:=true rgb_camera.color_profile:="1280,720,30" depth_module.depth_profile:="1280,720,30" rgb_camera.enable_auto_exposure:=true rgb_camera.exposure:="100"

/camera/camera1

# AUTO
ros2 param set /camera/camera1 rgb_camera.enable_auto_exposure true
ros2 param set /camera/camera1 rgb_camera.enable_auto_white_balance true


# 자동 노출 끄기
ros2 param set /camera/camera rgb_camera.enable_auto_exposure false

# 수동 노출 값 설정 (예: 150)
ros2 param set /camera/camera rgb_camera.exposure 150


# 자동 화이트 밸런스 끄기
ros2 param set /camera/camera rgb_camera.enable_auto_white_balance false

# 수동 WB 값 설정 (예: 4600)
ros2 param set /camera/camera rgb_camera.white_balance 4600.0

# Saturation
ros2 param set /camera/camera rgb_camera.saturation 50

# Contrast
ros2 param set /camera/camera rgb_camera.contrast 50


# RESET
ros2 param set /camera/camera rgb_camera.enable_auto_exposure false
ros2 param set /camera/camera rgb_camera.exposure 100
ros2 param set /camera/camera rgb_camera.enable_auto_white_balance false
ros2 param set /camera/camera rgb_camera.white_balance 4500.0
ros2 param set /camera/camera rgb_camera.saturation 50
ros2 param set /camera/camera rgb_camera.contrast 50

# AUTO
ros2 param set /camera/camera rgb_camera.enable_auto_exposure true
ros2 param set /camera/camera rgb_camera.enable_auto_white_balance true