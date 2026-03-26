## Camera
`
cd /home/irol/test
source install/setup.bash
ros2 launch realsense2_camera rs_launch.py camera_name:="camera1" pointcloud.enable:=true rgb_camera.color_profile:="1280,720,30" depth_module.depth_profile:="1280,720,30" rgb_camera.enable_auto_exposure:=true
`

`
cd /home/irol/test
source install/setup.bash
ros2 launch realsense2_camera rs_launch.py camera_name:="camera1" pointcloud.enable:=true rgb_camera.color_profile:="1280,720,30" depth_module.depth_profile:="1280,720,30" rgb_camera.enable_auto_exposure:=false rgb_camera.exposure:="100"
`

`
python3 /home/min/7cmdehdrb/project_sky/src/test/integration_image_node.py
`

## Static TF
`
ros2 run tf2_ros static_transform_publisher -0.04 -0.37 0.45 0.0 0.0 0.7071 0.7071 world camera1_link
`


## Object
`
python3 src/object_tracker/object_tracker/yolo_node.py
`

`
python3 src/object_tracker/object_tracker/closest_object_node.py
`

`
python3 src/test/integration_image_node.py
`

## FOR TEST

`
python3 src/object_tracker/object_tracker/unused/fake_camera_publisher.py
`

`
python3 src/object_tracker/object_tracker/unused/fake_depth_publisher.py
`


## FCN
`
python3 src/fcn_network/fcn_network/grid_node.py
`

`
python3 src/fcn_network/fcn_network/fcn_node.py
`

`
python3 src/fcn_network/fcn_network/drl_node.py
`
`
python3 src/fcn_network/fcn_network/drop_grid_node.py
`

## TEST
`
python3 src/fcn_network/fcn_network/test_drl_node.py
`




`
cd ~/7cmdehdrb/project_sky
source /opt/ros/humble/setup.bash
colcon build --allow-overriding ur_description
`

`
source /opt/ros/humble/setup.bash

colcon build \
  --packages-select serial \
  --cmake-args -DCMAKE_POSITION_INDEPENDENT_CODE=ON
`

`
source install/setup.bash

colcon build \
  --packages-skip serial \
  --allow-overriding ur_description
`

`
LIBGL_ALWAYS_SOFTWARE=1 ros2 run moveit_setup_assistant moveit_setup_assistant
`


---

## SSH

sudo systemctl restart ssh
sudo systemctl restart ssh.socket


sudo systemctl start ssh
sudo systemctl start ssh.socket

sudo systemctl stop ssh
sudo systemctl stop ssh.socket

scp -P 2222 FILE ssu@dhlee04.iptime.org:/home/ssu


scp -P 2222 /home/min/7cmdehdrb/project_sky/src/test/yolo/python_yolo.zip ssu@dhlee04.iptime.org:/home/ssu
scp -P 2222 /home/min/7cmdehdrb/project_sky/src/test/yolo/yolo_dataset.zip ssu@dhlee04.iptime.org:/home/ssu


/home/min/7cmdehdrb/project_sky/src/test/yolo/python_yolo.zip
/home/min/7cmdehdrb/project_sky/src/test/yolo/yolo_dataset.zip