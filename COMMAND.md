## Camera
`
cd /home/irol/test
source install/setup.bash
ros2 launch realsense2_camera rs_launch.py camera_name:="camera1" pointcloud.enable:=true rgb_camera.color_profile:="1280,720,30" depth_module.depth_profile:="1280,720,30" rgb_camera.enable_auto_exposure:=true
`

`
python3 /home/irol/project_sky/src/test/integration_image_node.py
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

## FCN
`
python3 src/fcn_network/fcn_network/grid_node.py
`

`
python3 src/fcn_network/fcn_network/fcn_node.py
`

`
 
`


## TEST
`
python3 src/fcn_network/fcn_network/test_drl_node.py
`

