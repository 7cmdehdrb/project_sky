# Project Overview

This project consists of multiple ROS packages that support various functionalities, including utility functions, custom messages, neural network inference, and object tracking.


```bash
sudo apt install ros-humble-ros2-control ros-humble-controller-manager
```

## Packages

### 1. `base_package`
This package provides a collection of frequently used utility functions that are commonly called by other packages.

### 2. `custom_msgs`
This package defines all custom message types used in this project, including ROS client messages.

### 3. `fcn_network`
This is the core package responsible for image preprocessing and Fully Convolutional Network (FCN) inference.
- It crops an input image of size `1280x720` to `640x480` before passing it through the FCN model.
- The FCN model can be downloaded via `src/fcn_network/fcn_network/install.py`.

#### Key Components
- **`fcn_server.py`**: Loads the model, processes real-time images, and responds to client requests. It accepts a request parameter `target_cls` (string) and responds with:
  - `target_col` (int): The column where the target object is located. Returns `-1` if an error occurs.
  - `empty_cols` (list of int): A list of columns where the target object can move. Returns an empty list if an error occurs.
  
- **`fcn_client.py`**: Connects to two servers and sequentially sends requests.

- **`pointcloud_grid_identifier.py`**: A ROS server that subscribes to point cloud data and verifies whether the object in `target_col` can move to one of the `empty_cols`.
  - Request structure: Same as `fcn_server.py`'s response.
  - Response structure:
    - `action` (bool): `True` for sweeping, `False` for grasping.
    - `moving_row` (string): The row of the moving object, returns "Z" in case of an error.
    - `moving_cols` (list of int): Columns where the object can move, returns `[-1]` in case of an error.

### 4. `object_tracker`
A client package for using `megapose6d` for object tracking.

#### Key Components
- **`real_time_segmentation.py`**: Performs real-time segmentation using YOLOv11 and publishes the resulting bounding box data.
- **`real_time_tracking_client.py`**: A WebSocket client that directly communicates with the `megapose` server. It initializes tracking using bounding boxes received from `real_time_segmentation.py` and publishes pose data.

## Submodules

### `third_party/visp`
This submodule enables the `megapose6d` server functionality.

#### Installation & Execution Guide

- Install essential packages
```bash
sudo apt-get install build-essential cmake-curses-gui git wget
```

- Set environment (1)
```bash
echo "export VISP_WS=$YOUR-VISP-DIRECTORY/visp-ws" >> ~/.bashrc
source ~/.bashrc
mkdir -p $VISP_WS
```

- Install build-dependency packages
```bash
sudo apt-get install libopencv-dev libx11-dev liblapack-dev libeigen3-dev libv4l-dev  libzbar-dev libpthread-stubs0-dev libdc1394-dev nlohmann-json3-dev
```

- Build. IMPORTANT: DO NOT forget to put ".." when run <i>cmake</i> 
```bash
mkdir -p $VISP_WS/visp-build
cd $VISP_WS/visp-build
cmake ..
make -j$(nproc)
```

- Set environment (2)
```bash
echo "export VISP_DIR=$VISP_WS/visp-build" >> ~/.bashrc
source ~/.bashrc
```

- Install megapose server. This installation will take several minutes
```bash
cd $VISP_WS/script/megapose_server
python $VISP_WS/script/megapose_serverinstall.py
```

- Activate conda environment. Megapose server only runs on this conda environment
```bash
conda activate megapose
```

```bash
python $VISP_WS/script/megapose_server/run.py --host 127.0.0.1 --port 5555 --model RGB --meshes-directory YOUR-MESHES-DIRECTORY
```

This setup installs dependencies, builds the `visp` workspace, and runs the `megapose6d` server with the appropriate model and mesh directory configuration.

---

This README provides an overview of the project's structure and functionalities. For further details, refer to the corresponding package documentation.



FUCKING TEST

1. Launch UR5e

```bash
ros2 launch robot_control robot_launch.launch.py 
```

2. Launch Gripper

```bash
ros2 launch robotiq_description robotiq_control.launch.py launch_rviz:=false
```

3. Launch Main Camera

```bash
ros2 launch realsense2_camera rs_launch.py camera_name:="camera1" pointcloud.enable:=true rgb_camera color_profile:="1280,720,30" depth_module.depth_profile:="1280,720,30" rgb_camera enable_auto_exposure:=false rgb_camera.exposure:="100" usb_port_id:="6-3.3"
```
4. Launch Side Camera

```bash
ros2 launch realsense2_camera rs_launch.py camera_name:="camera2" rgb_camera.color_profile:="1280,720,30" rgb_camera.enable_auto_exposure:=true usb_port_id:="6-3.1"
```

5. Run Action Camera

```bash
python3 /home/irol/workspace/project_sky/src/base_package/base_package/unused/video_to_ros.py --topic /action_camera/color/image_raw --video /dev/video6
```

6. Launch FCM

```bash
ros2 launch fcn_network fcn_network.launch.py model_file:=/home/irol/workspace/project_sky/src/fcn_network/resource/best_model_0414_grasping_only.pth grid_data_file:=/home/irol/workspace/project_sky/src/fcn_network/resource/grid_data.json fcn_gamma:=1.0
```

- If you use DRL,

```bash
ros2 launch fcn_network direct_fcn_network.launch.py model_file:=/home/irol/workspace/project_sky/src/fcn_network/resource/best_model_0414_grasping_only.pth grid_data_file:=/home/irol/workspace/project_sky/src/fcn_network/resource/grid_data.json fcn_gamma:=1.0
```

7. Launch Object Tracker

```bash
ros2 launch object_tracker object_tracker.launch.py model_file:=/home/irol/workspace/project_sky/src/object_tracker/resource/best_hg.pt grid_data_file:=/home/irol/workspace/project_sky/src/fcn_network/resource/grid_data.json obj_bounds_file:=/home/irol/workspace/project_sky/src/object_tracker/resource/obj_bounds.json
```

8. Launch Log Server

```bash
python3 /home/irol/workspace/project_sky/src/robot_control/robot_control/log_server.py --exp_attempt <YOUR-EXPERIMENT-NUMBER>
```

9. Launch Video Record

```bash
python3 /home/irol/workspace/project_sky/src/base_package/base_package/unused/video_recoder.py --record <TRUE-IF-YOU-WANT-TO-RECORD> --exp_attempt <YOUR-EXPERIMENT-NUMBER>
```

10. RUN MAIN CODE

```bash
python3 /home/irol/workspace/project_sky/src/robot_control/robot_control/main.py --model_file /home/irol/workspace/project_sky/src/fcn_network/resource/best_model_0414_grasping_only.pth --grid_data_file /home/irol/workspace/project_sky/src/fcn_network/resource/grid_data.json --drop_grid_data_file /home/irol/workspace/project_sky/src/fcn_network/resource/drop_grid_data.json --debug false --mode <MODE-TO-RUN> --target_cls <TARGET-CLASS-NAME> 
```