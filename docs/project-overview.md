# 프로젝트 전체 설명

## 프로젝트 목적

**DRL-Occluded-Object-Search**는 ROS 2 기반의 가려진 객체 탐색 및 로봇 조작 파이프라인입니다. RGB 이미지, Depth 이미지, PointCloud를 이용해 환경과 객체 상태를 인식하고, FCN과 DRL 정책을 통해 목표 객체를 찾기 위한 행동을 결정한 뒤 UR5e 로봇과 Robotiq 그리퍼 제어 계층으로 연결하는 구조를 지향합니다.

이 프로젝트의 핵심 목표는 다음과 같습니다.

- 카메라 입력에서 객체 탐지 및 segmentation 결과를 생성합니다.
- Depth 및 PointCloud 기반으로 객체의 전방 거리와 column별 상태를 계산합니다.
- FCN 모델로 목표 객체의 위치 가능성 분포를 예측합니다.
- DRL 정책으로 현재 상태에서 수행할 행동과 대상 column을 결정합니다.
- Drop Grid와 로봇 제어 패키지를 통해 집은 물체의 배치 및 실제 조작 시퀀스로 확장합니다.

## 전체 시스템 흐름

```text
RGB Image --------------------> yolo_node -------------------> segmented bbox / image
   |                                                                  |
   |                                                                  v
   +--------------------------> fcn_node <------------------------ drl_node

Depth Image ------------------> closest_object_node ------------> closest object ids

PointCloud -------------------> grid_node ----------------------> front object distance

Drop request -----------------> drop_grid_node -----------------> next drop cell

Policy / sequence result ----> robot_control -------------------> UR5e + Robotiq gripper
```

시스템은 크게 **비동기 상태 갱신 파이프라인**과 **동기 추론 요청 파이프라인**으로 구성됩니다.

1. `yolo_node`, `closest_object_node`, `grid_node`가 ROS topic을 통해 환경 상태를 지속적으로 갱신합니다.
2. `fcn_node`, `drl_node`, `drop_grid_node`는 service 기반으로 필요한 시점에 추론 또는 의사결정을 수행합니다.
3. `robot_control` 패키지는 정책 결과를 실제 로봇 동작으로 연결하는 상위 제어 계층입니다.

## 주요 ROS 2 패키지

### `base_package`

공통 유틸리티 패키지입니다. 이미지 변환, 객체 이름/ID 관리, transform 처리 등 여러 노드가 공유하는 기반 클래스를 포함합니다.

주요 구성:

- `image_manager.py`: ROS Image와 OpenCV 이미지 사이의 변환 및 이미지 전처리 보조 기능
- `object_manager.py`: 객체 클래스 이름, ID, 색상 등 객체 메타데이터 관리
- `transform_manager.py`: TF 및 좌표 변환 관련 보조 기능

### `custom_msgs`

프로젝트 전용 ROS 메시지와 서비스를 정의합니다.

주요 메시지:

- `BoundingBox.msg`
- `BoundingBoxMultiArray.msg`
- `BoundingBox3D.msg`
- `BoundingBox3DMultiArray.msg`

주요 서비스:

- `GetFCNResult.srv`: FCN 위치 분포 추론 요청/응답
- `GetPolicyAction.srv`: DRL 정책 행동 요청/응답
- `GetNextDropCell.srv`: 다음 drop cell 요청/응답

### `object_tracker`

RGB/Depth 기반 객체 인식 및 시각화 노드 패키지입니다.

주요 노드:

- `yolo_node.py`: RGB 이미지에서 YOLO 기반 객체 탐지 및 segmentation 수행
- `closest_object_node.py`: segmentation mask와 Depth를 결합해 column별 가장 가까운 객체 ID 계산
- `integration_image_node.py`: 여러 시각화 topic을 하나의 디버그 이미지로 통합
- `action_cam.py`: 액션 카메라 또는 카메라 보조 기능 관련 노드

### `fcn_network`

FCN, DRL, grid, drop grid 계층을 포함하는 추론 중심 패키지입니다.

주요 노드:

- `fcn_node.py`: 목표 객체의 위치 가능성 분포를 서비스로 제공
- `drl_node.py`: FCN 결과와 현재 환경 상태를 결합해 정책 행동 결정
- `grid_node.py`: PointCloud를 grid 형태로 해석하고 전방 거리 상태 발행
- `drop_grid_node.py`: 물체를 내려놓을 grid cell 선택
- `random_node.py`, `xray_node.py`: 실험 또는 대체 추론 흐름에 사용되는 보조 노드

### `mcts`

Monte Carlo Tree Search 기반 의사결정 실험 패키지입니다. DRL 정책과 별개로 탐색 기반 정책을 실험하거나 비교하기 위한 구조로 볼 수 있습니다.

### `robot_control`

UR5e와 Robotiq gripper를 실제로 움직이기 위한 상위 제어 패키지입니다.

주요 파일:

- `main.py`: 전체 조작 플로우의 진입점
- `controller.py`: 로봇 및 그리퍼 제어 함수 집합
- `action_sequence.py`: 정책 결과를 실제 조작 순서로 변환하는 시퀀스 로직
- `launch/`: static TF 및 UR startup launch 파일

### `ros2_robotiq_gripper-humble`

Robotiq gripper ROS 2 Humble 연동용 외부/벤더 패키지입니다. gripper description, driver, controller, serial 통신 관련 구성이 포함되어 있습니다.

### `ur5e_robotiq_config`

UR5e와 Robotiq gripper의 MoveIt/로봇 설정 패키지입니다. SRDF, kinematics, planning pipeline, controller 설정, RViz 설정 등이 포함됩니다.

## 핵심 노드 책임 요약

| 노드 | 패키지 | 책임 | 주요 출력 |
| --- | --- | --- | --- |
| `yolo_node` | `object_tracker` | RGB 이미지 기반 객체 탐지/분할 | segmented image, bounding boxes |
| `closest_object_node` | `object_tracker` | Depth와 mask 기반 column별 가까운 객체 판정 | closest object ids, overlay image |
| `integration_image_node` | `object_tracker` | 디버그 이미지 통합 | integrated image |
| `grid_node` | `fcn_network` | PointCloud 기반 grid/거리 계산 | front object distance, markers |
| `fcn_node` | `fcn_network` | 목표 객체 위치 분포 추론 | FCN service response, visualization |
| `drl_node` | `fcn_network` | DRL 정책 행동 결정 | policy action service response |
| `drop_grid_node` | `fcn_network` | drop 위치 선택 | next drop cell service response |
| `mcts_node` | `mcts` | MCTS 기반 정책 실험 | MCTS decision |
| `main` | `robot_control` | 최상위 로봇 조작 실행 | robot/gripper action |

## 참고 문서

- [아키텍처 및 노드 구동 원리](./architecture.md)
- [디렉토리 구조 설명](./directory-structure.md)
- [카메라 관련 메모](./camera.md)
- [명령어 메모](./commands.md)
