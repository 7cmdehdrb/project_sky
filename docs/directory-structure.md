# 디렉토리 구조 설명

이 문서는 저장소 최상단과 `src/` 하위 ROS 2 패키지의 역할을 설명합니다.

## 최상단 구조

```text
.
├── docs/                         # 프로젝트 문서 모음
├── src/                          # ROS 2 workspace source packages
├── console/                      # 실행/디버깅 로그 또는 콘솔 명령 메모
├── requirements.txt              # Python 의존성 목록
├── default.rviz                  # 기본 RViz 설정
├── rviz.rviz                     # RViz 설정
├── chomp_planning.yaml           # CHOMP planning 설정
├── ompl_planning.yaml            # OMPL planning 설정
├── extensions.list               # 개발 환경 확장 목록
├── retest_list                   # 재테스트 항목 메모
└── README.md                     # 문서 허브 역할의 최상단 README
```

## `docs/`

프로젝트 설명 문서를 보관하는 폴더입니다.

```text
docs/
├── project-overview.md           # 프로젝트 전체 설명
├── directory-structure.md        # 디렉토리 구조 설명
├── architecture.md               # 기존 README에서 분리한 아키텍처/노드 구동 원리
├── camera.md                     # 기존 CAMERA.md에서 이동한 카메라 문서
├── commands.md                   # 기존 COMMAND.md에서 이동한 명령어 문서
└── last-readme.md                # 기존 LAST_README.md에서 이동한 참고 문서
```

## `src/base_package/`

여러 패키지가 공통으로 사용하는 기반 유틸리티 패키지입니다.

```text
src/base_package/
├── base_package/
│   ├── image_manager.py          # 이미지 변환/전처리 보조 클래스
│   ├── object_manager.py         # 객체 ID/이름/색상 관리
│   ├── transform_manager.py      # TF 및 좌표 변환 보조 클래스
│   ├── header.py                 # 공통 import/header 모음
│   ├── launch/                   # RealSense 등 실행 보조 launch
│   ├── before/                   # 이전 구현 보관
│   └── unused/                   # 현재 주 실행 경로에서 제외된 실험 코드
├── package.xml
└── setup.py
```

## `src/custom_msgs/`

프로젝트 전용 ROS 인터페이스 정의 패키지입니다.

```text
src/custom_msgs/
├── msg/
│   ├── BoundingBox.msg
│   ├── BoundingBoxMultiArray.msg
│   ├── BoundingBox3D.msg
│   └── BoundingBox3DMultiArray.msg
├── srv/
│   ├── GetFCNResult.srv
│   ├── GetNextDropCell.srv
│   └── GetPolicyAction.srv
├── before/                       # 이전 서비스 정의 보관
├── CMakeLists.txt
└── package.xml
```

## `src/object_tracker/`

카메라 기반 객체 인식 및 디버그 시각화 패키지입니다.

```text
src/object_tracker/
├── object_tracker/
│   ├── yolo_node.py              # YOLO 객체 탐지 및 segmentation 노드
│   ├── closest_object_node.py    # Depth + mask 기반 가장 가까운 객체 판정 노드
│   ├── integration_image_node.py # 여러 시각화 이미지를 통합하는 노드
│   ├── action_cam.py             # 카메라/액션 관련 보조 노드
│   ├── before/                   # 이전 구현 보관
│   └── unused/                   # 실험/비활성 코드
├── launch/
│   └── object_tracker.launch.py  # object tracker 관련 launch
├── resource/                     # 객체 bounds, simulation stats 등 리소스
├── package.xml
└── setup.py
```

## `src/fcn_network/`

FCN, DRL, grid, drop-grid 관련 추론 패키지입니다.

```text
src/fcn_network/
├── fcn_network/
│   ├── fcn_node.py               # FCN service node
│   ├── fcn_manager.py            # FCN 모델 로드/추론/후처리
│   ├── drl_node.py               # 정책 action service node
│   ├── drl_manager.py            # DRL policy 추론 및 state 구성
│   ├── grid_node.py              # PointCloud 기반 거리/grid publisher
│   ├── grid_manager.py           # grid 계산 및 marker 생성
│   ├── drop_grid_node.py         # drop cell service node
│   ├── drop_grid_manager.py      # drop grid 상태/선택 로직
│   ├── random_node.py            # random policy 실험 노드
│   ├── xray_node.py              # 추가 실험/시각화 노드
│   ├── before/                   # 이전 구현 보관
│   └── unused/                   # 실험/비활성 코드
├── launch/                       # 다양한 FCN/DRL/MCTS 조합 launch
├── resource/                     # grid, drop grid, dataset stats 등 리소스
├── html/                         # 웹 기반 시각화 또는 디버그 asset
├── package.xml
└── setup.py
```

## `src/mcts/`

MCTS 기반 탐색 정책 실험 패키지입니다.

```text
src/mcts/
├── mcts/
│   ├── mcts_node.py              # MCTS ROS node
│   ├── mcts_manager.py           # MCTS decision manager
│   └── unused/                   # 이전/실험 코드
├── package.xml
└── setup.py
```

## `src/robot_control/`

UR5e와 Robotiq gripper를 실제 조작 시퀀스로 제어하는 패키지입니다.

```text
src/robot_control/
├── robot_control/
│   ├── main.py                   # 최상위 로봇 조작 진입점
│   ├── controller.py             # 로봇 및 gripper 제어 함수
│   ├── action_sequence.py        # 정책 action을 조작 sequence로 변환
│   ├── gripper_test.py           # gripper 테스트 코드
│   └── before/                   # 이전 제어 구현 보관
├── launch/
│   ├── static_tf_th.launch.py
│   └── ur_startup.launch.py
├── resource/                     # 실행 리소스 및 실험 결과 CSV
├── package.xml
└── setup.py
```

## `src/ros2_robotiq_gripper-humble/`

Robotiq gripper를 ROS 2 Humble에서 사용하기 위한 외부/벤더 패키지 묶음입니다.

```text
src/ros2_robotiq_gripper-humble/
├── robotiq_controllers/          # gripper controller plugin
├── robotiq_description/          # URDF/xacro 및 controller 설정
├── robotiq_driver/               # hardware interface 및 serial driver
├── robotiq_hardware_tests/       # hardware integration tests
├── serial/                       # serial 통신 라이브러리
└── README.md
```

## `src/ur5e_robotiq_config/`

UR5e + Robotiq 조합의 MoveIt 설정 패키지입니다.

```text
src/ur5e_robotiq_config/
├── config/                       # kinematics, controllers, planning pipeline 설정
├── launch/                       # MoveIt/RViz/demo launch 파일
├── srdf/                         # semantic robot description
├── rviz/                         # MoveIt RViz 설정
├── warehouse_ros_mongo/          # warehouse DB 설정
├── CMakeLists.txt
└── package.xml
```

## 문서 관리 원칙

- 최상단 `README.md`는 상세 내용을 모두 담기보다 `docs/` 문서로 연결하는 허브 역할을 합니다.
- 긴 아키텍처 설명은 `docs/architecture.md`에 둡니다.
- 프로젝트 목적과 전체 흐름은 `docs/project-overview.md`에 둡니다.
- 디렉토리 및 패키지 설명은 `docs/directory-structure.md`에 둡니다.
