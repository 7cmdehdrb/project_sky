# DRL-Occluded-Object-Search

ROS 2 기반의 **가려진 객체 탐색 및 로봇 조작 파이프라인**입니다. 이 저장소는 카메라 인지, Depth/PointCloud 기반 환경 상태 추정, FCN/DRL 기반 의사결정, Drop Grid, UR5e + Robotiq 제어 구성을 함께 다룹니다.

상세 설명은 최상단 README에 모두 넣지 않고 `docs/` 문서로 분리했습니다. README는 프로젝트를 빠르게 파악하고 필요한 문서로 이동하는 허브 역할을 합니다.

## 문서 바로가기

- [프로젝트 전체 설명](docs/project-overview.md)
- [디렉토리 구조 설명](docs/directory-structure.md)
- [아키텍처 및 노드 구동 원리](docs/architecture.md)
- [카메라 관련 메모](docs/camera.md)
- [명령어 메모](docs/commands.md)
- [이전 README 참고 문서](docs/last-readme.md)

## 한눈에 보는 구성

```text
RGB / Depth / PointCloud
        |
        v
object_tracker  ──>  fcn_network  ──>  robot_control
        |                  |
        |                  ├── FCN prediction
        |                  ├── DRL policy action
        |                  └── Drop grid decision
        |
        └── YOLO segmentation / closest object classification
```

## 주요 패키지

| 패키지 | 역할 |
| --- | --- |
| `base_package` | 이미지, 객체, transform 등 공통 유틸리티 |
| `custom_msgs` | 프로젝트 전용 ROS msg/srv 정의 |
| `object_tracker` | YOLO segmentation, closest object, 통합 이미지 디버깅 |
| `fcn_network` | FCN, DRL, grid, drop grid 추론 노드 |
| `mcts` | MCTS 기반 의사결정 실험 |
| `robot_control` | UR5e + Robotiq 상위 제어 |
| `ros2_robotiq_gripper-humble` | Robotiq gripper ROS 2 Humble 연동 패키지 |
| `ur5e_robotiq_config` | UR5e + Robotiq MoveIt 설정 |

## 빠른 시작 방향

1. ROS 2 workspace로 저장소를 준비합니다.
2. `requirements.txt`와 각 ROS 패키지 의존성을 설치합니다.
3. `colcon build`로 workspace를 빌드합니다.
4. `docs/commands.md`와 각 launch 파일을 참고해 카메라, 인지, 추론, 제어 노드를 실행합니다.

자세한 노드별 topic/service 관계는 [아키텍처 및 노드 구동 원리](docs/architecture.md)를 참고하세요.
