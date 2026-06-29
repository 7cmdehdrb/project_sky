# 프로젝트 아키텍처 및 노드 구동 원리

본 프로젝트는 ROS 2 기반으로 영상 인식(YOLO, FCN), 깊이 정보 처리(Grid Distance), 그리고 강화학습(DRL) 추론 로직을 연동하여 로봇 또는 시스템을 제어합니다. 각 노드는 알고리즘 처리 및 외부 모듈 연동을 담당하는 **Manager 클래스(YoloManager, ImageManager, FCNManager, RLPolicyManager 등)**를 별도로 선언하고 초기화하여 사용하는 형태로 설계되어 있습니다. 이를 통해 ROS 2 통신 로직과 도메인 로직을 분리하여 유지보수성을 높였습니다.

이 문서에서는 제공된 5개의 핵심 노드의 역할과 구조, Publisher/Subscriber 패턴 및 Client/Server(Req-Res) 통신 구조에 대해 중점적으로 설명합니다.

---

## 1. Yolo Node (`yolo_node.py` - `RealTimeSegmentationNode`)

YOLO 모델을 통해 카메라 이미지로부터 객체를 탐지하고 세그먼테이션(Segmentation) 바운딩 박스를 추출하는 노드입니다.
- **사용되는 Manager**: `YoloManager` (추론 담당), `ImageManager` (이미지 송수신 및 전처리 담당), `ObjectManager` (클래스 이름 정규화 및 색상 매핑 담당)
- **통신 구조 (Pub/Sub)**:
  - **[Sub]** `/camera/camera1/color/image_raw` (`sensor_msgs/Image`): 카메라 원본 이미지 수신
  - **[Pub]** `/real_time_segmentation_node/segmented_image` (`sensor_msgs/Image`): 바운딩 박스가 오버레이된 시각화 이미지 발행
  - **[Pub]** `/real_time_segmentation_node/segmented_bbox` (`custom_msgs/BoundingBoxMultiArray`): 탐지된 객체의 클래스 이름, 신뢰도(Confidence), 바운딩 박스 좌표, 마스크 데이터 등을 배열 형태로 발행

## 2. Closest Object Node (`closest_object_node.py` - `ClosestObjectClassifierNode`)

YOLO에서 탐지된 객체의 마스크 데이터와 Depth 이미지를 매칭하여, 4개의 구역(Column)별로 가장 가까운 객체가 무엇인지 분류해내는 노드입니다.
- **사용되는 Manager**: `ImageManager`, `ObjectManager`
- **통신 구조 (Pub/Sub)**:
  - **[Sub]** `/real_time_segmentation_node/segmented_bbox` (`custom_msgs/BoundingBoxMultiArray`): YOLO 노드에서 탐지된 객체 정보 수신
  - **[Sub]** `/camera/camera1/depth/image_rect_raw` (`sensor_msgs/Image`): 16비트 Depth 이미지 수신
  - **[Pub]** `/closest_object_classifier/closest_object_ids` (`std_msgs/Int32MultiArray`): 각 구역(Column)별로 가장 가까운 객체의 ID 목록(배열) 발행 (-1일 경우 존재하지 않음을 의미)
  - **[Pub]** `/closest_object_classifier/closest_object_overlay` (`sensor_msgs/Image`): Depth 이미지 위에 감지된 객체 정보(마스크, 이름)를 입힌 시각화 이미지 발행

## 3. Grid Node (`grid_node.py` - `GridDistancePublisherNode`)

깊이 카메라의 3D 데이터(PointCloud2)를 그리드 환경으로 매핑하고, 전방에 위치한 장애물이나 특정 구역의 거리를 계산하는 노드입니다.
- **사용되는 Manager**: `GridManager` (그리드 상태 추적 및 Marker 연산)
- **통신 구조 (Pub/Sub)**:
  - **[Sub]** `/camera/camera1/depth/color/points` (`sensor_msgs/PointCloud2`): 깊이 카메라로부터 3D 포인트 클라우드 수신
  - **[Pub]** `/grid_markers` (`visualization_msgs/MarkerArray`): Rviz2 시각화용 3D 마커 발행
  - **[Pub]** `/front_object_distance` (`std_msgs/Float32MultiArray`): 각 컬럼(Column) 별 전방 객체까지의 거리를 수치화하여 발행 (강화학습 상태 공간 등에 활용)

## 4. FCN Node (`fcn_node.py` - `FCNServiceNode` / Node B)

목표 객체(Target)의 위치 분포를 추론하기 위해 FCN(Fully Convolutional Network)을 구동하는 **서비스 서버** 노드입니다. 요청이 들어왔을 때만 추론을 수행하여 반환합니다.
- **사용되는 Manager**: `FCNManager` (FCN 모델 추론 및 후처리), `ImageManager` (상시 이미지 구독 및 시각화용 퍼블리시)
- **통신 구조 (Pub/Sub 및 Req/Res)**:
  - **[Sub]** `/camera/camera1/color/image_raw` (`sensor_msgs/Image`): 항상 최신 이미지를 유지하기 위해 구독
  - **[Pub]** `/fcn_service_node/pdm_visualization`, `/fcn_service_node/target_map_visualization`: 1D PDM 그래프 및 맵 결과 시각화 이미지 발행 (타이머 기반)
  - **[Service Server]** `get_fcn_prediction` (`custom_msgs/srv/GetFCNResult`): 
    - **(Req)**: `target_class_idx`, `weight` 가중치
    - **(Res)**: 입력된 타겟 및 가중치를 기반으로 추론된 구역별 1D 분포 점수 데이터 (`response.data`)

## 5. DRL Node (`drl_node.py` - `PolicyServiceNode` / Node A)

현재 환경의 State(전방 객체 거리, 가장 가까운 객체 ID 등)와 FCN 결과 메세지를 통합하여, 강화학습(DRL) 모델 기반의 최적 Action을 추론한 후 반환하는 **서비스 서버** 노드이자 **클라이언트**입니다.
- **사용되는 Manager**: `RLPolicyManager` (상태값 종합 및 ONNX Policy 모델 기반 추론)
- **통신 구조 (Pub/Sub 및 Req/Res)**:
  - **[Sub]** `/front_object_distance` (`std_msgs/Float32MultiArray`): `grid_node`에서 발행한 구역별 거리 수신하여 상태 저장
  - **[Sub]** `/closest_object_classifier/closest_object_ids` (`std_msgs/Int32MultiArray`): `closest_object_node`에서 구역별 가장 가까운 객체의 ID 수신
  - **[Service Client]** `get_fcn_prediction` (`custom_msgs/srv/GetFCNResult`): Main 노드로부터 정책 요청 시, 로드된 최신 정보를 바탕으로 `FCN Node`에 추론 요청 (동기 시점 결합)
  - **[Service Server]** `get_policy_action` (`custom_msgs/srv/GetPolicyAction`):
    - **(Req)**: `target_id` (메인 제어기에서 전달된 목표 객체의 ID)
    - **(Res)**: 내부적으로 FCN Node의 응답과 현재 환경의 로컬 State(Sub로 받아온 거리 및 ID 데이터)를 `RLPolicyManager`에 주입하여 도출한 `action_type`과 `target_column`을 반환

---

## 💡 종합 통신 흐름 요약

이 시스템은 퍼블리셔-서브스크라이버 기반의 **비동기 상태 업데이트**와 서비스/클라이언트 기반의 **동기 추론 요청** 메커니즘이 혼합되어 있습니다.

1. **상태(Status) 갱신 파이프라인 (Pub/Sub 지속 동작)**
   - `yolo_node` & `closest_object_node` -> 객체 정보 및 마스크, 구역 내 가장 가까운 객체 판별 (`/closest_object_classifier/closest_object_ids`)
   - `grid_node` -> 전방 포인트 클라우드 분석하여 구역 별 객체 거리 측정 (`/front_object_distance`)
   - 앞선 두가지 지속적인 환경 State는 `drl_node (PolicyServiceNode)` 내부 Manager에 갱신됩니다.

2. **Req-Res (Service) 추론 연쇄 파이프라인 (이벤트 기반)**
   - **(External Main -> DRL Node 시작)**: 시스템 제어기(Main)가 `drl_node`의 `get_policy_action` 서비스를 호출하면서 `target_id`를 요청.
   - **(DRL Node -> FCN Node)**: `drl_node`는 즉시 `fcn_node`의 `get_fcn_prediction` 서비스를 호출하여 해당 `target_id`의 공간적 분포 현황 점수를 요청.
   - **(FCN Node -> DRL Node)**: `fcn_node`는 가지고 있는 최신 이미지를 바탕으로 추론을 진행해 FCN 예측값을 `drl_node`로 반환(Response).
   - **(DRL Node 반환)**: `drl_node`는 그동안 모아두었던 환경 State 정보와 방금 FCN으로부터 받은 분포 점수를 결합해 정책 모델(RLPolicy) 추론을 실행하고, 메인 제어기 측으로 최종 `action_type` 및 `target_column`을 반환합니다.


---


# DRL-Occluded-Object-Search

## 개요

이 프로젝트는 **ROS 2 기반의 가려진(occluded) 객체 탐색 및 로봇 조작 파이프라인**을 목표로 합니다. 현재 코드베이스를 기준으로 보면 시스템은 크게 다음 네 층으로 나뉩니다.

1. **인지(Perception)**: RGB / Depth / PointCloud를 받아 객체 탐지와 환경 상태를 추출
2. **추론(Inference)**: FCN과 DRL 정책을 통해 목표 물체의 가능 위치와 행동(action) 계산
3. **배치(Placement)**: 집은 물체를 어디에 내려놓을지 Drop Grid 서비스로 결정
4. **제어(Control, 진행 중)**: `robot_control` 패키지가 최상위에서 로봇 팔과 그리퍼를 실제로 움직이는 구조

현재 구현 완료로 볼 수 있는 핵심 파일은 아래 6개입니다.

- `integration_image_node.py`
- `yolo_node.py`
- `closest_object_node.py`
- `grid_node.py`
- `fcn_node.py`
- `drop_grid_node.py`

추가로, 실제 상위 제어 계층으로 발전할 예정인 `src/robot_control/robot_control` 하위 3개 파일도 현재 클래스 구조만으로 역할을 충분히 파악할 수 있으므로, 본 README 하단에서 별도 섹션으로 정리합니다.

---

## 전체 ROS2 연결 구조

아래 흐름으로 이해하면 현재 시스템 구성이 가장 자연스럽습니다.

```text
RGB Image --------------------> yolo_node -------------------> /real_time_segmentation_node/segmented_bbox
   |                               |                                          |
   |                               └--> /real_time_segmentation_node/segmented_image
   |
   └----------------------------> fcn_node --(service: get_fcn_prediction)--> DRL / 상위 제어기

Depth Image ------------------> closest_object_node ---------> /closest_object_classifier/closest_object_ids
   |                               |
   |                               └--> /closest_object_classifier/closest_object_overlay
   |
PointCloud -------------------> grid_node -------------------> /front_object_distance
                                   |
                                   └--> /grid_markers

RViz/Debug Images -----------> integration_image_node ------> /integration_image_node/integrated_image

Drop decision request -------> drop_grid_node --------------> service: request_drop_cell

(진행 중)
robot_control/main.py -------> controller.py / action_sequence.py 를 이용해
                               최종 행동을 실제 UR5e + gripper 동작으로 연결
```

즉, **YOLO + Depth + PointCloud가 환경 상태를 만든 뒤**, **FCN / DRL / Drop Grid / Robot Control**이 그 상태를 소비하는 계층형 구조입니다.

---

## 구현 완료 노드 상세 설명

## 1. `src/test/integration_image_node.py`

### 역할
`integration_image_node.py`는 여러 노드의 시각화 출력을 한 화면에 모아 보여주는 **통합 디버그/모니터링 노드**입니다. 알고리즘 핵심 로직을 수행한다기보다는, 현재 시스템이 어떤 중간 결과를 만들어내는지를 한 번에 확인할 수 있게 해 줍니다.

### 입력 토픽
- `/camera/camera1/color/image_raw`
- `/closest_object_classifier/closest_object_overlay`
- `/real_time_segmentation_node/segmented_image`
- `/fcn_service_node/pdm_visualization`
- `/fcn_service_node/target_map_visualization`

### 출력 토픽
- `/integration_image_node/integrated_image`

### 내부 동작
- 각 토픽에서 받은 이미지를 `ImageManager`로 디코딩합니다.
- 크기가 다르면 640x480 기준으로 보정합니다.
- 외부 PNG 이미지를 추가로 불러와 마지막 패널에 배치합니다.
- 상단 3장, 하단 3장을 이어 붙여 하나의 2x3 대시보드 이미지를 생성합니다.

### 시스템 내 의미
이 노드는 다른 노드들의 의존 대상은 아니지만, 다음을 동시에 확인하는 데 매우 유용합니다.
- 원본 카메라 입력이 정상인지
- YOLO 분할 결과가 적절한지
- Closest object 분류가 잘 되고 있는지
- FCN의 1D/2D 시각화가 합리적인지

즉, **전체 인지 파이프라인의 관측용 허브**에 해당합니다.

---

## 2. `src/object_tracker/object_tracker/yolo_node.py`

### 역할
`yolo_node.py`의 `RealTimeSegmentationNode`는 RGB 카메라 이미지를 입력으로 받아 **객체 탐지 + 마스크 기반 분할 결과**를 생성하는 노드입니다. 사실상 후속 인지 모듈들이 참조하는 가장 앞단의 객체 인식기입니다.

### 핵심 클래스 구성
- `YoloManager`
  - YOLO 모델 로드
  - ROS Image / NumPy / PIL Image를 공통 포맷으로 전처리
  - 추론 결과 반환
- `RealTimeSegmentationNode`
  - 카메라 구독
  - YOLO 추론 실행
  - Bounding Box / Mask 결과를 ROS 메시지로 발행
  - 시각화 이미지를 함께 발행

### 입력 토픽
- `/camera/camera1/color/image_raw` (`sensor_msgs/Image`)

### 출력 토픽
- `/real_time_segmentation_node/segmented_image`
- `/real_time_segmentation_node/segmented_bbox`

### 실제 수행 기능
1. RGB 이미지를 crop 합니다.
2. YOLO로 객체 검출과 segmentation mask를 계산합니다.
3. 신뢰도 임계값(`conf_threshold`) 이하 결과는 제거합니다.
4. `ObjectManager`를 이용해 클래스 이름을 정규화합니다.
5. 마스크를 원본 이미지 크기로 복원한 뒤 `BoundingBoxMultiArray`로 발행합니다.
6. 바운딩 박스와 클래스명을 그린 시각화 이미지를 함께 발행합니다.

### 시스템 내 연결
- **직접 연결되는 다음 노드**: `closest_object_node.py`
- `closest_object_node`는 여기서 발행한 `/segmented_bbox`를 사용하여, “보이는 객체들 중 각 열(column)에서 가장 가까운 물체가 무엇인지”를 계산합니다.
- `integration_image_node`는 `/segmented_image`를 받아 전체 디버그 화면을 구성합니다.

즉, `yolo_node.py`는 **후속 depth 기반 판단의 출발점**입니다.

---

## 3. `src/object_tracker/object_tracker/closest_object_node.py`

### 역할
`closest_object_node.py`의 `ClosestObjectClassifierNode`는 YOLO가 준 마스크와 depth 이미지를 결합하여, **화면을 여러 열로 나누었을 때 각 열에서 가장 가까운 물체 ID**를 계산합니다.

이 노드는 “무엇이 보였는가?”를 넘어, **“무엇이 가장 앞에 있는가?”**를 추출합니다.

### 입력 토픽
- `/real_time_segmentation_node/segmented_bbox`
- `/camera/camera1/depth/image_rect_raw`

### 출력 토픽
- `/closest_object_classifier/closest_object_ids`
- `/closest_object_classifier/closest_object_overlay`

### 실제 수행 기능
1. YOLO가 보낸 `BoundingBoxMultiArray`에서 클래스명과 마스크를 읽습니다.
2. Depth 이미지를 수신해 crop 및 보정합니다.
3. 마스크 내부 픽셀만 depth 값으로 추출합니다.
4. outlier를 제거한 뒤 평균 거리를 계산합니다.
5. 화면을 4개 column으로 나눠 각 객체가 어느 column에 속하는지 판정합니다.
6. 각 column마다 가장 가까운 물체의 object ID를 선택해 `Int32MultiArray`로 발행합니다.
7. Depth 위에 마스크와 클래스명을 오버레이한 시각화 이미지를 발행합니다.

### 시스템 내 연결
- **입력 측면**에서는 `yolo_node.py`에 의존합니다.
- **출력 측면**에서는 정책 결정 계층이 이 정보를 사용합니다.
  - `drl_node.py`는 `/closest_object_classifier/closest_object_ids`를 받아 정책 상태(state)의 일부로 사용합니다.
- `integration_image_node`는 `/closest_object_classifier/closest_object_overlay`를 받아 모니터링 화면을 만듭니다.

즉, 이 노드는 **2D 인식 결과를 실제 행동 결정을 위한 전방 객체 상태로 바꿔 주는 브리지**입니다.

---

## 4. `src/fcn_network/fcn_network/grid_node.py`

### 역할
`grid_node.py`의 `GridDistancePublisherNode`는 PointCloud를 그리드 형태로 해석하여, **각 column 전방에서 가장 먼저 만나는 점유 구역까지의 거리**를 계산합니다.

이 노드는 물체의 종류를 맞추는 것이 아니라, **공간이 얼마나 막혀 있는지**를 정량화하는 역할을 합니다.

### 입력 토픽
- `/camera/camera1/depth/color/points` (`sensor_msgs/PointCloud2`)

### 출력 토픽
- `/grid_markers`
- `/front_object_distance`
- `/processed_pointcloud` (디버깅용)

### 실제 수행 기능
1. 최신 PointCloud를 버퍼링합니다.
2. `TransformManager`로 좌표계를 `camera1_link` 기준으로 맞춥니다.
3. `PointCloudTransformer`로 PointCloud2를 NumPy로 변환합니다.
4. `GridManager`에 occupancy를 업데이트합니다.
5. RViz용 `MarkerArray`를 발행합니다.
6. 각 column에서 가장 앞쪽 row를 찾고 이를 거리 값으로 치환합니다.
7. 최종 거리 배열을 `/front_object_distance`로 발행합니다.

### 시스템 내 연결
- `drl_node.py`는 `/front_object_distance`를 구독하여 현재 전방 혼잡도/장애물 상황을 상태값으로 사용합니다.
- `controller.py`는 `/grid_markers`를 collision object로 변환해 MoveIt planning scene에 반영하도록 설계되어 있습니다.

즉, `grid_node.py`는 **정책 추론과 모션 플래닝 양쪽에서 모두 중요한 공간 인식 노드**입니다.

---

## 5. `src/fcn_network/fcn_network/fcn_node.py`

### 역할
`fcn_node.py`의 `FCNServiceNode`는 최신 RGB 이미지에 대해 FCN(Fully Convolutional Network)을 실행하여, **목표 객체가 어느 column 쪽에 존재할 가능성이 높은지**를 추론하는 서비스 노드입니다.

### 입력
- 상시 구독 토픽: `/camera/camera1/color/image_raw`
- 서비스 요청: `get_fcn_prediction`

### 출력
- 서비스 응답: `weighted_peak_data`
- 시각화 토픽:
  - `/fcn_service_node/pdm_visualization`
  - `/fcn_service_node/target_map_visualization`

### 실제 수행 기능
1. 카메라 RGB를 계속 받아 최신 프레임을 메모리에 유지합니다.
2. 서비스 요청이 오면 그 시점의 최신 이미지를 복사합니다.
3. `FCNManager.predict()`로 2D 결과 맵을 계산합니다.
4. `post_process_results()`로 1D PDM과 column별 가중 결과를 만듭니다.
5. 그 결과를 서비스 응답으로 반환합니다.
6. 별도 타이머에서 1D 그래프와 2D target map 시각화 이미지를 계속 발행합니다.

### 시스템 내 연결
- `drl_node.py`가 `get_fcn_prediction` 서비스 클라이언트입니다.
- `integration_image_node`는 FCN의 두 시각화 토픽을 받아 전체 화면에 포함합니다.

즉, FCN 노드는 **현재 프레임에 대한 목표 객체 분포 추론기**이며, DRL이 행동을 정할 때 즉시 호출되는 **동기식 추론 백엔드**입니다.

---

## 6. `src/fcn_network/fcn_network/drop_grid_node.py`

### 역할
`drop_grid_node.py`의 `DropGridNode`는 집은 물체를 다음에 어디에 놓을지 결정하는 **배치 위치 할당 서비스**입니다.

탐색/집기 이후 단계에서 “다음 drop cell은 어디인가?”를 알려 주는 배치 관리자라고 보면 됩니다.

### 입력
- 서비스 요청: `request_drop_cell`

### 출력
- 서비스 응답: 다음 drop cell의 row / col / center / size
- 토픽: `/drop_grid_node/drop_grid_markers`

### 실제 수행 기능
1. `DropGridManager`를 초기화합니다.
2. Drop 우선순위(`DropPriority`)와 row/column 진행 방향을 설정합니다.
3. 서비스 요청이 오면 `get_next_drop_cell()`로 다음 셀을 계산합니다.
4. 응답에 row/col/frame_id/center_coord/size를 채웁니다.
5. 내부 상태를 `drop()`으로 갱신하여 같은 셀을 다시 주지 않게 합니다.
6. 별도의 루프에서 grid marker를 주기적으로 발행합니다.

### 시스템 내 연결
- 현재 상위 제어 노드에서 직접 연결하는 코드는 아직 보강 중이지만, 구조상 이 노드는 **grasp 이후 place 단계**에서 사용될 서비스입니다.
- 특히 `action_sequence.py`의 `GraspActionSequence._place()`가 아직 TODO 상태이므로, 향후 이 서비스와 결합될 가능성이 매우 높습니다.

즉, `drop_grid_node.py`는 **탐색/집기 이후의 배치 전략을 담당하는 후반부 모듈**입니다.

---

## 구현 완료 노드들의 유기적 연결 관계

## 1. 비동기 상태 갱신 파이프라인
다음 노드들은 센서 데이터를 지속적으로 받아 환경 상태를 계속 갱신합니다.

- `yolo_node.py`
  - RGB 기반 객체 탐지 결과 생성
- `closest_object_node.py`
  - YOLO 마스크 + Depth 결합으로 column별 최전방 객체 ID 계산
- `grid_node.py`
  - PointCloud 기반 column별 전방 거리 계산
- `fcn_node.py`
  - 최신 RGB 프레임 캐시 유지
- `integration_image_node.py`
  - 위 시각화 결과를 종합해 관찰 화면 구성

이 단계는 모두 **Pub/Sub 기반의 비동기 흐름**입니다.

## 2. 동기 추론/의사결정 파이프라인
실제 행동 결정을 위해서는 보통 상위 제어기에서 요청이 들어와야 합니다.

현재 코드상 이 역할은 `drl_node.py`가 맡고 있습니다.

1. `drl_node.py`는 평소에 아래 상태를 구독합니다.
   - `/front_object_distance` from `grid_node.py`
   - `/closest_object_classifier/closest_object_ids` from `closest_object_node.py`
2. 상위 노드가 `get_policy_action` 서비스를 호출합니다.
3. `drl_node.py`는 그 즉시 `fcn_node.py`의 `get_fcn_prediction` 서비스를 호출합니다.
4. FCN 분포 결과 + 거리 정보 + 최전방 객체 ID를 합쳐 `RLPolicyManager`가 action을 산출합니다.
5. 최종적으로 어떤 행동 타입을 수행할지 결정해 상위 제어기에 반환합니다.

즉, 시스템의 핵심은 다음과 같이 요약할 수 있습니다.

- **YOLO**: 무엇이 보이는가?
- **Closest Object**: 그중 무엇이 가장 앞에 있는가?
- **Grid**: 공간이 얼마나 막혀 있는가?
- **FCN**: 목표 객체가 어느 쪽에 있을 가능성이 높은가?
- **DRL**: 지금 어떤 행동을 해야 하는가?
- **Drop Grid**: 집은 뒤 어디에 놓아야 하는가?
- **Integration Image**: 위 상태를 사람이 한 번에 확인

---

## `src/robot_control/robot_control` 하위 3개 파일 설명 (진행 중)

이 섹션의 3개 파일은 아직 완성형은 아니지만, 코드 구조상 **향후 시스템의 최상단 오케스트레이터**가 될 가능성이 매우 높습니다. 즉, 위에서 설명한 인지/추론 결과를 받아 실제 UR5e와 그리퍼를 움직이는 계층입니다.

## 1. `src/robot_control/robot_control/main.py`

### 역할
`main.py`의 `MainControlNode`는 최상위 ROS2 노드로서, 현재는 **UR5e 컨트롤러와 액션 시퀀스 객체들을 생성해 두는 엔트리 포인트** 역할을 합니다.

### 현재 코드상 확인되는 기능
- `UR5eController` 인스턴스 생성
- `GraspActionSequence` 생성
- `SweepLeftActionSequence` 생성
- `SweepRightActionSequence` 생성
- 노드를 스핀하며 상위 제어 루프의 틀 유지

### 의미
지금은 외부 서비스 호출이나 상태 머신이 본격적으로 구현되지는 않았지만, 구조상 다음 역할을 맡게 됩니다.
- DRL이 정한 action type 수신
- 해당 action sequence 선택
- target point를 주입
- 로봇 팔 / 그리퍼 실제 동작 실행

즉, `main.py`는 **전체 시스템의 실행 진입점이자 행동 조정자(orchestrator)** 입니다.

---

## 2. `src/robot_control/robot_control/action_sequence.py`

### 역할
`action_sequence.py`는 로봇 동작을 단일 명령이 아니라 **여러 단계의 절차적 시퀀스(sequence)** 로 정의하기 위한 파일입니다.

### 핵심 클래스
- `AxisDirection`
  - 축 방향 단위 벡터 정의
  - target point를 특정 방향으로 offset 하는 데 사용
- `ActionSequence`
  - 모든 동작 시퀀스의 부모 클래스
  - `target_point`, `waypoints`, `state`를 관리
- `GraspActionSequence`
  - HOME → APPROACH → GRASP → PLACE → RELEASE → RETURN_HOME 흐름 정의
- `SweepLeftActionSequence`
  - 좌측 sweeping 동작의 뼈대 정의
- `SweepRightActionSequence`
  - 우측 sweeping 동작의 뼈대 정의

### 현재 코드상 확인되는 기능
- Grasp 시퀀스는 비교적 구체적입니다.
  - 홈 이동
  - 타깃 전방 접근
  - 그리퍼 닫기
  - 배치 단계(현재 TODO)
  - 그리퍼 열기
  - 홈 복귀
- Sweep 계열은 아직 골격만 마련된 상태입니다.

### 의미
이 파일은 정책 결과를 실제 로봇 모션으로 번역하는 **행동 문법(action grammar)** 역할을 합니다. 향후 DRL이 `grasp`, `sweep_left`, `sweep_right` 같은 이산 행동을 반환하면, 그 결과를 받아 각 시퀀스를 순차 실행하는 구조로 자연스럽게 확장할 수 있습니다.

---

## 3. `src/robot_control/robot_control/controller.py`

### 역할
`controller.py`는 실제 MoveIt2 서비스, joint state, collision object, gripper action을 감싸는 **하드웨어/모션 제어 계층**입니다.

### 핵심 클래스
- `UR5eController`
  - 관절 상태 구독
  - grid marker를 collision object로 변환
  - FK / IK / Cartesian / Kinematic planning / trajectory execution 수행
- `RobotiqController`
  - 그리퍼 제어 전담

### 현재 코드상 확인되는 기능
- 홈/세이프티/웨이팅 joint preset 보유
- FK 기반으로 home/safety/waiting pose 계산 가능
- `/joint_states`를 통해 현재 로봇 상태 반영
- `/grid_markers`를 받아 planning scene의 collision object로 변환
- planning scene을 갱신한 뒤 Cartesian path를 계획/실행하는 구조 보유
- 실패 시 재시도하는 헬퍼 로직 포함

### 의미
이 파일이 중요한 이유는, 앞단 perception이 만든 grid 정보가 **실제 MoveIt collision avoidance**로 연결되는 접점이기 때문입니다. 즉, `grid_node.py`가 만든 환경 점유 정보가 `controller.py`를 통해 실제 로봇 경로 계획 안전성에 반영될 수 있습니다.

---

## 현재 기준으로 보는 상위 통합 구조

진행 중인 `robot_control` 계층까지 포함하면, 시스템의 최종 그림은 아래처럼 정리할 수 있습니다.

```text
[센서 계층]
RGB / Depth / PointCloud

    ↓

[인지 계층]
yolo_node
closest_object_node
grid_node
fcn_node
integration_image_node(모니터링)

    ↓

[정책 계층]
drl_node

    ↓

[실행 계층 - 진행 중]
robot_control/main.py
  ├─ action_sequence.py
  └─ controller.py

    ↓

[후처리 / 배치]
drop_grid_node
```

다만 현재 코드만 기준으로 보면, **`drl_node.py`와 `robot_control` 사이의 실제 서비스/토픽 연결은 아직 본격 구현 전**으로 보입니다. 따라서 README를 읽을 때는 아래처럼 이해하면 정확합니다.

- 인지/추론 노드들은 상당 부분 동작 흐름이 구체화되어 있음
- drop grid도 독립 서비스로 구현되어 있음
- 최상위 로봇 제어 계층은 클래스 설계와 제어 인터페이스가 준비되고 있는 단계임

---

## 정리

이 레포지터리의 현재 핵심은 **ROS2 기반 다중 인지 노드가 환경 상태를 만들고, FCN/DRL이 행동 결정을 준비하며, 향후 `robot_control`이 이를 실제 로봇 동작으로 연결하는 구조**라고 볼 수 있습니다.

특히 다음 연결을 이해하면 전체 구조를 빠르게 파악할 수 있습니다.

- `yolo_node.py` → `closest_object_node.py`
- `grid_node.py` → `drl_node.py`
- `closest_object_node.py` → `drl_node.py`
- `fcn_node.py` ↔ `drl_node.py` (service)
- 여러 시각화 토픽 → `integration_image_node.py`
- 향후 `drl_node.py` / 상위 제어기 → `robot_control/main.py`
- 향후 place 단계 → `drop_grid_node.py`

즉, 이 프로젝트는 **탐지 → 상태 추정 → 정책 결정 → 조작 실행 → 배치**로 이어지는 전체 파이프라인을 향해 구성되고 있습니다.