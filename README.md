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