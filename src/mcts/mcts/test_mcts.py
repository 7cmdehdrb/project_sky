import numpy as np
import time
import random
import math
from abc import ABC, abstractmethod
from typing import List, Any


# ==========================================
# 1. MCTS 코어 인터페이스 및 엔진 (수정 최소화)
# ==========================================
class GameState(ABC):
    @abstractmethod
    def get_possible_actions(self) -> List[Any]:
        pass

    @abstractmethod
    def take_action(self, action: Any) -> "GameState":
        pass

    @abstractmethod
    def is_terminal(self) -> bool:
        pass

    @abstractmethod
    def get_reward(self) -> float:
        pass


class MCTSNode:
    def __init__(self, state: GameState, parent=None, action=None):
        self.state = state
        self.parent = parent
        self.action = action
        self.children = {}
        self.visits = 0
        self.value = 0.0

    @property
    def is_fully_expanded(self) -> bool:
        return len(self.children) == len(self.state.get_possible_actions())

    def best_child(self, c_param: float = 1.414) -> "MCTSNode":
        best_score = -float("inf")
        best_node = None
        for child in self.children.values():
            if child.visits == 0:
                score = float("inf")
            else:
                exploitation = child.value / child.visits
                exploration = c_param * math.sqrt(math.log(self.visits) / child.visits)
                score = exploitation + exploration

            if score > best_score:
                best_score = score
                best_node = child
        return best_node


class MCTS:
    def __init__(self, time_limit: float = 1.0, exploration_constant: float = 20.0):
        # 리워드 스케일이 100단위이므로 exploration_constant(C) 값도 키워줍니다.
        self.time_limit = time_limit
        self.exploration_constant = exploration_constant

    def search(self, initial_state: GameState) -> Any:
        root = MCTSNode(state=initial_state)
        start_time = time.time()

        while time.time() - start_time < self.time_limit:
            node = self._select_and_expand(root)
            reward = self._simulate(node.state)
            self._backpropagate(node, reward)

        # 탐색 종료 후 가장 많이 방문하고 가치가 높은 노드 선택
        return root.best_child(c_param=0.0).action

    def _select_and_expand(self, node: MCTSNode) -> MCTSNode:
        while not node.state.is_terminal():
            if not node.is_fully_expanded:
                return self._expand(node)
            else:
                node = node.best_child(self.exploration_constant)
        return node

    def _expand(self, node: MCTSNode) -> MCTSNode:
        actions = node.state.get_possible_actions()
        for action in actions:
            if action not in node.children:
                new_state = node.state.take_action(action)
                new_node = MCTSNode(state=new_state, parent=node, action=action)
                node.children[action] = new_node
                return new_node
        raise Exception("확장 불가 상태")

    def _simulate(self, state: GameState) -> float:
        current_state = state
        # 무작위 Rollout 수행
        while not current_state.is_terminal():
            possible_actions = current_state.get_possible_actions()
            action = random.choice(possible_actions)
            current_state = current_state.take_action(action)
        return current_state.get_reward()

    def _backpropagate(self, node: MCTSNode, reward: float):
        while node is not None:
            node.visits += 1
            node.value += reward
            node = node.parent


# ==========================================
# 2. 그리드 기반 창고(물류) 환경 구현
# ==========================================
class GridWorldState(GameState):
    def __init__(self, grid: np.ndarray, steps: int = 0):
        self.grid = np.copy(grid)  # 상태 변화를 위해 깊은 복사
        self.steps = steps
        self.rows, self.cols = self.grid.shape

    def get_possible_actions(self) -> List[Any]:
        actions = []
        # 제약 조건: "앞(Row 0 방향)에 물체가 없어야 뺄 수 있다"
        # 각 열(col)별로 0(빈칸)이 아닌 가장 처음 만나는 요소만 추출 가능
        for c in range(self.cols):
            for r in range(self.rows):
                if self.grid[r, c] != 0:
                    actions.append((r, c))  # (행, 열) 좌표를 액션으로 사용
                    break
        return actions

    def take_action(self, action: tuple) -> "GameState":
        r, c = action
        new_grid = np.copy(self.grid)
        # 물체를 뽑았으므로 해당 자리는 0(빈칸)이 됨
        new_grid[r, c] = 0

        # 스텝을 1 증가시킨 새로운 상태 반환 (패널티 누적 역할)
        return GridWorldState(new_grid, self.steps + 1)

    def is_terminal(self) -> bool:
        # 타겟(2)이 그리드 상에 더 이상 없으면 목표 달성으로 종료
        if not np.any(self.grid == 2):
            return True
        # 뽑을 수 있는 물체가 없는데 타겟도 못 뽑았다면 갇힌 상태로 종료
        if len(self.get_possible_actions()) == 0:
            return True
        return False

    def get_reward(self) -> float:
        if not np.any(self.grid == 2):
            # 성공 리워드: 100점 - (물체를 뽑기 위해 소모한 스텝 수)
            # 최단 시간(최소 횟수)으로 뽑을수록 높은 리워드를 받음
            return 100.0 - self.steps
        else:
            # 실패 리워드: 강한 패널티
            return -100.0

    def print_grid(self):
        """
        현재 상태를 보기 좋게 출력하는 헬퍼 함수
        0: 빈칸, 1: 일반 물체, 2: 타겟(목표), -1: 알 수 없음
        """
        symbols = {0: " 🔲 ", 1: " 📦 ", 2: " 🎯 ", -1: " ❓ "}
        for r in range(self.rows - 1, -1, -1):
            row_str = "".join([symbols[val] for val in self.grid[r]])
            print(f"Row {r} | {row_str}")
        print()


# ==========================================
# 3. 테스트 및 실행 코드
# ==========================================
if __name__ == "__main__":
    # 4x5 보드 생성 (Row 0이 가장 앞쪽, Row 3이 가장 안쪽)
    # 0: 비어있음, 1: 일반 물체, 2: 타겟(목표), -1: 알 수 없음
    initial_grid = np.array(
        [
            [1, 0, 1, -1, 1],  # 가장 앞쪽 (입구)
            [1, 1, -1, 1, 1],
            [1, 1, -1, 1, 1],  # 타겟은 (2, 2) 위치에 존재
            [1, 1, 1, 1, 1],  # 가장 깊숙한 곳
        ]
    )

    print("🚀 창고 물체 추출 시스템 시작")
    print("목표: 가장 적은 횟수로 🎯(타겟)을 추출하라!")
    print("조건: 입구(Row 0) 방향에 막힌 물체가 없어야 추출 가능\n")

    current_state = GridWorldState(initial_grid)
    current_state.print_grid()

    # MCTS 엔진 설정 (매 턴마다 1초씩 고민)
    mcts_engine = MCTS(time_limit=1.0)

    while not current_state.is_terminal():
        print(f"--- [Step {current_state.steps + 1}] MCTS 탐색 중... ---")
        best_action = mcts_engine.search(current_state)

        r, c = best_action
        target_type = current_state.grid[r, c]

        if target_type == 2:
            print(f"✅ MCTS 결정: 위치 {best_action}의 🎯 타겟 추출 완료!")
        elif target_type == -1:
            print(f"⚠️ MCTS 결정: 위치 {best_action}의 ❓ 미확인 물체 치우기")
        else:
            print(f"📦 MCTS 결정: 위치 {best_action}의 일반 물체 치우기")

        current_state = current_state.take_action(best_action)
        current_state.print_grid()

    print(f"🏁 시뮬레이션 종료. 총 {current_state.steps}번의 동작을 수행했습니다.")
