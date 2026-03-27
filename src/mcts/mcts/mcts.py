from abc import ABC, abstractmethod
from typing import List, Any

import math

import time
import random


class GameState(ABC):
    """
    MCTS가 탐색할 게임 상태의 필수 규격입니다.
    반드시 이 클래스를 상속받아 모든 메서드를 구현해야 합니다.
    """

    @abstractmethod
    def get_possible_actions(self) -> List[Any]:
        """현재 상태에서 가능한 모든 행동(Action) 리스트 반환"""
        pass

    @abstractmethod
    def take_action(self, action: Any) -> "GameState":
        """행동을 적용한 **새로운** GameState 객체 반환 (Deepcopy 등 활용)"""
        pass

    @abstractmethod
    def is_terminal(self) -> bool:
        """게임이 끝났는지 여부 (True/False)"""
        pass

    @abstractmethod
    def get_reward(self) -> float:
        """종료 상태에서의 보상 (예: 승 1, 무 0, 패 -1)"""
        pass


class MCTSNode:
    def __init__(self, state: GameState, parent=None, action=None):
        self.state = state
        self.parent = parent
        self.action = action  # 부모 노드에서 이 노드로 오기 위해 취한 행동
        self.children = {}  # {action: MCTSNode}
        self.visits = 0  # 방문 횟수 (N)
        self.value = 0.0  # 누적 보상 (W)

    @property
    def is_fully_expanded(self) -> bool:
        """가능한 모든 행동이 자식 노드로 확장되었는지 확인"""
        return len(self.children) == len(self.state.get_possible_actions())

    def best_child(self, c_param: float = 1.414) -> "MCTSNode":
        """UCB1 알고리즘을 사용하여 최적의 자식 노드 선택"""
        best_score = -float("inf")
        best_node = None

        for child in self.children.values():
            if child.visits == 0:
                score = float("inf")  # 한 번도 방문 안 한 곳은 무조건 탐색
            else:
                # UCB1 공식: (승률) + C * sqrt(ln(부모 방문수) / 내 방문수)
                exploitation = child.value / child.visits
                exploration = c_param * math.sqrt(math.log(self.visits) / child.visits)
                score = exploitation + exploration

            if score > best_score:
                best_score = score
                best_node = child

        return best_node


class MCTS:
    def __init__(self, time_limit: float = 1.0, exploration_constant: float = 1.414):
        """
        :param time_limit: 탐색에 사용할 시간 (초 단위)
        :param exploration_constant: UCB1 탐색 상수 (일반적으로 루트 2 사용)
        """
        self.time_limit = time_limit
        self.exploration_constant = exploration_constant

    def search(self, initial_state: GameState) -> Any:
        # 타입 검사: 강제된 규격을 따르는지 한 번 더 확인
        if not isinstance(initial_state, GameState):
            raise TypeError("초기 상태 객체는 반드시 GameState를 상속받아야 합니다.")

        root = MCTSNode(state=initial_state)
        start_time = time.time()

        # 정해진 시간 동안 계속해서 시뮬레이션 반복
        while time.time() - start_time < self.time_limit:
            # 1. Selection & 2. Expansion
            node = self._select_and_expand(root)
            # 3. Simulation (Rollout)
            reward = self._simulate(node.state)
            # 4. Backpropagation
            self._backpropagate(node, reward)

        # 탐색이 끝난 후 최고 효율의 행동 반환 (이때는 탐색 상수 C=0으로 활용만 함)
        return root.best_child(c_param=0.0).action

    def _select_and_expand(self, node: MCTSNode) -> MCTSNode:
        """단말 노드에 도달할 때까지 UCB1으로 선택하고, 가능하면 새로운 노드 확장"""
        while not node.state.is_terminal():
            if not node.is_fully_expanded:
                return self._expand(node)
            else:
                node = node.best_child(self.exploration_constant)
        return node

    def _expand(self, node: MCTSNode) -> MCTSNode:
        """현재 노드에서 아직 시도해보지 않은 행동 하나를 골라 자식 노드 생성"""
        actions = node.state.get_possible_actions()
        for action in actions:
            if action not in node.children:
                new_state = node.state.take_action(action)
                new_node = MCTSNode(state=new_state, parent=node, action=action)
                node.children[action] = new_node
                return new_node
        raise Exception("확장할 수 없는 상태입니다.")

    def _simulate(self, state: GameState) -> float:
        """무작위로 끝까지 게임을 진행하여 결과를 반환 (Rollout)"""
        current_state = state
        while not current_state.is_terminal():
            possible_actions = current_state.get_possible_actions()
            action = random.choice(possible_actions)
            current_state = current_state.take_action(action)
        return current_state.get_reward()

    def _backpropagate(self, node: MCTSNode, reward: float):
        """결과 보상을 트리를 따라 루트까지 거슬러 올라가며 업데이트"""
        while node is not None:
            node.visits += 1
            node.value += reward
            node = node.parent
