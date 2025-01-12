from abc import ABC, abstractmethod
from typing import Dict, Tuple
from ..environment import Environment


class Agent(ABC):
    def __init__(self, position: Tuple[int, int]):
        self.position = position
        self.moves_count = 0
        self.cells_cleaned = 0

    @abstractmethod
    def see(self, environment: Environment) -> Dict:
        pass

    @abstractmethod
    def next(self, perception: Dict) -> str:
        pass

    @abstractmethod
    def action(self, action_type: str, environment: Environment) -> None:
        pass
