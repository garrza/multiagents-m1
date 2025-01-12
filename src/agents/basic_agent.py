from typing import Dict, List, Tuple
import random
from .base import Agent
from ..environment import Environment


class BasicAgent(Agent):
    def __init__(self, position: Tuple[int, int]):
        super().__init__(position)
        self.directions = [
            (-1, -1),
            (-1, 0),
            (-1, 1),
            (0, -1),
            (0, 1),
            (1, -1),
            (1, 0),
            (1, 1),
        ]

    def see(self, environment: Environment) -> Dict:
        """Perceive current cell state and surrounding cells"""
        return {
            "current_position": self.position,
            "is_dirty": environment.is_dirty(self.position),
            "valid_moves": self._get_valid_moves(environment),
        }

    def next(self, perception: Dict) -> str:
        """Decide next action based on perception"""
        if perception["is_dirty"]:
            return "CLEAN"
        return "MOVE"

    def action(self, action_type: str, environment: Environment) -> None:
        """Execute the decided action"""
        if action_type == "CLEAN":
            environment.clean_cell(self.position)
            self.cells_cleaned += 1
        elif action_type == "MOVE":
            valid_moves = self._get_valid_moves(environment)
            if valid_moves:
                new_position = random.choice(valid_moves)
                self.position = new_position
                self.moves_count += 1

    def _get_valid_moves(self, environment: Environment) -> List[Tuple[int, int]]:
        valid_moves = []
        for dx, dy in self.directions:
            new_pos = (self.position[0] + dx, self.position[1] + dy)
            if environment.is_valid_position(new_pos):
                valid_moves.append(new_pos)
        return valid_moves
