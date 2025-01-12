from typing import Dict, Tuple
import random
from .basic_agent import BasicAgent
from ..environment import Environment


class SmartAgent(BasicAgent):
    def __init__(self, position: Tuple[int, int]):
        super().__init__(position)
        # Track visited cells and their states
        self.visited_cells = set()
        # Track last few positions to avoid getting stuck
        self.position_history = []
        self.history_limit = 5

    def see(self, environment: Environment) -> Dict:
        """Enhanced perception that includes surrounding dirty cells"""
        # Get basic perception
        perception = super().see(environment)

        # Add memory of visited cells
        perception["visited_cells"] = self.visited_cells

        # Check surrounding cells for dirt
        surrounding_dirt = []
        for dx, dy in self.directions:
            new_pos = (self.position[0] + dx, self.position[1] + dy)
            if environment.is_valid_position(new_pos) and environment.is_dirty(new_pos):
                surrounding_dirt.append(new_pos)

        perception["surrounding_dirt"] = surrounding_dirt
        return perception

    def next(self, perception: Dict) -> str:
        """Smarter decision making based on enhanced perception"""
        # If current cell is dirty, clean it
        if perception["is_dirty"]:
            return "CLEAN"

        # Add current position to visited cells
        self.visited_cells.add(self.position)

        # Update position history
        self.position_history.append(self.position)
        if len(self.position_history) > self.history_limit:
            self.position_history.pop(0)

        # If there are dirty cells nearby, move to one of them
        if perception["surrounding_dirt"]:
            return "MOVE_TO_DIRT"

        # If we're stuck in a loop, try to break out
        if self._is_stuck():
            return "EXPLORE"

        # If we have unvisited valid moves, prefer those
        unvisited_moves = [
            pos for pos in perception["valid_moves"] if pos not in self.visited_cells
        ]
        if unvisited_moves:
            return "MOVE_TO_UNVISITED"

        # Default to standard movement
        return "MOVE"

    def action(self, action_type: str, environment: Environment) -> None:
        """Execute actions with smarter movement strategies"""
        if action_type == "CLEAN":
            environment.clean_cell(self.position)
            self.cells_cleaned += 1

        elif action_type == "MOVE_TO_DIRT":
            perception = self.see(environment)
            # Move to the nearest dirty cell
            new_position = min(
                perception["surrounding_dirt"],
                key=lambda pos: self._manhattan_distance(self.position, pos),
            )
            self.position = new_position
            self.moves_count += 1

        elif action_type == "EXPLORE":
            # Move to the furthest valid position from current location
            valid_moves = self._get_valid_moves(environment)
            if valid_moves:
                new_position = max(
                    valid_moves,
                    key=lambda pos: self._manhattan_distance(self.position, pos),
                )
                self.position = new_position
                self.moves_count += 1

        elif action_type == "MOVE_TO_UNVISITED":
            valid_moves = self._get_valid_moves(environment)
            unvisited_moves = [
                pos for pos in valid_moves if pos not in self.visited_cells
            ]
            if unvisited_moves:
                self.position = random.choice(unvisited_moves)
                self.moves_count += 1

        else:  # Standard MOVE
            super().action("MOVE", environment)

    def _manhattan_distance(self, pos1: Tuple[int, int], pos2: Tuple[int, int]) -> int:
        """Calculate Manhattan distance between two positions"""
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    def _is_stuck(self) -> bool:
        """Check if agent is stuck in a loop"""
        if len(self.position_history) < self.history_limit:
            return False
        # Check if we're revisiting the same positions
        return len(set(self.position_history)) < len(self.position_history) / 2
