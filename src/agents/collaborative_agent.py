from typing import Dict, List, Tuple, Set
from .base import Agent
from ..environment import Environment


class CollaborativeAgent(Agent):
    # Class variable to share information between agents
    _shared_memory: Dict[int, Dict] = {}
    _agent_counter = 0

    def __init__(self, position: Tuple[int, int]):
        super().__init__(position)
        # Assign unique ID to each agent
        self.agent_id = CollaborativeAgent._agent_counter
        CollaborativeAgent._agent_counter += 1
        
        # Initialize agent's shared memory
        CollaborativeAgent._shared_memory[self.agent_id] = {
            "position": position,
            "cleaned_cells": set(),
            "last_cleaned": None
        }
        
        self.directions = [
            (-1, -1), (-1, 0), (-1, 1),
            (0, -1),           (0, 1),
            (1, -1),  (1, 0),  (1, 1)
        ]

    def see(self, environment: Environment) -> Dict:
        """Perceive current cell state, surrounding cells, and other agents' information"""
        # Update position in shared memory
        CollaborativeAgent._shared_memory[self.agent_id]["position"] = self.position
        
        other_agents_positions = {
            agent_id: info["position"] 
            for agent_id, info in CollaborativeAgent._shared_memory.items()
            if agent_id != self.agent_id
        }
        
        return {
            "current_position": self.position,
            "is_dirty": environment.is_dirty(self.position),
            "valid_moves": self._get_valid_moves(environment),
            "other_agents": other_agents_positions,
            "shared_cleaned_cells": self._get_all_cleaned_cells()
        }

    def next(self, perception: Dict) -> str:
        """Decide next action based on perception and shared information"""
        if perception["is_dirty"]:
            return "CLEAN"
        
        # If cell is clean, move to a position that:
        # 1. Is valid
        # 2. Is not currently occupied by another agent
        # 3. Preferably hasn't been cleaned recently
        valid_moves = perception["valid_moves"]
        if not valid_moves:
            return "MOVE"  # Will stay in place if no valid moves
            
        other_positions = set(perception["other_agents"].values())
        cleaned_cells = perception["shared_cleaned_cells"]
        
        # Filter out positions where other agents are
        valid_moves = [move for move in valid_moves if move not in other_positions]
        if not valid_moves:
            return "MOVE"
            
        # Prefer moves to uncleaned cells
        uncleaned_moves = [move for move in valid_moves if move not in cleaned_cells]
        if uncleaned_moves:
            return "MOVE"
            
        return "MOVE"

    def action(self, action_type: str, environment: Environment) -> None:
        """Execute the decided action and update shared memory"""
        if action_type == "CLEAN":
            environment.clean_cell(self.position)
            self.cells_cleaned += 1
            # Update shared memory
            CollaborativeAgent._shared_memory[self.agent_id]["cleaned_cells"].add(self.position)
            CollaborativeAgent._shared_memory[self.agent_id]["last_cleaned"] = self.position
            
        elif action_type == "MOVE":
            perception = self.see(environment)
            valid_moves = perception["valid_moves"]
            other_positions = set(perception["other_agents"].values())
            cleaned_cells = perception["shared_cleaned_cells"]
            
            if valid_moves:
                # Filter out positions where other agents are
                valid_moves = [move for move in valid_moves if move not in other_positions]
                if valid_moves:
                    # Prefer uncleaned cells
                    uncleaned_moves = [move for move in valid_moves if move not in cleaned_cells]
                    if uncleaned_moves:
                        self.position = self._choose_best_move(uncleaned_moves)
                    else:
                        self.position = self._choose_best_move(valid_moves)
                    self.moves_count += 1
                    # Update position in shared memory
                    CollaborativeAgent._shared_memory[self.agent_id]["position"] = self.position

    def _get_valid_moves(self, environment: Environment) -> List[Tuple[int, int]]:
        """Get all valid moves from current position"""
        valid_moves = []
        for dx, dy in self.directions:
            new_pos = (self.position[0] + dx, self.position[1] + dy)
            if environment.is_valid_position(new_pos):
                valid_moves.append(new_pos)
        return valid_moves

    def _get_all_cleaned_cells(self) -> Set[Tuple[int, int]]:
        """Combine cleaned cells information from all agents"""
        all_cleaned = set()
        for agent_info in CollaborativeAgent._shared_memory.values():
            all_cleaned.update(agent_info["cleaned_cells"])
        return all_cleaned

    def _choose_best_move(self, moves: List[Tuple[int, int]]) -> Tuple[int, int]:
        """Choose the best move based on distance from other agents"""
        if not moves:
            return self.position
            
        # Get other agents' positions
        other_positions = [
            info["position"] 
            for agent_id, info in CollaborativeAgent._shared_memory.items()
            if agent_id != self.agent_id
        ]
        
        # Calculate the sum of distances to other agents for each move
        move_scores = {}
        for move in moves:
            total_distance = sum(
                abs(move[0] - pos[0]) + abs(move[1] - pos[1])
                for pos in other_positions
            )
            move_scores[move] = total_distance
            
        # Choose the move that maximizes distance from other agents
        return max(move_scores.items(), key=lambda x: x[1])[0]
