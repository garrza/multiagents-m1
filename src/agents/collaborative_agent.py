from typing import Tuple, List, Set, Dict
from .base import Agent
import numpy as np


class CollaborativeAgent(Agent):
    """
    A collaborative vacuum cleaner agent that communicates with other agents
    to optimize cleaning efficiency.
    
    Features:
    - Maintains a shared knowledge of cleaned areas
    - Coordinates with other agents to avoid duplicate work
    - Uses a zone-based cleaning strategy
    - Adapts its behavior based on other agents' positions
    """
    
    # Class variable to share knowledge between agents
    shared_cleaned_cells: Set[Tuple[int, int]] = set()
    shared_assigned_zones: dict = {}
    
    def __init__(self, initial_position: Tuple[int, int]):
        super().__init__(initial_position)
        self.assigned_zone = None
        self.zone_center = None
        self.exploration_radius = 3
    
    def see(self, env) -> Dict:
        """
        Perceive the environment and gather information about the current state.
        Includes both local perception and shared knowledge from other agents.
        """
        current_pos = self.position
        perception = {
            "position": current_pos,
            "is_dirty": env.is_dirty(current_pos),
            "shared_cleaned_cells": self.shared_cleaned_cells.copy(),
            "assigned_zone": self.assigned_zone,
            "zone_center": self.zone_center
        }
        return perception
    
    def next(self, perception: Dict) -> str:
        """
        Decide the next action based on current perception and shared knowledge.
        """
        if perception["is_dirty"]:
            return "CLEAN"
        
        next_pos = self._get_next_position_from_perception(perception)
        current_pos = perception["position"]
        
        # Determine movement direction
        if next_pos[0] < current_pos[0]:
            return "UP"
        elif next_pos[0] > current_pos[0]:
            return "DOWN"
        elif next_pos[1] < current_pos[1]:
            return "LEFT"
        elif next_pos[1] > current_pos[1]:
            return "RIGHT"
        
        return "NOOP"
    
    def action(self, action_type: str, env) -> None:
        """
        Execute the decided action and update shared knowledge.
        """
        if action_type == "CLEAN":
            env.clean_cell(self.position)
            self.cells_cleaned += 1
            self.shared_cleaned_cells.add(self.position)
        elif action_type in ["UP", "DOWN", "LEFT", "RIGHT"]:
            new_pos = self._get_new_position(action_type)
            if self._is_valid_position(new_pos, env):
                self.position = new_pos
                self.moves_count += 1
    
    def _get_new_position(self, action: str) -> Tuple[int, int]:
        """Calculate new position based on action"""
        x, y = self.position
        if action == "UP":
            return (x - 1, y)
        elif action == "DOWN":
            return (x + 1, y)
        elif action == "LEFT":
            return (x, y - 1)
        elif action == "RIGHT":
            return (x, y + 1)
        return (x, y)
    
    def _get_next_position_from_perception(self, perception: Dict) -> Tuple[int, int]:
        """Determine next position based on perception"""
        current_pos = perception["position"]
        
        # If no zone assigned or current zone is complete, find new zone
        if not perception["assigned_zone"]:
            return self._find_nearest_dirty_cell_from_perception(perception)
        
        # Get valid moves within the zone
        valid_moves = []
        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            new_pos = (current_pos[0] + dx, current_pos[1] + dy)
            if self._is_in_zone(new_pos, perception["assigned_zone"]):
                valid_moves.append(new_pos)
        
        if not valid_moves:
            return self._find_nearest_dirty_cell_from_perception(perception)
        
        # Move towards zone center
        zone_center = perception["zone_center"]
        return min(valid_moves, key=lambda pos: abs(pos[0] - zone_center[0]) + abs(pos[1] - zone_center[1]))
    
    def _find_nearest_dirty_cell_from_perception(self, perception: Dict) -> Tuple[int, int]:
        """Find nearest dirty cell using perception data"""
        current_pos = perception["position"]
        cleaned_cells = perception["shared_cleaned_cells"]
        
        # For simplicity, return current position if no information about dirty cells
        # In practice, this would use the environment to find actual dirty cells
        return current_pos
    
    def _is_in_zone(self, pos: Tuple[int, int], zone: Tuple[Tuple[int, int], Tuple[int, int]]) -> bool:
        """Check if position is within given zone"""
        if not zone:
            return True
        
        (min_x, min_y), (max_x, max_y) = zone
        return min_x <= pos[0] <= max_x and min_y <= pos[1] <= max_y
    
    def _assign_new_zone(self, env) -> None:
        """Assigns a new zone to the agent based on dirty areas and other agents' positions"""
        width, height = env.width, env.height
        
        # Find areas with high concentration of dirt
        dirty_cells = []
        for i in range(width):
            for j in range(height):
                if env.is_dirty((i, j)) and (i, j) not in self.shared_cleaned_cells:
                    dirty_cells.append((i, j))
        
        if not dirty_cells:
            # If no dirty cells found, explore randomly
            self.assigned_zone = None
            self.zone_center = self.position
            return
        
        # Calculate dirt density for potential zones
        best_zone = None
        best_score = -1
        
        for center in dirty_cells:
            if center in self.shared_assigned_zones.values():
                continue
                
            score = self._calculate_zone_score(center, dirty_cells, env)
            
            if score > best_score:
                best_score = score
                best_zone = center
        
        if best_zone:
            self.zone_center = best_zone
            self.assigned_zone = self._get_zone_bounds(best_zone, env)
            self.shared_assigned_zones[id(self)] = best_zone
    
    def _calculate_zone_score(self, center: Tuple[int, int], dirty_cells: List[Tuple[int, int]], env) -> float:
        """Calculates a score for a potential zone based on dirt density and distance to other agents"""
        dirt_count = 0
        total_distance_to_dirt = 0
        
        for cell in dirty_cells:
            distance = abs(cell[0] - center[0]) + abs(cell[1] - center[1])
            if distance <= self.exploration_radius:
                dirt_count += 1
                total_distance_to_dirt += distance
        
        if dirt_count == 0:
            return 0
        
        # Calculate average distance to other agents' zones
        min_distance_to_other = float('inf')
        for other_center in self.shared_assigned_zones.values():
            if other_center:
                distance = abs(center[0] - other_center[0]) + abs(center[1] - other_center[1])
                min_distance_to_other = min(min_distance_to_other, distance)
        
        if min_distance_to_other == float('inf'):
            min_distance_to_other = 0
        
        # Score combines dirt density and distance to other agents
        score = (dirt_count / (total_distance_to_dirt + 1)) * min_distance_to_other
        return score
    
    def _get_zone_bounds(self, center: Tuple[int, int], env) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        """Gets the boundaries of a zone centered at the given position"""
        min_x = max(0, center[0] - self.exploration_radius)
        max_x = min(env.width - 1, center[0] + self.exploration_radius)
        min_y = max(0, center[1] - self.exploration_radius)
        max_y = min(env.height - 1, center[1] + self.exploration_radius)
        
        return ((min_x, min_y), (max_x, max_y))
