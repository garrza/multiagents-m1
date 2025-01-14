from typing import Dict, Tuple, List
from .basic_agent import BasicAgent
from ..environment import Environment
import random


class EfficientAgent(BasicAgent):
    def __init__(self, position: Tuple[int, int]):
        super().__init__(position)
        self.visited_cells = set()  # Memoria de celdas visitadas

    def see(self, environment: Environment) -> Dict:
        """Percibir el entorno y agregar celdas sucias visibles."""
        perception = super().see(environment)

        # Celdas sucias alrededor
        dirty_positions = [
            (self.position[0] + dx, self.position[1] + dy)
            for dx, dy in self.directions
            if environment.is_valid_position((self.position[0] + dx, self.position[1] + dy)) 
            and environment.is_dirty((self.position[0] + dx, self.position[1] + dy))
        ]
        perception["surrounding_dirty"] = dirty_positions

        # Celdas ya visitadas
        perception["visited_cells"] = self.visited_cells
        return perception

    def next(self, perception: Dict) -> str:
        """Decidir la próxima acción."""
        self.visited_cells.add(self.position)  # Marcar posición actual como visitada

        if perception["is_dirty"]:  # Limpiar celda actual si está sucia
            return "CLEAN"

        if perception["surrounding_dirty"]:  # Moverse a una celda sucia cercana
            return "MOVE_TO_DIRT"

        # Si no hay celdas sucias cercanas, buscar celdas no visitadas
        unvisited_moves = [
            pos for pos in perception["valid_moves"] if pos not in self.visited_cells
        ]
        if unvisited_moves:
            return "MOVE_TO_UNVISITED"

        # Si todo está visitado y limpio, moverse aleatoriamente
        return "MOVE"

    def action(self, action_type: str, environment: Environment) -> None:
        """Ejecutar la acción."""
        if action_type == "CLEAN":
            environment.clean_cell(self.position)
            self.cells_cleaned += 1

        elif action_type == "MOVE_TO_DIRT":
            # Moverse a la celda sucia más cercana
            dirty_positions = self.see(environment)["surrounding_dirty"]
            if dirty_positions:
                self.position = min(
                    dirty_positions,
                    key=lambda pos: self._manhattan_distance(self.position, pos),
                )
                self.moves_count += 1

        elif action_type == "MOVE_TO_UNVISITED":
            # Moverse a una celda no visitada
            unvisited_moves = [
                pos for pos in self._get_valid_moves(environment)
                if pos not in self.visited_cells
            ]
            if unvisited_moves:
                self.position = random.choice(unvisited_moves)
                self.moves_count += 1

        else:
            # Movimiento estándar (aleatorio)
            super().action("MOVE", environment)

    def _manhattan_distance(self, pos1: Tuple[int, int], pos2: Tuple[int, int]) -> int:
        """Calcular distancia Manhattan entre dos celdas."""
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
