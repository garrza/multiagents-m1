import numpy as np
import random
from typing import List, Tuple, Dict


class Environment:
    def __init__(self, width: int, height: int, dirty_percentage: float):
        self.width = width
        self.height = height
        self.grid = np.zeros((height, width), dtype=bool)  # False = clean, True = dirty
        self._initialize_dirty_cells(dirty_percentage)

    def _initialize_dirty_cells(self, percentage: float) -> None:
        total_cells = self.width * self.height
        dirty_cells = int(total_cells * percentage)
        cells = [(x, y) for x in range(self.height) for y in range(self.width)]
        dirty_positions = random.sample(cells, dirty_cells)
        for x, y in dirty_positions:
            self.grid[x][y] = True

    def is_dirty(self, position: Tuple[int, int]) -> bool:
        return self.grid[position[0]][position[1]]

    def clean_cell(self, position: Tuple[int, int]) -> None:
        self.grid[position[0]][position[1]] = False

    def is_valid_position(self, position: Tuple[int, int]) -> bool:
        return 0 <= position[0] < self.height and 0 <= position[1] < self.width

    def get_dirty_count(self) -> int:
        return np.sum(self.grid)
