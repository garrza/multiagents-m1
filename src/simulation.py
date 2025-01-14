from typing import List, Dict
from .environment import Environment
from .agents.base import Agent
from .agents.basic_agent import BasicAgent
from .agents.smart_agent import SmartAgent
from .agents.efficient_agent import EfficientAgent


class Simulation:
    def __init__(
        self,
        width: int,
        height: int,
        num_agents: int,
        dirty_percentage: float,
        max_time: int,
    ):
        self.environment = Environment(width, height, dirty_percentage)
        self.agents = self._initialize_agents(num_agents)
        self.max_time = max_time
        self.current_time = 0
        self.initial_dirty = self.environment.get_dirty_count()

    def _initialize_agents(self, num_agents: int) -> List[Agent]:
        agents = []
        for i in range(num_agents):
            if i % 3 == 0:  # Alternar entre los tipos de agentes
                agents.append(BasicAgent((0, 0)))
            elif i % 3 == 1:
                agents.append(SmartAgent((0, 0)))
            else:
                agents.append(EfficientAgent((0, 0)))
        return agents

    def run(self) -> Dict:
        while (
            self.current_time < self.max_time and self.environment.get_dirty_count() > 0
        ):
            for agent in self.agents:
                perception = agent.see(self.environment)
                action = agent.next(perception)
                agent.action(action, self.environment)
            self.current_time += 1

        return self._get_statistics()

    def _get_statistics(self) -> Dict:
        total_moves = sum(agent.moves_count for agent in self.agents)
        cells_cleaned = sum(agent.cells_cleaned for agent in self.agents)
        final_dirty = self.environment.get_dirty_count()
        clean_percentage = (
            (self.initial_dirty - final_dirty) / self.initial_dirty
        ) * 100

        return {
            "time_taken": self.current_time,
            "clean_percentage": clean_percentage,
            "total_moves": total_moves,
            "cells_cleaned": cells_cleaned,
        }
