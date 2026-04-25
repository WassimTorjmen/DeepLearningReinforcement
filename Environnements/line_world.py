import numpy as np


class LineWorld:
    def __init__(self, size=6):
        self.size           = size
        self.start_position = 0
        self.goal_position  = size - 1
        self.agent_position = self.start_position
        self.done           = False

    def reset(self):
        self.agent_position = self.start_position
        self.done           = False
        return self.agent_position

    def get_actions(self):
        actions = []
        if self.agent_position > 0:             actions.append(0)
        if self.agent_position < self.size - 1: actions.append(1)
        return actions

    def available_actions(self):
        return self.get_actions()

    def get_action_mask(self):
        return [
            1 if self.agent_position > 0             else 0,
            1 if self.agent_position < self.size - 1 else 0,
        ]

    def step(self, action):
        if self.done:
            return self.agent_position, 0, True
        action  = int(action)
        delta   = action * 2 - 1
        new_pos = self.agent_position + delta
        self.agent_position = max(0, min(self.size - 1, new_pos))
        if self.agent_position == self.goal_position:
            self.done = True
            return self.agent_position, 1, True
        return self.agent_position, 0, False

    def is_game_over(self):
        return self.done

    def encode_state(self):
        norm      = self.size - 1
        pos_agent = np.array([self.agent_position / norm], dtype=np.float32)
        pos_goal  = np.array([self.goal_position  / norm], dtype=np.float32)
        grid_map  = np.zeros(self.size, dtype=np.float32)
        grid_map[self.agent_position] = 1.0
        return np.concatenate([pos_agent, pos_goal, grid_map])

    def encode_action_vector(self, action: int) -> np.ndarray:
        vec = np.zeros(2, dtype=np.float32)
        vec[int(action)] = 1.0
        return vec

    def render(self):
        cells = []
        for i in range(self.size):
            if i == self.agent_position:   cells.append("[A]")
            elif i == self.goal_position:  cells.append("[G]")
            else:                          cells.append("[ ]")
        print(" ".join(cells))
        if self.done:
            print("L'agent a atteint l'objectif !")