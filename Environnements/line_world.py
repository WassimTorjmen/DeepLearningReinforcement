"""
LineWorld : environnement 1D le plus simple du projet.
L'agent part à gauche (case 0), doit atteindre l'objectif à droite (case size-1).
2 actions : 0 = gauche, 1 = droite. Récompense +1 si objectif atteint, 0 sinon.
"""

import numpy as np


class LineWorld:
    def __init__(self, size=6):
        # Configuration de la ligne et état initial
        self.size           = size
        self.start_position = 0
        self.goal_position  = size - 1
        self.agent_position = self.start_position
        self.done           = False

    def reset(self):
        # Remet l'agent au départ et relance la partie
        self.agent_position = self.start_position
        self.done           = False
        return self.agent_position

    def get_actions(self):
        # Liste des actions légales selon la position
        actions = []
        if self.agent_position > 0:             actions.append(0)  # gauche possible
        if self.agent_position < self.size - 1: actions.append(1)  # droite possible
        return actions

    def available_actions(self):
        # Alias compatible avec l'interface des autres envs
        return self.get_actions()

    def get_action_mask(self):
        # Masque binaire [gauche, droite] pour les réseaux qui filtrent les actions
        return [
            1 if self.agent_position > 0             else 0,
            1 if self.agent_position < self.size - 1 else 0,
        ]

    def step(self, action):
        # Applique une action et retourne (état, récompense, terminé)
        if self.done:
            return self.agent_position, 0, True
        action  = int(action)
        delta   = action * 2 - 1                                  # 0 → -1, 1 → +1
        new_pos = self.agent_position + delta
        self.agent_position = max(0, min(self.size - 1, new_pos)) # clamp aux bornes
        if self.agent_position == self.goal_position:
            self.done = True
            return self.agent_position, 1, True                   # objectif atteint
        return self.agent_position, 0, False

    def is_game_over(self):
        return self.done

    def encode_state(self):
        # Encodage utilisé par les réseaux : positions normalisées + one-hot de la grille
        norm      = self.size - 1
        pos_agent = np.array([self.agent_position / norm], dtype=np.float32)
        pos_goal  = np.array([self.goal_position  / norm], dtype=np.float32)
        grid_map  = np.zeros(self.size, dtype=np.float32)
        grid_map[self.agent_position] = 1.0
        return np.concatenate([pos_agent, pos_goal, grid_map])

    def encode_action_vector(self, action: int) -> np.ndarray:
        # Vecteur one-hot de l'action (utile pour MuZero / dynamics nets)
        vec = np.zeros(2, dtype=np.float32)
        vec[int(action)] = 1.0
        return vec

    def render(self):
        # Affichage texte simple : [A]=agent, [G]=objectif, [ ]=case vide
        cells = []
        for i in range(self.size):
            if i == self.agent_position:   cells.append("[A]")
            elif i == self.goal_position:  cells.append("[G]")
            else:                          cells.append("[ ]")
        print(" ".join(cells))
        if self.done:
            print("L'agent a atteint l'objectif !")
