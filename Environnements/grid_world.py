"""
GridWorld : environnement 2D.
Départ haut-gauche, objectif (+1) haut-droite, piège (-1) bas-droite.
4 actions : 0 = haut, 1 = bas, 2 = gauche, 3 = droite.
"""

import numpy as np

# Déplacements (drow, dcol) associés à chaque indice d'action
_ACTIONS = {
    0: (-1,  0),  # haut
    1: ( 1,  0),  # bas
    2: ( 0, -1),  # gauche
    3: ( 0,  1),  # droite
}


class GridWorld:
    def __init__(self, rows=5, cols=5):
        # Dimensions et positions clés du plateau
        self.rows           = rows
        self.cols           = cols
        self.start_position = (0, 0)
        self.goal_position  = (0, cols - 1)        # haut droite  → +1
        self.trap_position  = (rows - 1, cols - 1) # bas droite   → -1
        self.agent_position = self.start_position
        self.done           = False

    def reset(self):
        # Remet l'agent au départ
        self.agent_position = self.start_position
        self.done           = False
        return self.agent_position

    def get_actions(self):
        # Filtre les actions qui sortiraient du plateau
        row, col = self.agent_position
        actions  = []
        if row > 0:             actions.append(0)
        if row < self.rows - 1: actions.append(1)
        if col > 0:             actions.append(2)
        if col < self.cols - 1: actions.append(3)
        return actions

    def available_actions(self):
        return self.get_actions()

    def get_action_mask(self):
        # Masque binaire utilisé par les politiques pour ignorer les actions invalides
        row, col = self.agent_position
        return [
            1 if row > 0             else 0,
            1 if row < self.rows - 1 else 0,
            1 if col > 0             else 0,
            1 if col < self.cols - 1 else 0,
        ]

    def step(self, action):
        # Applique l'action et calcule la récompense
        if self.done:
            return self.agent_position, 0, True
        row, col   = self.agent_position
        drow, dcol = _ACTIONS[int(action)]
        new_row    = max(0, min(self.rows - 1, row + drow))
        new_col    = max(0, min(self.cols - 1, col + dcol))
        self.agent_position = (new_row, new_col)

        # Récompense terminale selon la case atteinte
        if self.agent_position == self.goal_position:
            self.done = True
            return self.agent_position, 1, True
        if self.agent_position == self.trap_position:
            self.done = True
            return self.agent_position, -1, True

        return self.agent_position, 0, False

    def is_game_over(self):
        return self.done

    def encode_state(self):
        # Vecteur d'état : positions normalisées (agent, but, piège) + one-hot de la grille
        ar, ac = self.agent_position
        gr, gc = self.goal_position
        tr, tc = self.trap_position
        row_norm = self.rows - 1
        col_norm = self.cols - 1
        pos_agent = np.array([ar / row_norm, ac / col_norm], dtype=np.float32)
        pos_goal  = np.array([gr / row_norm, gc / col_norm], dtype=np.float32)
        pos_trap  = np.array([tr / row_norm, tc / col_norm], dtype=np.float32)
        grid_map  = np.zeros(self.rows * self.cols, dtype=np.float32)
        grid_map[ar * self.cols + ac] = 1.0
        return np.concatenate([pos_agent, pos_goal, pos_trap, grid_map])

    def encode_action_vector(self, action: int) -> np.ndarray:
        # One-hot d'action (4 dimensions)
        vec = np.zeros(4, dtype=np.float32)
        vec[int(action)] = 1.0
        return vec

    def render(self):
        # Affichage texte : A=agent, G=objectif, T=piège, .=vide
        for row in range(self.rows):
            line = []
            for col in range(self.cols):
                if (row, col) == self.agent_position:  line.append("A")
                elif (row, col) == self.goal_position: line.append("G")
                elif (row, col) == self.trap_position: line.append("T")
                else:                                   line.append(".")
            print(" ".join(line))
        print()
