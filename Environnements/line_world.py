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

    # ------------------------------------------------------------------
    #  Actions
    # ------------------------------------------------------------------

    def get_actions(self):
        """
        0 = gauche, 1 = droite.
        On ne propose que les actions qui ne sortent pas de la ligne.
        """
        actions = []
        if self.agent_position > 0:              actions.append(0)
        if self.agent_position < self.size - 1:  actions.append(1)
        return actions

    def get_action_mask(self):
        """
        Vecteur de taille 2 pour les agents deep learning.
        1 = action valide, 0 = action invalide (bord).

        Exemple agent en position 0 (bord gauche) : [0, 1]
        Exemple agent en position 5 (bord droit)  : [1, 0]
        Exemple agent en position 2 (milieu)       : [1, 1]
        """
        return [
            1 if self.agent_position > 0             else 0,   # gauche
            1 if self.agent_position < self.size - 1 else 0,   # droite
        ]

    # ------------------------------------------------------------------
    #  Logique de jeu
    # ------------------------------------------------------------------

    def step(self, action):
        if self.done:
            return self.agent_position, 0, True

        action = int(action)

        # Déplacement : -1 pour gauche, +1 pour droite
        # On utilise (action * 2 - 1) : action=0 → -1, action=1 → +1
        delta = action * 2 - 1
        new_pos = self.agent_position + delta

        # Clamp dans les bornes — l'agent ne sort jamais de la ligne
        self.agent_position = max(0, min(self.size - 1, new_pos))

        if self.agent_position == self.goal_position:
            self.done = True
            return self.agent_position, 1, True

        return self.agent_position, 0, False

    # ------------------------------------------------------------------
    #  Encoding état — vecteur float32 taille size + 3
    # ------------------------------------------------------------------

    def encode_state(self):
        """
        Vecteur d'état pour le réseau de neurones.

        Structure :
        ┌──────────────────────────────────────────────────────────────┐
        │ Position agent normalisée : 1 valeur                        │
        │   agent_position / (size - 1)  → entre 0.0 et 1.0          │
        │                                                             │
        │ Position goal normalisée  : 1 valeur                        │
        │   goal_position / (size - 1)   → toujours 1.0              │
        │                                                             │
        │ Carte one-hot : size valeurs                                │
        │   1.0 sur la case de l'agent, 0.0 partout ailleurs         │
        └──────────────────────────────────────────────────────────────┘

        Exemple size=6, agent en position 2 :
          position normalisée : [0.4]
          goal normalisé      : [1.0]
          carte (6 valeurs)   : [0, 0, 1, 0, 0, 0]

        Total = 1 + 1 + size
        Pour size=6 : 1 + 1 + 6 = 8
        """
        norm = self.size - 1

        pos_agent = np.array([self.agent_position / norm], dtype=np.float32)
        pos_goal  = np.array([self.goal_position  / norm], dtype=np.float32)

        # Carte one-hot
        grid_map = np.zeros(self.size, dtype=np.float32)
        grid_map[self.agent_position] = 1.0

        return np.concatenate([pos_agent, pos_goal, grid_map])

    # ------------------------------------------------------------------
    #  Encoding action — vecteur float32 taille 2
    # ------------------------------------------------------------------

    def encode_action_vector(self, action: int) -> np.ndarray:
        """
        Vecteur one-hot de taille 2.

        Index :  0        1
        Action : gauche   droite

        Exemple : action=1 (droite) → [0, 1]
        Exemple : action=0 (gauche) → [1, 0]
        """
        vec = np.zeros(2, dtype=np.float32)
        vec[int(action)] = 1.0
        return vec

    # ------------------------------------------------------------------
    #  Affichage console
    # ------------------------------------------------------------------

    def render(self):
        cells = []
        for i in range(self.size):
            if i == self.agent_position:
                cells.append("[A]")
            elif i == self.goal_position:
                cells.append("[G]")
            else:
                cells.append("[ ]")
        print(" ".join(cells))

        if self.done:
            print("L'agent a atteint l'objectif !")