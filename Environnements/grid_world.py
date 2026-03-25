import numpy as np

# Les 4 actions possibles — pré-calculées comme deltas (drow, dcol)
# Evite les if/elif en cascade dans step()
_ACTIONS = {
    0: (-1,  0),   # haut
    1: ( 1,  0),   # bas
    2: ( 0, -1),   # gauche
    3: ( 0,  1),   # droite
}


class GridWorld:

    def __init__(self, rows=5, cols=5):
        self.rows = rows
        self.cols = cols

        # Positions stockées comme tuples Python — pas numpy
        self.start_position = (0, 0)
        self.goal_position  = (rows - 1, cols - 1)

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
        Retourne les actions valides selon la position courante.
        0=haut, 1=bas, 2=gauche, 3=droite
        """
        row, col = self.agent_position
        actions  = []

        # On vérifie les 4 directions dans l'ordre fixe
        if row > 0:              actions.append(0)   # haut  possible
        if row < self.rows - 1:  actions.append(1)   # bas   possible
        if col > 0:              actions.append(2)   # gauche possible
        if col < self.cols - 1:  actions.append(3)   # droite possible

        return actions

    def get_action_mask(self):
        """
        Vecteur de taille 4 pour les agents deep learning.
        1 = action valide, 0 = action invalide (mur).

        Exemple coin haut-gauche (0,0) : [0, 1, 0, 1]
          → ne peut pas aller en haut ni à gauche
        """
        row, col = self.agent_position
        return [
            1 if row > 0             else 0,   # haut
            1 if row < self.rows - 1 else 0,   # bas
            1 if col > 0             else 0,   # gauche
            1 if col < self.cols - 1 else 0,   # droite
        ]

    # ------------------------------------------------------------------
    #  Logique de jeu
    # ------------------------------------------------------------------

    def step(self, action):
        if self.done:
            return self.agent_position, 0, True

        action = int(action)

        row, col    = self.agent_position
        drow, dcol  = _ACTIONS[action]

        # Calcul de la nouvelle position
        new_row = row + drow
        new_col = col + dcol

        # Clamp dans les bornes — si l'action est invalide, l'agent ne bouge pas
        # (plus robuste que lever une exception)
        new_row = max(0, min(self.rows - 1, new_row))
        new_col = max(0, min(self.cols - 1, new_col))

        self.agent_position = (new_row, new_col)

        if self.agent_position == self.goal_position:
            self.done = True
            return self.agent_position, 1, True

        return self.agent_position, 0, False

    # ------------------------------------------------------------------
    #  Encoding état — vecteur float32 taille (rows*cols)*2 + 2
    # ------------------------------------------------------------------

    def encode_state(self):
        """
        Vecteur d'état pour le réseau de neurones.

        Structure :
        ┌──────────────────────────────────────────────────────────────┐
        │ Position agent : 2 valeurs normalisées                      │
        │   [row / (rows-1),  col / (cols-1)]                        │
        │   → valeurs entre 0.0 et 1.0                               │
        │                                                             │
        │ Position goal  : 2 valeurs normalisées                      │
        │   [row / (rows-1),  col / (cols-1)]                        │
        │   → toujours (1.0, 1.0) pour une grille standard           │
        │   → utile si le goal change entre les épisodes             │
        │                                                             │
        │ Carte one-hot  : rows*cols valeurs                         │
        │   1.0 sur la case de l'agent, 0.0 partout ailleurs         │
        └──────────────────────────────────────────────────────────────┘

        Exemple grille 5x5, agent en (1,2) :
          position normalisée : [0.25, 0.5]
          goal normalisé      : [1.0,  1.0]
          carte (25 valeurs)  : 0 partout sauf index 7 (=1*5+2) à 1.0

        Total = 2 + 2 + rows*cols
        Pour une grille 5x5 : 2 + 2 + 25 = 29
        """
        ar, ac = self.agent_position
        gr, gc = self.goal_position

        # Normalisation : ramène les coordonnées entre 0 et 1
        row_norm = self.rows - 1
        col_norm = self.cols - 1

        pos_agent = np.array([ar / row_norm, ac / col_norm], dtype=np.float32)
        pos_goal  = np.array([gr / row_norm, gc / col_norm], dtype=np.float32)

        # Carte one-hot : 1.0 sur la case de l'agent
        grid_map         = np.zeros(self.rows * self.cols, dtype=np.float32)
        grid_map[ar * self.cols + ac] = 1.0

        return np.concatenate([pos_agent, pos_goal, grid_map])

    # ------------------------------------------------------------------
    #  Encoding action — vecteur float32 taille 4
    # ------------------------------------------------------------------

    def encode_action_vector(self, action: int) -> np.ndarray:
        """
        Vecteur one-hot de taille 4.
        Une seule position à 1.0.

        Index :  0      1      2        3
        Action : haut   bas    gauche   droite

        Exemple : action=3 (droite) → [0, 0, 0, 1]
        """
        vec = np.zeros(4, dtype=np.float32)
        vec[int(action)] = 1.0
        return vec

    # ------------------------------------------------------------------
    #  Affichage console
    # ------------------------------------------------------------------

    def render(self):
        for row in range(self.rows):
            line = []
            for col in range(self.cols):
                if (row, col) == self.agent_position:
                    line.append("A")
                elif (row, col) == self.goal_position:
                    line.append("G")
                else:
                    line.append(".")
            print(" ".join(line))
        print()

        if self.done:
            print("L'agent a atteint l'objectif !")