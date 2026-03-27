import numpy as np


class LineWorld:
    """
    Environnement de navigation 1D pour le reinforcement learning.

    L'agent se déplace sur une ligne de `size` cases (indices 0 à size-1).
    Il démarre à la position 0 (extrémité gauche) et doit atteindre la
    position size-1 (extrémité droite) pour obtenir une récompense de +1.

    C'est l'environnement le plus simple du projet : il sert de base pour
    tester et déboguer les algorithmes RL avant de passer à des environnements
    plus complexes (GridWorld, TicTacToe, Quarto).

    Interface standard partagée par tous les environnements du projet :
        reset()                → réinitialise et renvoie l'état initial
        get_actions()          → liste des actions valides (entiers)
        get_action_mask()      → masque binaire pour réseaux de neurones
        step(action)           → (état, récompense, terminé)
        encode_state()         → vecteur numpy float32 pour le réseau
        encode_action_vector() → vecteur one-hot de l'action
        render()               → affichage console
    """

    def __init__(self, size=6):
        # Taille de la ligne (nombre de cases)
        self.size           = size
        # L'agent démarre toujours à l'extrémité gauche (case 0)
        self.start_position = 0
        # L'objectif est toujours l'extrémité droite (case size-1)
        self.goal_position  = size - 1

        # État courant de l'agent
        self.agent_position = self.start_position
        # Indique si l'épisode est terminé (l'agent a atteint le goal)
        self.done           = False

    def reset(self):
        """Réinitialise l'environnement pour un nouvel épisode.
        Remet l'agent à la position de départ et renvoie cette position."""
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
        """
        Exécute une action et fait avancer l'environnement d'un pas.

        Paramètres :
            action (int) : 0 pour aller à gauche, 1 pour aller à droite.

        Retourne un tuple (état, récompense, terminé) :
            - état       : nouvelle position de l'agent (int)
            - récompense : 1 si l'agent atteint le goal, 0 sinon
            - terminé    : True si l'épisode est fini (goal atteint)

        Si l'épisode est déjà terminé, l'appel est sans effet (récompense 0).
        """
        # Sécurité : si l'épisode est déjà fini, on ne fait rien
        if self.done:
            return self.agent_position, 0, True

        action = int(action)

        # Conversion action → déplacement :
        #   action=0 (gauche) → delta = 0*2-1 = -1
        #   action=1 (droite) → delta = 1*2-1 = +1
        delta = action * 2 - 1
        new_pos = self.agent_position + delta

        # Clamp dans les bornes [0, size-1] — l'agent ne sort jamais de la ligne
        self.agent_position = max(0, min(self.size - 1, new_pos))

        # Vérification de victoire : l'agent a atteint l'objectif
        if self.agent_position == self.goal_position:
            self.done = True
            return self.agent_position, 1, True   # récompense +1

        # L'épisode continue, pas de récompense
        return self.agent_position, 0, False

    # ------------------------------------------------------------------
    #  Encoding état — vecteur float32 taille size + 3
    # ------------------------------------------------------------------

    def encode_state(self):
        """
        Construit un vecteur d'état numérique pour alimenter un réseau de neurones.

        Le vecteur combine deux types d'information complémentaires :
        - Des positions normalisées (valeurs continues entre 0 et 1), qui donnent
          au réseau une notion de distance et de progression vers l'objectif.
        - Une carte one-hot (valeurs binaires), qui donne la position exacte
          de l'agent sur la grille sans ambiguïté.

        Cette double représentation (continue + discrète) facilite l'apprentissage :
        le réseau peut exploiter soit la distance normalisée, soit la position exacte.

        Structure du vecteur (taille = 2 + size) :
        ┌──────────────────────────────────────────────────────────────┐
        │ [0]          Position agent normalisée (0.0 à 1.0)          │
        │ [1]          Position goal normalisée  (toujours 1.0)       │
        │ [2..size+1]  Carte one-hot : 1.0 sur la case de l'agent    │
        └──────────────────────────────────────────────────────────────┘

        Exemple size=6, agent en position 2 :
          [0.4, 1.0, 0, 0, 1, 0, 0, 0]   →  vecteur de taille 8
        """
        # Facteur de normalisation pour ramener les positions entre 0 et 1
        norm = self.size - 1

        # Position continue normalisée de l'agent (ex: pos 2 sur 5 → 0.4)
        pos_agent = np.array([self.agent_position / norm], dtype=np.float32)
        # Position continue normalisée du goal (toujours 1.0 car goal = size-1)
        pos_goal  = np.array([self.goal_position  / norm], dtype=np.float32)

        # Carte one-hot : un vecteur de zéros avec un 1 à la position de l'agent
        grid_map = np.zeros(self.size, dtype=np.float32)
        grid_map[self.agent_position] = 1.0

        # Concaténation des trois composantes en un seul vecteur plat
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
        """
        Affiche l'état courant de la ligne dans la console.

        Légende : [A] = agent, [G] = goal, [ ] = case vide.
        Exemple pour size=6, agent en position 2 :
            [ ] [ ] [A] [ ] [ ] [G]
        """
        cells = []
        for i in range(self.size):
            if i == self.agent_position:
                cells.append("[A]")     # Position actuelle de l'agent
            elif i == self.goal_position:
                cells.append("[G]")     # Position de l'objectif
            else:
                cells.append("[ ]")     # Case vide
        print(" ".join(cells))

        if self.done:
            print("L'agent a atteint l'objectif !")