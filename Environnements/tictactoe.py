"""
TicTacToe (morpion) : 2 joueurs sur grille 3x3.
Joueur 1 = X (1), Joueur 2 = O (-1). 9 actions = 9 cases.
"""

import numpy as np

# Toutes les combinaisons gagnantes (3 lignes, 3 colonnes, 2 diagonales)
_WIN_COMBOS = [
    (0, 1, 2), (3, 4, 5), (6, 7, 8),
    (0, 3, 6), (1, 4, 7), (2, 5, 8),
    (0, 4, 8), (2, 4, 6),
]


class TicTacToe:
    def __init__(self):
        self.reset()

    def reset(self):
        # Plateau vide, X commence
        self.board          = [0] * 9
        self.current_player = 1
        self.done           = False
        self.winner         = None
        return self.board[:]

    def get_actions(self):
        # Une action = poser sur une case vide
        return [i for i in range(9) if self.board[i] == 0]

    def available_actions(self):
        return self.get_actions()

    def get_action_mask(self):
        # 1 si la case est libre, 0 sinon
        return [1 if self.board[i] == 0 else 0 for i in range(9)]

    def check_winner(self):
        # Retourne 1 ou -1 si une combinaison est complète, sinon None
        board = self.board
        for a, b, c in _WIN_COMBOS:
            v = board[a]
            if v != 0 and v == board[b] == board[c]:
                return v
        return None

    def step(self, action):
        # Joue un coup : pose la marque du joueur courant et change de tour
        if self.done:
            raise ValueError("La partie est déjà terminée.")
        action = int(action)
        if self.board[action] != 0:
            raise ValueError(f"Case {action} déjà occupée.")
        self.board[action] = self.current_player

        # Cas victoire
        winner = self.check_winner()
        if winner is not None:
            self.done   = True
            self.winner = winner
            return self.board[:], 1, True, {"winner": winner}

        # Cas match nul (plateau plein sans gagnant)
        if 0 not in self.board:
            self.done   = True
            self.winner = 0
            return self.board[:], 0, True, {"winner": 0}

        # Sinon, on alterne le joueur
        self.current_player = -self.current_player
        return self.board[:], 0, False, {"winner": None}

    def is_game_over(self):
        return self.done

    def encode_state(self):
        # Encodage one-hot par case : [vide, X, O] → 9 cases × 3 = 27 dim
        out = np.zeros(27, dtype=np.float32)
        for i, cell in enumerate(self.board):
            base = i * 3
            if cell == 0:   out[base]     = 1.0
            elif cell == 1: out[base + 1] = 1.0
            else:           out[base + 2] = 1.0
        return out

    def encode_action_vector(self, action: int) -> np.ndarray:
        # One-hot d'action sur 9 dimensions
        vec = np.zeros(9, dtype=np.float32)
        vec[int(action)] = 1.0
        return vec

    def render(self):
        # Affichage console du plateau
        symbols = {1: "X", -1: "O", 0: "."}
        s = [symbols[v] for v in self.board]
        print(f" {s[0]} | {s[1]} | {s[2]}")
        print("---+---+---")
        print(f" {s[3]} | {s[4]} | {s[5]}")
        print("---+---+---")
        print(f" {s[6]} | {s[7]} | {s[8]}")
        print()
        if self.done:
            if self.winner == 1:    print("X a gagné !")
            elif self.winner == -1: print("O a gagné !")
            else:                   print("Match nul.")
