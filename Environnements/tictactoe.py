import numpy as np

# Pré-calcul des 8 combinaisons gagnantes — calculé une seule fois au démarrage
_WIN_COMBOS = [
    (0, 1, 2), (3, 4, 5), (6, 7, 8),  # lignes
    (0, 3, 6), (1, 4, 7), (2, 5, 8),  # colonnes
    (0, 4, 8), (2, 4, 6),             # diagonales
]


class TicTacToe:

    def __init__(self):
        self.reset()

    def reset(self):
        # Liste Python de 9 entiers : 0=vide, 1=X, -1=O
        # Liste plutôt que numpy array — plus rapide pour les petits accès unitaires
        self.board = [0] * 9

        # X commence toujours
        self.current_player = 1

        self.done   = False
        self.winner = None

        return self.board[:]   # copie légère, pas np.copy()

    # ------------------------------------------------------------------
    #  Actions
    # ------------------------------------------------------------------

    def get_actions(self):
        # List comprehension : plus rapide qu'une boucle for + append
        return [i for i in range(9) if self.board[i] == 0]

    def get_action_mask(self):
        """
        Vecteur de taille 9 pour les agents deep learning.
        1 = action valide (case vide), 0 = action invalide.
        """
        return [1 if self.board[i] == 0 else 0 for i in range(9)]

    # ------------------------------------------------------------------
    #  Logique de jeu
    # ------------------------------------------------------------------

    def check_winner(self):
        """
        Parcourt les 8 combinaisons gagnantes.
        Retourne 1 si X gagne, -1 si O gagne, None sinon.
        """
        board = self.board   # référence locale : évite self.board à chaque accès

        for a, b, c in _WIN_COMBOS:
            v = board[a]
            if v != 0 and v == board[b] == board[c]:
                # v vaut 1 (X) ou -1 (O) — pas besoin de deux if séparés
                return v

        return None

    def is_draw(self):
        # Match nul = pas de gagnant ET aucune case vide
        if self.check_winner() is not None:
            return False
        return 0 not in self.board   # 'in' sur une liste de 9 éléments = très rapide

    def step(self, action):
        if self.done:
            raise ValueError("La partie est déjà terminée.")

        action = int(action)

        # Vérification directe sur le tableau — pas besoin de reconstruire get_actions()
        if self.board[action] != 0:
            raise ValueError(f"Case {action} déjà occupée.")

        # Le joueur courant joue
        self.board[action] = self.current_player

        # Vérification victoire
        winner = self.check_winner()
        if winner is not None:
            self.done   = True
            self.winner = winner
            return self.board[:], 1, True, {"winner": winner}

        # Vérification match nul
        if 0 not in self.board:
            self.done   = True
            self.winner = 0
            return self.board[:], 0, True, {"winner": 0}

        # Partie continue — changement de joueur
        # -self.current_player : 1 -> -1 -> 1 -> ...
        self.current_player = -self.current_player
        return self.board[:], 0, False, {"winner": None}

    # ------------------------------------------------------------------
    #  Encoding état — vecteur float32 taille 27
    # ------------------------------------------------------------------

    def encode_state(self):
        """
        Vecteur d'état de taille 27 pour le réseau de neurones.

        Structure :
        ┌─────────────────────────────────────────────────────┐
        │ Plateau  : 9 cases × 3 valeurs = 27                │
        │   par case : [est_vide, est_X, est_O]              │
        │   one-hot : une seule valeur à 1 par case          │
        └─────────────────────────────────────────────────────┘

        Exemples :
          case vide → [1, 0, 0]
          case X    → [0, 1, 0]
          case O    → [0, 0, 1]

        Pourquoi one-hot plutôt que la valeur brute (0, 1, -1) ?
        Le réseau de neurones traite mieux des valeurs positives
        séparées que des valeurs négatives mélangées.

        Total = 27
        """
        out = np.zeros(27, dtype=np.float32)

        for i, cell in enumerate(self.board):
            base = i * 3
            if cell == 0:
                out[base]     = 1.0   # case vide
            elif cell == 1:
                out[base + 1] = 1.0   # X
            else:
                out[base + 2] = 1.0   # O

        return out

    # ------------------------------------------------------------------
    #  Encoding action — vecteur float32 taille 9
    # ------------------------------------------------------------------

    def encode_action_vector(self, action: int) -> np.ndarray:
        """
        Vecteur one-hot de taille 9.
        Une seule position à 1.0, tout le reste à 0.

        Exemple : action = 4 (case centrale)
        → [0, 0, 0, 0, 1, 0, 0, 0, 0]

        Le plateau en one-hot :
          0 | 1 | 2
          ---------
          3 | 4 | 5
          ---------
          6 | 7 | 8
        """
        vec = np.zeros(9, dtype=np.float32)
        vec[int(action)] = 1.0
        return vec

    # ------------------------------------------------------------------
    #  Affichage console
    # ------------------------------------------------------------------

    def render(self):
        symbols = {1: "X", -1: "O", 0: "."}
        s = [symbols[v] for v in self.board]

        print(f" {s[0]} | {s[1]} | {s[2]}")
        print("---+---+---")
        print(f" {s[3]} | {s[4]} | {s[5]}")
        print("---+---+---")
        print(f" {s[6]} | {s[7]} | {s[8]}")
        print()

        if self.done:
            if self.winner == 1:
                print("X a gagné !")
            elif self.winner == -1:
                print("O a gagné !")
            else:
                print("Match nul.")