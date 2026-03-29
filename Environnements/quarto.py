import numpy as np

# Indices des 10 lignes gagnantes (indices plats 0 à 15)
_WIN_LINES = [
    (0,  1,  2,  3),
    (4,  5,  6,  7),
    (8,  9,  10, 11),
    (12, 13, 14, 15),
    (0,  4,  8,  12),
    (1,  5,  9,  13),
    (2,  6,  10, 14),
    (3,  7,  11, 15),
    (0,  5,  10, 15),
    (3,  6,  9,  12),
]

_LINES_FOR_CELL = [[] for _ in range(16)]
for _line in _WIN_LINES:
    for _cell in _line:
        _LINES_FOR_CELL[_cell].append(_line)

_PIECE_ATTRS = tuple(
    ((i >> 0) & 1, (i >> 1) & 1, (i >> 2) & 1, (i >> 3) & 1)
    for i in range(16)
)


class _Board2D:
    """Vue 2D sur le plateau 1D — permet env.board[i][j] pour le GUI."""
    def __init__(self, flat):
        self._flat = flat

    def __getitem__(self, i):
        row = self._flat[i * 4:(i + 1) * 4]
        return [None if v == -1 else v for v in row]


class QuartoEnv:
    BOARD_SIZE   = 4
    NUM_PIECES   = 16
    PHASE_CHOOSE = 0
    PHASE_PLACE  = 1

    def __init__(self):
        self.reset()

    def reset(self):
        self._board         = [-1] * 16
        self.available      = set(range(16))
        self.piece_to_play  = -1
        self.current_player = 1
        self._phase         = self.PHASE_CHOOSE
        self.done           = False
        self.winner         = 0
        return self.get_state()

    # ------------------------------------------------------------------
    #  Propriétés de compatibilité GUI
    # ------------------------------------------------------------------

    @property
    def board(self):
        """env.board[i][j] → vue 2D. None = case vide."""
        return _Board2D(self._board)

    @property
    def phase(self):
        """Le GUI compare env.phase == 'choose' ou 'place'."""
        return "choose" if self._phase == self.PHASE_CHOOSE else "place"

    @phase.setter
    def phase(self, value):
        if isinstance(value, str):
            self._phase = self.PHASE_CHOOSE if value == "choose" else self.PHASE_PLACE
        else:
            self._phase = int(value)

    @property
    def available_pieces(self):
        """Le GUI utilise env.available_pieces."""
        return list(self.available)

    @property
    def selected_piece(self):
        """Le GUI cherche la pièce à placer via env.selected_piece."""
        return None if self.piece_to_play == -1 else self.piece_to_play

    # ------------------------------------------------------------------
    #  Interface publique
    # ------------------------------------------------------------------

    def get_state(self):
        return (
            self._board[:],
            set(self.available),
            self.piece_to_play,
            self.current_player,
            self._phase,
            self.done,
            self.winner,
        )

    def get_actions(self):
        if self.done:
            return []
        if self._phase == self.PHASE_CHOOSE:
            return list(self.available)
        return [16 + i for i in range(16) if self._board[i] == -1]

    def get_action_mask(self):
        mask = [0] * 32
        if self.done:
            return mask
        if self._phase == self.PHASE_CHOOSE:
            for p in self.available:
                mask[p] = 1
        else:
            for i in range(16):
                if self._board[i] == -1:
                    mask[16 + i] = 1
        return mask

    def step(self, action):
        if self.done:
            return self.get_state(), 0, True, None

        action = int(action)

        # PHASE PLACE
        if self._phase == self.PHASE_PLACE:
            pos = action - 16
            self._board[pos] = self.piece_to_play

            if self._check_quarto(pos):
                self.done   = True
                self.winner = self.current_player
                return self.get_state(), 1, True, None

            if not self.available:
                self.done   = True
                self.winner = 0
                return self.get_state(), 0, True, None

            self._phase = self.PHASE_CHOOSE
            return self.get_state(), 0, False, None

        # PHASE CHOOSE
        piece = action
        self.available.discard(piece)
        self.piece_to_play  = piece
        self.current_player = 2 if self.current_player == 1 else 1
        self._phase         = self.PHASE_PLACE
        return self.get_state(), 0, False, None

    # ------------------------------------------------------------------
    #  Vérification quarto
    # ------------------------------------------------------------------

    def _check_quarto(self, last_pos: int) -> bool:
        board = self._board
        for line in _LINES_FOR_CELL[last_pos]:
            a, b, c, d = line
            pa = board[a]
            if pa == -1: continue
            pb = board[b]
            if pb == -1: continue
            pc = board[c]
            if pc == -1: continue
            pd = board[d]
            if pd == -1: continue
            aa = _PIECE_ATTRS[pa]
            ab = _PIECE_ATTRS[pb]
            ac = _PIECE_ATTRS[pc]
            ad = _PIECE_ATTRS[pd]
            if (
                (aa[0] == ab[0] == ac[0] == ad[0]) or
                (aa[1] == ab[1] == ac[1] == ad[1]) or
                (aa[2] == ab[2] == ac[2] == ad[2]) or
                (aa[3] == ab[3] == ac[3] == ad[3])
            ):
                return True
        return False

    def check_quarto(self):
        for line in _WIN_LINES:
            a, b, c, d = line
            if (self._board[a] != -1 and self._board[b] != -1 and
                    self._board[c] != -1 and self._board[d] != -1):
                aa = _PIECE_ATTRS[self._board[a]]
                ab = _PIECE_ATTRS[self._board[b]]
                ac = _PIECE_ATTRS[self._board[c]]
                ad = _PIECE_ATTRS[self._board[d]]
                if (
                    (aa[0] == ab[0] == ac[0] == ad[0]) or
                    (aa[1] == ab[1] == ac[1] == ad[1]) or
                    (aa[2] == ab[2] == ac[2] == ad[2]) or
                    (aa[3] == ab[3] == ac[3] == ad[3])
                ):
                    return True
        return False

    def is_board_full(self):
        return -1 not in self._board

    # ------------------------------------------------------------------
    #  Encodage état — vecteur float32 taille 105
    # ------------------------------------------------------------------

    def encode_state(self):
        out = np.zeros(105, dtype=np.float32)

        if self.piece_to_play == -1:
            out[0] = 1.0
        else:
            attrs = _PIECE_ATTRS[self.piece_to_play]
            out[1] = attrs[0]; out[2] = attrs[1]
            out[3] = attrs[2]; out[4] = attrs[3]

        base = 5
        for i, piece in enumerate(self._board):
            idx = base + i * 5
            if piece == -1:
                out[idx] = 1.0
            else:
                attrs = _PIECE_ATTRS[piece]
                out[idx+1] = attrs[0]; out[idx+2] = attrs[1]
                out[idx+3] = attrs[2]; out[idx+4] = attrs[3]

        for p in self.available:
            out[85 + p] = 1.0

        out[101 + self._phase] = 1.0
        out[103 + (0 if self.current_player == 1 else 1)] = 1.0

        return out

    def encode_action_vector(self, action: int) -> np.ndarray:
        vec = np.zeros(32, dtype=np.float32)
        vec[int(action)] = 1.0
        return vec

    def encode_action(self, action):
        if self._phase == self.PHASE_CHOOSE:
            return int(action)
        if isinstance(action, tuple):
            return 16 + action[0] * 4 + action[1]
        return int(action)

    def decode_action(self, encoded_action):
        encoded_action = int(encoded_action)
        if encoded_action < 16:
            return encoded_action
        idx = encoded_action - 16
        return (idx >> 2, idx & 3)

    def render(self):
        print("Joueur courant :", self.current_player)
        print("Phase :", self.phase)
        print("Pièce à jouer :", self.piece_to_play if self.piece_to_play != -1 else None)
        print("Pièces disponibles :", sorted(self.available))
        print()
        for row in range(4):
            cells = []
            for col in range(4):
                p = self._board[row * 4 + col]
                cells.append("[    ]" if p == -1 else f"[{p:04b}]")
            print(" ".join(cells))
        print()
        if self.done:
            if self.winner:
                print(f"Partie terminée. Gagnant : joueur {self.winner}")
            else:
                print("Partie terminée. Match nul.")
