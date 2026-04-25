"""
Interface graphique pygame : Humain (J1) vs RandomAgent (J2) sur Quarto.
- Phase CHOOSE : cliquer sur une pièce du panneau de droite pour la donner à l'IA.
- Phase PLACE  : cliquer sur une case du plateau pour y poser la pièce reçue.
- R : reset la partie. L'IA joue avec un délai pour la lisibilité.
"""

import os
import sys
import pygame

# Permet d'importer Agents.* depuis la racine du projet
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from quarto import QuartoEnv
from Agents.random_agent import RandomAgent

pygame.init()

# =========================
# ENV
# =========================
env   = QuartoEnv()
agent = RandomAgent()
env.reset()

# =========================
# WINDOW
# =========================
WIDTH, HEIGHT = 1400, 1050
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Quarto - Humain vs Random")

clock = pygame.time.Clock()

# =========================
# LAYOUT
# =========================
BOARD_X, BOARD_Y = 40,  240
CELL_SIZE         = 135

PIECE_X, PIECE_Y  = 720, 240
PIECE_SIZE         = 135

TOP_TEXT_Y = 70
SUB_TEXT_Y = 130

# =========================
# COLORS
# =========================
BG       = (0,   0,   0)
CELL_BG  = (205, 205, 205)
GRID     = (0,   0,   0)
WHITE    = (245, 245, 245)
BLACK    = (25,  25,  25)
TEXT     = (255, 255, 255)
ACCENT   = (120, 180, 255)
RED      = (220, 80,  80)
GREEN    = (90,  200, 120)

# =========================
# FONTS
# =========================
title_font = pygame.font.Font(None, 56)
info_font  = pygame.font.Font(None, 34)
small_font = pygame.font.Font(None, 28)

# =========================
# AI CONTROL
# =========================
AI_DELAY_MS  = 1400
pending_ai   = False
last_ai_time = 0


# =========================
# HELPERS
# =========================
def draw_piece(piece, x, y, size):
    # Décode les 4 attributs binaires de la pièce et la dessine
    color  = WHITE if (piece & 1) == 0 else BLACK   # bit 0 : couleur
    shape  = (piece >> 1) & 1                        # bit 1 : forme (rond/carré)
    filled = (piece >> 2) & 1                        # bit 2 : plein/creux
    big    = (piece >> 3) & 1                        # bit 3 : grand/petit

    cx = x + size // 2
    cy = y + size // 2

    body_h = int(size * 0.58) if big else int(size * 0.43)
    body_w = int(size * 0.34) if big else int(size * 0.26)

    top_y     = cy - body_h // 2
    body_rect = pygame.Rect(cx - body_w // 2, top_y, body_w, body_h)

    if shape == 1:
        if filled == 1:
            pygame.draw.rect(screen, color, body_rect, border_radius=5)
        else:
            pygame.draw.rect(screen, color, body_rect, width=6, border_radius=5)
    else:
        pygame.draw.ellipse(screen, color, body_rect, 0 if filled == 1 else 6)

    line_y = top_y + int(body_h * 0.42)
    pygame.draw.line(
        screen,
        (160, 160, 160) if color == WHITE else (110, 110, 110),
        (cx - body_w // 2, line_y),
        (cx + body_w // 2, line_y),
        4,
    )


def draw_cell(x, y, size):
    pygame.draw.rect(screen, CELL_BG, (x, y, size, size))
    pygame.draw.rect(screen, GRID,    (x, y, size, size), 3)


def draw_board():
    for i in range(4):
        for j in range(4):
            x = BOARD_X + j * CELL_SIZE
            y = BOARD_Y + i * CELL_SIZE
            draw_cell(x, y, CELL_SIZE)

            # env.board[i][j] retourne None ou l'entier de la pièce
            piece = env.board[i][j]
            if piece is not None:
                draw_piece(piece, x, y, CELL_SIZE)

    label = info_font.render("Plateau", True, TEXT)
    screen.blit(label, (BOARD_X, BOARD_Y - 45))


def draw_available_pieces():
    rects        = []
    piece_slots  = env.available_pieces   # liste des pièces disponibles

    for index in range(16):
        row = index // 4
        col = index % 4

        x = PIECE_X + col * PIECE_SIZE
        y = PIECE_Y + row * PIECE_SIZE

        draw_cell(x, y, PIECE_SIZE)

        if index < len(piece_slots):
            piece = piece_slots[index]
            draw_piece(piece, x, y, PIECE_SIZE)
            rects.append((pygame.Rect(x, y, PIECE_SIZE, PIECE_SIZE), piece))

    label = info_font.render("Pièces disponibles", True, TEXT)
    screen.blit(label, (PIECE_X, PIECE_Y - 45))

    return rects


def draw_piece_to_place():
    piece = env.selected_piece
    if piece is None or env.phase != "place" or env.done:
        return

    box_x    = 1120
    box_y    = 80
    box_size = 170

    txt = info_font.render("Pièce à placer", True, ACCENT)
    screen.blit(txt, (box_x, box_y))

    pygame.draw.rect(screen, CELL_BG, (box_x, box_y + 40, box_size, box_size))
    pygame.draw.rect(screen, GRID,    (box_x, box_y + 40, box_size, box_size), 3)
    draw_piece(piece, box_x, box_y + 40, box_size)


def draw_status():
    if env.done:
        if env.winner == 1:
            msg   = "Partie terminée — Humain a gagné !"
            color = GREEN
        elif env.winner == 2:
            msg   = "Partie terminée — Random a gagné !"
            color = RED
        else:
            msg   = "Partie terminée — Match nul"
            color = TEXT
    else:
        if env.phase == "choose":
            msg = f"Joueur {env.current_player} — choisissez une pièce pour l'adversaire"
        else:
            msg = f"Joueur {env.current_player} — placez la pièce reçue"
        color = TEXT

    text = title_font.render(msg, True, color)
    screen.blit(text, (40, TOP_TEXT_Y))
    screen.blit(small_font.render("R : recommencer", True, (180, 180, 180)), (40, SUB_TEXT_Y))


def draw_turn_info():
    y     = 860
    lines = [
        f"Phase actuelle : {env.phase}",
        f"Joueur courant : {env.current_player}",
    ]
    if pending_ai and not env.done:
        lines.append("Le bot réfléchit...")

    for i, line in enumerate(lines):
        screen.blit(info_font.render(line, True, TEXT), (40, y + i * 34))


def human_click_on_board(mx, my):
    """Retourne (row, col) si le clic est dans le plateau, sinon None."""
    if (BOARD_X <= mx < BOARD_X + 4 * CELL_SIZE and
            BOARD_Y <= my < BOARD_Y + 4 * CELL_SIZE):
        col = (mx - BOARD_X) // CELL_SIZE
        row = (my - BOARD_Y) // CELL_SIZE
        return (row, col)
    return None


# =========================
# MAIN LOOP
# =========================
running = True

while running:
    clock.tick(60)
    screen.fill(BG)

    draw_status()
    draw_board()
    piece_rects = draw_available_pieces()
    draw_piece_to_place()
    draw_turn_info()

    for event in pygame.event.get():

        if event.type == pygame.QUIT:
            running = False

        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_r:
                env.reset()
                pending_ai   = False
                last_ai_time = 0

        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if env.done:
                continue
            if pending_ai:
                continue
            if env.current_player != 1:
                continue

            mx, my = pygame.mouse.get_pos()

            # -------- PHASE PLACE --------
            if env.phase == "place":
                cell = human_click_on_board(mx, my)
                if cell is not None:
                    row, col = cell
                    # Encodage (row, col) → entier 16-31 attendu par l'env
                    action = 16 + row * 4 + col
                    if action in env.get_actions():
                        env.step(action)
                        if not env.done and env.current_player == 2:
                            pending_ai   = True
                            last_ai_time = pygame.time.get_ticks()

            # -------- PHASE CHOOSE --------
            elif env.phase == "choose":
                for rect, piece in piece_rects:
                    if rect.collidepoint(mx, my):
                        if piece in env.get_actions():
                            env.step(piece)
                            if not env.done and env.current_player == 2:
                                pending_ai   = True
                                last_ai_time = pygame.time.get_ticks()
                        break

    # =========================
    # IA — une action par délai
    # =========================
    if pending_ai and not env.done and env.current_player == 2:
        now = pygame.time.get_ticks()
        if now - last_ai_time >= AI_DELAY_MS:
            action = agent.Choisir_action(env)
            env.step(action)
            last_ai_time = now

            if env.current_player != 2 or env.done:
                pending_ai = False

    pygame.display.flip()

pygame.quit()