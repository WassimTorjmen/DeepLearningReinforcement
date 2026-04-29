import os
import sys
import pygame
import torch
import numpy as np

# project_root pointe vers DeepLearningReinforcement/
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, "Agents"))
sys.path.append(os.path.join(project_root, "Environnements"))

from quarto import QuartoEnv
from Agents.random_agent import RandomAgent
from Agents.PPO_A2C import PPOAgent

pygame.init()

# ── Chargement du modèle ──────────────────────────────────────
# Le .pt est dans Experimentations/models/ppo_a2c/
agent_rl   = PPOAgent(105, 32, hidden_size=128)
MODEL_PATH = os.path.join(project_root, "models", "ppo_a2c", "PPO_A2C_Quarto.pt")
if os.path.exists(MODEL_PATH):
    agent_rl.load(MODEL_PATH)
    print("✓ Modèle chargé :", MODEL_PATH)
else:
    print("✗ Modèle non trouvé :", MODEL_PATH)

agent_random = RandomAgent()
env = QuartoEnv()

# ── Fenêtre ───────────────────────────────────────────────────
WIDTH, HEIGHT = 1400, 1050
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Quarto — Humain vs Agent RL")
clock = pygame.time.Clock()

BOARD_X, BOARD_Y = 40,  240
CELL_SIZE         = 135
PIECE_X, PIECE_Y  = 720, 240
PIECE_SIZE         = 135
TOP_TEXT_Y        = 70
SUB_TEXT_Y        = 130

BG      = (0,   0,   0)
CELL_BG = (205, 205, 205)
GRID    = (0,   0,   0)
WHITE   = (245, 245, 245)
BLACK   = (25,  25,  25)
TEXT    = (255, 255, 255)
ACCENT  = (120, 180, 255)
RED     = (220, 80,  80)
GREEN   = (90,  200, 120)
BLUE    = (80,  140, 255)

title_font = pygame.font.Font(None, 46)
info_font  = pygame.font.Font(None, 34)
small_font = pygame.font.Font(None, 28)

MODE        = 1
in_menu     = True
AI_DELAY_MS = 1500
pending_ai  = False
last_ai_time = 0


def agent_rl_action():
    state     = env.encode_state()
    available = env.get_actions()
    state_t   = torch.FloatTensor(state).unsqueeze(0).to(agent_rl.device)
    with torch.no_grad():
        probs = agent_rl.policy(state_t).squeeze(0)
    mask = torch.zeros(32).to(agent_rl.device)
    for a in available:
        mask[a] = 1.0
    probs = probs * mask
    probs = probs / (probs.sum() + 1e-8)
    return int(probs.argmax().item())


def draw_piece(piece, x, y, size):
    color  = WHITE if (piece & 1) == 0 else BLACK
    shape  = (piece >> 1) & 1
    filled = (piece >> 2) & 1
    big    = (piece >> 3) & 1
    cx, cy = x + size // 2, y + size // 2
    body_h = int(size * 0.58) if big else int(size * 0.43)
    body_w = int(size * 0.34) if big else int(size * 0.26)
    top_y     = cy - body_h // 2
    body_rect = pygame.Rect(cx - body_w // 2, top_y, body_w, body_h)
    if shape == 1:
        pygame.draw.rect(screen, color, body_rect, border_radius=5) if filled else \
        pygame.draw.rect(screen, color, body_rect, width=6, border_radius=5)
    else:
        pygame.draw.ellipse(screen, color, body_rect, 0 if filled else 6)
    line_y = top_y + int(body_h * 0.42)
    pygame.draw.line(screen, (160,160,160) if color==WHITE else (110,110,110),
                     (cx-body_w//2, line_y), (cx+body_w//2, line_y), 4)

def draw_cell(x, y, size):
    pygame.draw.rect(screen, CELL_BG, (x, y, size, size))
    pygame.draw.rect(screen, GRID,    (x, y, size, size), 3)

def draw_board():
    for i in range(4):
        for j in range(4):
            x, y = BOARD_X + j * CELL_SIZE, BOARD_Y + i * CELL_SIZE
            draw_cell(x, y, CELL_SIZE)
            piece = env.board[i][j]
            if piece is not None:
                draw_piece(piece, x, y, CELL_SIZE)
    screen.blit(info_font.render("Plateau", True, TEXT), (BOARD_X, BOARD_Y - 45))

def draw_available_pieces():
    rects = []
    for index, piece in enumerate(env.available_pieces):
        row, col = index // 4, index % 4
        x, y = PIECE_X + col * PIECE_SIZE, PIECE_Y + row * PIECE_SIZE
        draw_cell(x, y, PIECE_SIZE)
        draw_piece(piece, x, y, PIECE_SIZE)
        rects.append((pygame.Rect(x, y, PIECE_SIZE, PIECE_SIZE), piece))
    screen.blit(info_font.render("Pièces disponibles", True, TEXT), (PIECE_X, PIECE_Y - 45))
    return rects

def draw_piece_to_place():
    piece = env.selected_piece
    if piece is None or env.phase != "place" or env.done:
        return
    box_x, box_y, box_size = 1120, 80, 170
    screen.blit(info_font.render("Pièce à placer", True, ACCENT), (box_x, box_y))
    pygame.draw.rect(screen, CELL_BG, (box_x, box_y+40, box_size, box_size))
    pygame.draw.rect(screen, GRID,    (box_x, box_y+40, box_size, box_size), 3)
    draw_piece(piece, box_x, box_y+40, box_size)

def draw_status():
    if env.done:
        if env.winner == 1:
            if MODE == 1 or MODE == 3:
                msg = "Vous avez gagné !"
            else:
                msg = "Agent RL a gagné !"
            color = GREEN
        elif env.winner == 2:
            if MODE == 1:
                msg = "Agent RL a gagné !"
            elif MODE == 3:
                msg = "Random a gagné !"
            else:
                msg = "Random a gagné !"
            color = RED
        else:
            msg, color = "Match nul !", TEXT
    else:
        if MODE == 1:
            if env.current_player == 1:
                msg   = f"Votre tour (J1) — {'choisissez une pièce' if env.phase == 'choose' else 'placez la pièce'}"
                color = BLUE
            else:
                msg, color = "Agent RL réfléchit (J2)...", RED
        elif MODE == 2:
            msg   = "Agent RL joue (J1)..." if env.current_player == 1 else "Random joue (J2)..."
            color = BLUE if env.current_player == 1 else RED
        else:  # MODE 3 : Humain vs Random
            if env.current_player == 1:
                msg   = f"Votre tour (J1) — {'choisissez une pièce' if env.phase == 'choose' else 'placez la pièce'}"
                color = BLUE
            else:
                msg, color = "Random joue (J2)...", RED
    screen.blit(title_font.render(msg, True, color), (40, TOP_TEXT_Y))
    screen.blit(small_font.render("R : recommencer | M : menu", True, (180,180,180)), (40, SUB_TEXT_Y))

def draw_turn_info():
    if MODE == 1:
        mode_str = "Humain vs Agent RL"
    elif MODE == 2:
        mode_str = "Agent RL vs Random"
    else:
        mode_str = "Humain vs Random"
    lines = [f"Phase : {env.phase}", f"Joueur : {env.current_player}",
             f"Mode : {mode_str}"]
    for i, line in enumerate(lines):
        screen.blit(info_font.render(line, True, TEXT), (40, 860 + i * 34))

def draw_menu():
    screen.fill(BG)
    title = title_font.render("Quarto — Choisissez un mode", True, TEXT)
    screen.blit(title, (WIDTH//2 - title.get_width()//2, 100))
    btn1 = pygame.Rect(300, 220, 800, 80)
    btn2 = pygame.Rect(300, 340, 800, 80)
    btn3 = pygame.Rect(300, 460, 800, 80)
    pygame.draw.rect(screen, BLUE,   btn1, border_radius=10)
    pygame.draw.rect(screen, GREEN,  btn2, border_radius=10)
    pygame.draw.rect(screen, RED,    btn3, border_radius=10)
    t1 = info_font.render("Humain (J1) vs Agent RL (J2)", True, WHITE)
    t2 = info_font.render("Regarder : Agent RL (J1) vs Random (J2)", True, WHITE)
    t3 = info_font.render("Humain (J1) vs Random (J2)", True, WHITE)
    screen.blit(t1, (btn1.centerx - t1.get_width()//2, btn1.centery - t1.get_height()//2))
    screen.blit(t2, (btn2.centerx - t2.get_width()//2, btn2.centery - t2.get_height()//2))
    screen.blit(t3, (btn3.centerx - t3.get_width()//2, btn3.centery - t3.get_height()//2))
    screen.blit(small_font.render("Cliquez pour choisir", True, (180,180,180)),
                (WIDTH//2 - small_font.size("Cliquez pour choisir")[0]//2, 600))
    return btn1, btn2, btn3


# ── Boucle principale ─────────────────────────────────────────
running = True
env.reset()

while running:
    clock.tick(60)
    screen.fill(BG)

    if in_menu:
        btn1, btn2, btn3 = draw_menu()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.MOUSEBUTTONDOWN:
                if btn1.collidepoint(event.pos):
                    MODE = 1; env.reset(); in_menu = False; pending_ai = False
                if btn2.collidepoint(event.pos):
                    MODE = 2; env.reset(); in_menu = False
                    pending_ai = True; last_ai_time = pygame.time.get_ticks()
                if btn3.collidepoint(event.pos):
                    MODE = 3; env.reset(); in_menu = False; pending_ai = False
        pygame.display.flip()
        continue

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
                env.reset(); pending_ai = False
                if MODE == 2:
                    pending_ai = True; last_ai_time = pygame.time.get_ticks()
            if event.key == pygame.K_m:
                in_menu = True; env.reset(); pending_ai = False

        # Humain joue (MODE 1 ou 3, joueur 1)
        if (MODE == 1 or MODE == 3) and event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if env.done or pending_ai or env.current_player != 1:
                continue
            mx, my = pygame.mouse.get_pos()
            if env.phase == "place":
                if (BOARD_X <= mx < BOARD_X + 4*CELL_SIZE and
                        BOARD_Y <= my < BOARD_Y + 4*CELL_SIZE):
                    col = (mx - BOARD_X) // CELL_SIZE
                    row = (my - BOARD_Y) // CELL_SIZE
                    action = 16 + row * 4 + col
                    if action in env.get_actions():
                        env.step(action)
                        if not env.done and env.current_player == 2:
                            pending_ai = True; last_ai_time = pygame.time.get_ticks()
            elif env.phase == "choose":
                for rect, piece in piece_rects:
                    if rect.collidepoint(mx, my) and piece in env.get_actions():
                        env.step(piece)
                        if not env.done and env.current_player == 2:
                            pending_ai = True; last_ai_time = pygame.time.get_ticks()
                        break

    # Agent joue automatiquement
    now = pygame.time.get_ticks()
    if pending_ai and not env.done and now - last_ai_time >= AI_DELAY_MS:
        if MODE == 1 and env.current_player == 2:
            env.step(agent_rl_action())
            last_ai_time = now
            if env.done or env.current_player == 1:
                pending_ai = False
        elif MODE == 3 and env.current_player == 2:
            env.step(agent_random.Choisir_action(env))
            last_ai_time = now
            if env.done or env.current_player == 1:
                pending_ai = False
        elif MODE == 2:
            if env.current_player == 1:
                env.step(agent_rl_action())
            else:
                env.step(agent_random.Choisir_action(env))
            last_ai_time = now
            if env.done:
                pending_ai = False

    pygame.display.flip()

pygame.quit()