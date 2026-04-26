import os
import sys
import pygame
import torch
import numpy as np

# project_root = DeepLearningReinforcement/
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, "Agents"))
sys.path.append(os.path.join(project_root, "Environnements"))

from tictactoe import TicTacToe
from Agents.random_agent import RandomAgent

from Agents.Reinforce_critic import ReinforceAgentCritic

pygame.init()

# ── Chargement du modèle ──────────────────────────────────────
agent_rl   = ReinforceAgentCritic(27, 9)
MODEL_PATH = os.path.join(project_root, "REINFORCE_Critic_TicTacToe.pt")
if os.path.exists(MODEL_PATH):
    agent_rl.load(MODEL_PATH)
    print("✓ Modèle chargé :", MODEL_PATH)
else:
    print("✗ Modèle non trouvé :", MODEL_PATH)

agent_random = RandomAgent()
env = TicTacToe()

# ── Fenêtre ───────────────────────────────────────────────────
WIDTH, HEIGHT = 700, 720
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("TicTacToe — Humain vs Agent RL")

font       = pygame.font.Font(None, 150)
small_font = pygame.font.Font(None, 38)
info_font  = pygame.font.Font(None, 30)

WHITE = (255, 255, 255)
BLACK = (0,   0,   0)
BG    = (245, 245, 250)
BLUE  = (50,  100, 220)
RED   = (220, 60,  60)
GREEN = (50,  180, 80)

MODE    = 1
in_menu = True
AI_DELAY = 600
last_ai  = 0


def agent_rl_action():
    state     = env.encode_state()
    available = env.get_actions()
    state_t   = torch.FloatTensor(state).unsqueeze(0).to(agent_rl.device)
    with torch.no_grad():
        probs = agent_rl.policy(state_t).squeeze(0)
    mask = torch.zeros(9)
    mask[available] = 1.0
    probs = probs * mask
    probs = probs / (probs.sum() + 1e-8)
    return int(probs.argmax().item())


def draw_menu():
    screen.fill(BG)
    title = small_font.render("TicTacToe — Choisissez un mode", True, BLACK)
    screen.blit(title, (WIDTH//2 - title.get_width()//2, 80))

    btn1 = pygame.Rect(100, 180, 500, 70)
    btn2 = pygame.Rect(100, 290, 500, 70)
    pygame.draw.rect(screen, BLUE,  btn1, border_radius=10)
    pygame.draw.rect(screen, GREEN, btn2, border_radius=10)

    t1 = small_font.render("Humain (X) vs Agent RL (O)", True, WHITE)
    t2 = small_font.render("Regarder : Agent RL vs Random", True, WHITE)
    screen.blit(t1, (btn1.centerx - t1.get_width()//2, btn1.centery - t1.get_height()//2))
    screen.blit(t2, (btn2.centerx - t2.get_width()//2, btn2.centery - t2.get_height()//2))

    hint = info_font.render("M : revenir au menu | R : recommencer", True, (130, 130, 130))
    screen.blit(hint, (WIDTH//2 - hint.get_width()//2, 420))
    return btn1, btn2


def draw_game():
    screen.fill(BG)

    for x in [233, 466]:
        pygame.draw.line(screen, BLACK, (x, 60), (x, 620), 5)
    for y in [247, 433]:
        pygame.draw.line(screen, BLACK, (0, y), (700, y), 5)

    for i in range(9):
        row = i // 3
        col = i % 3
        cx  = col * 233 + 116
        cy  = row * 187 + 153
        if env.board[i] == 1:
            text = font.render("X", True, BLUE)
            screen.blit(text, (cx - text.get_width()//2, cy - text.get_height()//2))
        elif env.board[i] == -1:
            text = font.render("O", True, RED)
            screen.blit(text, (cx - text.get_width()//2, cy - text.get_height()//2))

    if env.done:
        if env.winner == 1:
            msg, color = ("Vous avez gagné !" if MODE == 1 else "Agent RL a gagné !"), GREEN
        elif env.winner == -1:
            msg, color = ("Agent RL a gagné !" if MODE == 1 else "Random a gagné !"), RED
        else:
            msg, color = "Match nul !", BLACK
    else:
        if MODE == 1:
            msg   = "Votre tour (X)" if env.current_player == 1 else "Agent RL réfléchit (O)..."
            color = BLUE if env.current_player == 1 else RED
        else:
            msg   = "Agent RL joue (X)..." if env.current_player == 1 else "Random joue (O)..."
            color = BLUE if env.current_player == 1 else RED

    status = small_font.render(msg, True, color)
    screen.blit(status, (WIDTH//2 - status.get_width()//2, 10))

    hint = info_font.render("R : recommencer | M : menu", True, (150, 150, 150))
    screen.blit(hint, (WIDTH//2 - hint.get_width()//2, 680))


# ── Boucle principale ─────────────────────────────────────────
clock   = pygame.time.Clock()
running = True
env.reset()

while running:
    clock.tick(60)

    if in_menu:
        btn1, btn2 = draw_menu()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.MOUSEBUTTONDOWN:
                if btn1.collidepoint(event.pos):
                    MODE = 1; env.reset(); in_menu = False; last_ai = 0
                if btn2.collidepoint(event.pos):
                    MODE = 2; env.reset(); in_menu = False
                    last_ai = pygame.time.get_ticks()
        pygame.display.flip()
        continue

    draw_game()

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_r:
                env.reset(); last_ai = pygame.time.get_ticks()
            if event.key == pygame.K_m:
                in_menu = True; env.reset()

        if MODE == 1 and event.type == pygame.MOUSEBUTTONDOWN:
            if not env.done and env.current_player == 1:
                mx, my = event.pos
                col = mx // 233
                row = (my - 60) // 187
                if 0 <= row < 3 and 0 <= col < 3:
                    action = row * 3 + col
                    if action in env.get_actions():
                        env.step(action)
                        last_ai = pygame.time.get_ticks()

    now = pygame.time.get_ticks()
    if not env.done and now - last_ai >= AI_DELAY:
        if MODE == 1 and env.current_player == -1:
            env.step(agent_rl_action())
            last_ai = now
        elif MODE == 2:
            if env.current_player == 1:
                env.step(agent_rl_action())
            else:
                env.step(agent_random.Choisir_action(env))
            last_ai = now

    pygame.display.flip()

pygame.quit()