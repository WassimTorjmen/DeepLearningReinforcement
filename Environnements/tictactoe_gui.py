"""
Interface graphique pygame : Joueur 1 (X) vs RandomAgent (O).
Joueur 1 peut être humain ou un agent entraîné.
  R     : reset la partie
  A     : bascule Humain ↔ Agent pour le joueur 1
"""

import os
import sys
import pygame

# Ajoute la racine du projet au PYTHONPATH pour pouvoir importer Agents.*
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from tictactoe import TicTacToe
from Agents.random_agent import RandomAgent

# ── Chargement optionnel d'un agent entraîné ─────────────────────────
# Cherche n'importe quel .pt dont le nom contient "TicTacToe", dans l'ordre :
#   1. REINFORCE_TicTacToe.pt  2. REINFORCE_*TicTacToe.pt  3. *TicTacToe.pt
try:
    import glob, torch
    from Agents.REINFORCE import ReinforceAgent

    # Cherche le premier fichier .pt correspondant à TicTacToe
    candidates = (
        glob.glob(os.path.join(project_root, "REINFORCE_TicTacToe.pt")) +
        glob.glob(os.path.join(project_root, "REINFORCE_*TicTacToe.pt")) +
        glob.glob(os.path.join(project_root, "*TicTacToe.pt"))
    )
    _MODEL_PATH = candidates[0] if candidates else None

    if _MODEL_PATH:
        # Déduit hidden_size depuis le checkpoint pour éviter les erreurs de shape
        ckpt        = torch.load(_MODEL_PATH, map_location="cpu")
        hidden_size = ckpt["net.0.bias"].shape[0]
        trained_agent = ReinforceAgent(state_size=27, num_actions=9, hidden_size=hidden_size)
        trained_agent.policy.load_state_dict(ckpt)
        trained_agent.policy.eval()
        print(f"Modèle chargé : {_MODEL_PATH}  (hidden_size={hidden_size})")
    else:
        trained_agent = ReinforceAgent(state_size=27, num_actions=9)
        print("Aucun modèle TicTacToe trouvé — poids aléatoires.")
except Exception as e:
    trained_agent = None
    print(f"Impossible de charger un agent entraîné : {e}")

# ── Mode du joueur 1 ── changer la valeur ici ou appuyer sur A en jeu ─
player1_mode = "human"   # "human" ou "agent"

# ─────────────────────────────────────────────────────────────────────

pygame.init()

env         = TicTacToe()
random_agent = RandomAgent()
env.reset()

width  = 600
height = 600

screen = pygame.display.set_mode((width, height))
pygame.display.set_caption(f"TicTacToe - {player1_mode.capitalize()} (X) vs Random (O)")

font       = pygame.font.Font(None, 150)
small_font = pygame.font.Font(None, 40)

running = True

while running:
    screen.fill((255, 255, 255))

    # ── Auto-play agent (hors events, avant le rendu) ─────────────────
    if player1_mode == "agent" and not env.done and env.current_player == 1:
        if trained_agent is not None:
            state  = env.encode_state()
            action = trained_agent.select_action(state, env.get_actions())
        else:
            action = random_agent.Choisir_action(env)  # fallback si pas de modèle
        state, reward, done, info = env.step(action)
        if not done:
            random_action = random_agent.Choisir_action(env)
            env.step(random_action)
        pygame.time.delay(1000)   # pause pour rendre le jeu lisible

    # ── Grille ────────────────────────────────────────────────────────
    pygame.draw.line(screen, (0, 0, 0), (200, 0),   (200, 600), 5)
    pygame.draw.line(screen, (0, 0, 0), (400, 0),   (400, 600), 5)
    pygame.draw.line(screen, (0, 0, 0), (0,   200), (600, 200), 5)
    pygame.draw.line(screen, (0, 0, 0), (0,   400), (600, 400), 5)

    # ── Événements ────────────────────────────────────────────────────
    for event in pygame.event.get():

        if event.type == pygame.QUIT:
            running = False

        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_r:
                env.reset()

            # Basculement Humain ↔ Agent
            if event.key == pygame.K_a:
                player1_mode = "agent" if player1_mode == "human" else "human"
                env.reset()   # reset pour repartir proprement
                pygame.display.set_caption(
                    f"TicTacToe - {player1_mode.capitalize()} (X) vs Random (O)"
                )

        # Clic humain (ignoré si mode agent)
        if event.type == pygame.MOUSEBUTTONDOWN and player1_mode == "human":
            if not env.done and env.current_player == 1:
                mouse_x, mouse_y = pygame.mouse.get_pos()
                col    = mouse_x // 200
                row    = mouse_y // 200
                action = row * 3 + col

                if action in env.get_actions():
                    state, reward, done, info = env.step(action)
                    if not done:
                        random_action = random_agent.Choisir_action(env)
                        env.step(random_action)

    # ── Affichage des pièces ──────────────────────────────────────────
    for i in range(9):
        row = i // 3
        col = i % 3
        x   = col * 200 + 70
        y   = row * 200 + 50

        if env.board[i] == 1:
            text = font.render("X", True, (0, 0, 255))
            screen.blit(text, (x, y))
        elif env.board[i] == -1:
            text = font.render("O", True, (255, 0, 0))
            screen.blit(text, (x, y))

    # ── Message de fin + mode courant ────────────────────────────────
    if env.done:
        if env.winner == 1:
            message = f"[{player1_mode.upper()}] X a gagné !"
        elif env.winner == -1:
            message = "Random (O) a gagné !"
        else:
            message = "Match nul !"
        text = small_font.render(message, True, (0, 0, 0))
        screen.blit(text, (20, 560))
    else:
        # Affiche le mode actuel en bas
        hint = small_font.render(
            f"Mode : {player1_mode.upper()}  |  A = basculer  |  R = reset",
            True, (150, 150, 150)
        )
        screen.blit(hint, (10, 565))

    pygame.display.update()

pygame.quit()
