import os
import sys

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)


import pygame
from tictactoe import TicTacToe
from Agents.random_agent import RandomAgent

pygame.init()

# Création de l'environnement et de l'agent random
env = TicTacToe()
agent = RandomAgent()
env.reset()

# Taille de la fenêtre
width = 600
height = 600

screen = pygame.display.set_mode((width, height))
pygame.display.set_caption("TicTacToe - Humain vs Random")

# Police pour afficher X et O
font = pygame.font.Font(None, 150)
small_font = pygame.font.Font(None, 40)

running = True

while running:
    screen.fill((255, 255, 255))

    # Dessiner la grille
    pygame.draw.line(screen, (0, 0, 0), (200, 0), (200, 600), 5)
    pygame.draw.line(screen, (0, 0, 0), (400, 0), (400, 600), 5)
    pygame.draw.line(screen, (0, 0, 0), (0, 200), (600, 200), 5)
    pygame.draw.line(screen, (0, 0, 0), (0, 400), (600, 400), 5)

    for event in pygame.event.get():
        # Fermer la fenêtre
        if event.type == pygame.QUIT:
            running = False

        # Touche R pour reset
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_r:
                env.reset()

        # quand je clique, ça veut dire je peux jouer 
        if event.type == pygame.MOUSEBUTTONDOWN:
            # On ne joue que si la partie n'est pas finie
            # et si c'est le tour du joueur humain (X = 1)
            if not env.done and env.current_player == 1:
                mouse_x, mouse_y = pygame.mouse.get_pos()

                col = mouse_x // 200
                row = mouse_y // 200
                action = row * 3 + col

                # Vérifie que la case est libre
                if action in env.get_actions():
                    # l'humain place son X
                    state, reward, done, info = env.step(action)

                    # Si la partie continue, le random joue
                    if not done:
                        random_action = agent.Choisir_action(env)
                        state, reward, done, info = env.step(random_action)

    # Dessiner X et O
    for i in range(9):
        row = i // 3
        col = i % 3

        x = col * 200 + 70
        y = row * 200 + 50

        if env.board[i] == 1:
            text = font.render("X", True, (0, 0, 255))
            screen.blit(text, (x, y))
        elif env.board[i] == -1:
            text = font.render("O", True, (255, 0, 0))
            screen.blit(text, (x, y))

    # Message de fin de partie
    if env.done:
        if env.winner == 1:
            message = "Le joueur humain (X) a gagne !"
        elif env.winner == -1:
            message = "Le joueur random (O) a gagne !"
        else:
            message = "Match nul !"

        text = small_font.render(message, True, (0, 0, 0))
        screen.blit(text, (20, 560))

    pygame.display.update()

pygame.quit()