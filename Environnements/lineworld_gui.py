"""
Interface graphique pygame pour LineWorld.
Flèches gauche/droite pour déplacer l'agent.
"""

import pygame
from line_world import LineWorld

# Initialisation de pygame
pygame.init()

# Création + reset de l'environnement
env = LineWorld()
env.reset()

# Taille de la fenêtre
width = 800
height = 200

# Création de la fenêtre
screen = pygame.display.set_mode((width, height))
pygame.display.set_caption("LineWorld")

# Boucle principale : tant que running est True, la fenêtre reste ouverte
running = True
while running:
    screen.fill((30, 30, 30))

    # Gestion des événements clavier / fermeture
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_LEFT:
                env.step(0)   # action 0 = gauche
            if event.key == pygame.K_RIGHT:
                env.step(1)   # action 1 = droite

    # Dessin des cases : gris par défaut, vert pour l'objectif, bleu pour l'agent
    cell_width = width // env.size
    for i in range(env.size):
        rect = pygame.Rect(i * cell_width, 50, cell_width - 10, 100)
        color = (100, 100, 100)
        if i == env.goal_position:
            color = (0, 255, 0)
        if i == env.agent_position:
            color = (0, 150, 255)
        pygame.draw.rect(screen, color, rect)

    pygame.display.update()

pygame.quit()
