import pygame
from grid_world import GridWorld

pygame.init()

# Création de l'environnement
env = GridWorld(rows=5, cols=5)
env.reset()

# Taille de la fenêtre
width = 500
height = 500

# on crée la fenetre 
screen = pygame.display.set_mode((width, height))
# on donne un titre à la fenetre 
pygame.display.set_caption("GridWorld")

# on crée une variable de controle 
running = True

while running:
    screen.fill((30, 30, 30))

    for event in pygame.event.get():

        # Fermer la fenêtre
        if event.type == pygame.QUIT:
            running = False

        # Gestion du clavier : on vérifie si une touche du clavier vient d'etre pressée 
        if event.type == pygame.KEYDOWN:

            # Reset avec R
            if event.key == pygame.K_r:
                env.reset()

            # Si la partie n'est pas finie
            if not env.done:

                # si je clique sur la flèche haut et que l'action haut = 0 est autorisé bah j'applique l'action 0
                if event.key == pygame.K_UP and 0 in env.get_actions():
                    env.step(0)

                # si je clique sur la flèche bas et que l'action est autorisé bah j'applique l'action 1
                elif event.key == pygame.K_DOWN and 1 in env.get_actions():
                    env.step(1)

                # GAUCHE
                elif event.key == pygame.K_LEFT and 2 in env.get_actions():
                    env.step(2)

                # DROITE
                elif event.key == pygame.K_RIGHT and 3 in env.get_actions():
                    env.step(3)

    # Taille des cases
    cell_width = width // env.cols
    cell_height = height // env.rows

    # Dessin de la grille
    for row in range(env.rows):
        for col in range(env.cols):

            rect = pygame.Rect(
                col * cell_width,
                row * cell_height,
                cell_width - 5,
                cell_height - 5
            )

            color = (100, 100, 100)

            # Objectif
            if (row, col) == env.goal_position:
                color = (0, 255, 0)

            # Agent
            if (row, col) == env.agent_position:
                color = (0, 150, 255)

            pygame.draw.rect(screen, color, rect)

    pygame.display.update()

pygame.quit()