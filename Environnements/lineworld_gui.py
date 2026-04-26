import os
import sys
import pygame

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, "Environnements"))

from line_world import LineWorld

# on initialise les modules de pygame
pygame.init()

# On crée l'environnement 
env = LineWorld()
#On remet le jeu dans son état initial 
env.reset()

# taille de la fenetre 
width = 800
height = 200

# On crée la fenetre graphique 
screen = pygame.display.set_mode((width, height))
pygame.display.set_caption("LineWorld")

#on crée une variable pour controler le jeu 
# tant que cette variable vaut True, la fenetre reste ouverte 
running = True 
while running:
    screen.fill((30,30,30))
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_LEFT:
                env.step(0)
            if event.key == pygame.K_RIGHT:
                env.step(1)
    cell_width = width // env.size
    for i in range(env.size):
        rect = pygame.Rect(i*cell_width,50,cell_width-10,100)
        color = (100,100,100)
        if i == env.goal_position:
            color = (0,255,0)
        if i == env.agent_position:
            color = (0,150,255)
        pygame.draw.rect(screen,color,rect)
    pygame.display.update()
pygame.quit()