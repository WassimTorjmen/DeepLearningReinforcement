# GridWorld — Encoding Vectors

Environnement de navigation 2D. L'agent part de (0, 0) et doit rejoindre (rows-1, cols-1).

## Vecteur d'état

Taille : **4 + rows * cols** (par défaut 29 pour une grille 5x5). Type : float32.

```
[ row_agent_norm | col_agent_norm | row_goal_norm | col_goal_norm | ---- carte one-hot (rows*cols) ---- ]
```

- Les 4 premières valeurs sont les coordonnées normalisées entre 0 et 1 de l'agent et du goal.
- Le reste est un vecteur one-hot aplati (row-major) qui marque la case de l'agent.

Exemple sur grille 5x5, agent en (1, 2) :

```
[0.25, 0.5, 1.0, 1.0, 0,0,0,0,0, 0,0,1,0,0, 0,0,0,0,0, 0,0,0,0,0, 0,0,0,0,0]
                       └ ligne 0 ┘ └ ligne 1 ┘
                              index 7 = 1*5+2 ↑
```

Meme logique que LineWorld : on donne au réseau à la fois la distance continue et la position exacte.

## Vecteur d'action

Taille : **4**. One-hot : `[haut, bas, gauche, droite]`.

Les deltas associés sont (-1,0), (+1,0), (0,-1), (0,+1).

## Masque d'action

Taille : **4**. Vaut 1 si la direction ne sort pas de la grille, 0 sinon.
