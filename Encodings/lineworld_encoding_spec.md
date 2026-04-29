# LineWorld — Encoding Vectors

Environnement de navigation 1D. L'agent part de la case 0 et doit rejoindre la case `size - 1`.

## Vecteur d'état

Taille : **2 + size** (par défaut 8 pour size=6). Type : float32.

```
[ pos_agent_norm | pos_goal_norm | ---- carte one-hot (size) ---- ]
```

- Les deux premières valeurs sont les positions normalisées entre 0 et 1 (division par `size - 1`).
- Le reste est un vecteur one-hot de taille `size` qui marque la case de l'agent.

Exemple avec size=6, agent en position 2 :

```
[0.4, 1.0, 0, 0, 1, 0, 0, 0]
```

On combine une valeur continue (distance au goal) et une valeur discrète (position exacte) pour donner au réseau deux angles d'apprentissage complémentaires.

## Vecteur d'action

Taille : **2**. One-hot : `[gauche, droite]`.

## Masque d'action

Taille : **2**. Vaut 1 si la direction ne sort pas de la ligne, 0 sinon.
