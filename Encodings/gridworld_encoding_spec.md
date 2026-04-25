# GridWorld - Encoding Vectors

Environnement de navigation 2D. L'agent part de `(0, 0)`, doit rejoindre
l'objectif en haut a droite `(0, cols - 1)`, et doit eviter le piege en bas a
droite `(rows - 1, cols - 1)`.

## Vecteur d'etat

Taille : **6 + rows * cols**.

Pour une grille 5x5, la taille est donc :

```text
6 + 5 * 5 = 31
```

Type : `float32`.

Structure :

```text
[
  row_agent_norm | col_agent_norm |
  row_goal_norm  | col_goal_norm  |
  row_trap_norm  | col_trap_norm  |
  ---- carte one-hot (rows * cols) ----
]
```

Les 6 premieres valeurs sont les coordonnees normalisees entre 0 et 1 :

- position de l'agent ;
- position de l'objectif ;
- position du piege.

Le reste est un vecteur one-hot aplati en ordre row-major qui marque la case de
l'agent.

## Exemple

Sur une grille 5x5 :

- agent en `(1, 2)` ;
- objectif en `(0, 4)` ;
- piege en `(4, 4)`.

Les coordonnees normalisees sont :

```text
agent   = [1 / 4, 2 / 4] = [0.25, 0.5]
goal    = [0 / 4, 4 / 4] = [0.0, 1.0]
trap    = [4 / 4, 4 / 4] = [1.0, 1.0]
```

La case `(1, 2)` correspond a l'index aplati :

```text
1 * 5 + 2 = 7
```

Vecteur d'etat :

```text
[
  0.25, 0.5,
  0.0, 1.0,
  1.0, 1.0,
  0,0,0,0,0,
  0,0,1,0,0,
  0,0,0,0,0,
  0,0,0,0,0,
  0,0,0,0,0
]
```

## Pourquoi cet encodage ?

On donne au reseau deux types d'information complementaires :

- les coordonnees normalisees indiquent la position relative de l'agent, de
  l'objectif et du piege ;
- le one-hot indique la position exacte de l'agent sans creer de fausse relation
  numerique entre les cases.

Le piege est encode explicitement pour que l'agent puisse apprendre a le
distinguer de l'objectif et des cases neutres.

## Vecteur d'action

Taille : **4**.

One-hot :

```text
[haut, bas, gauche, droite]
```

Correspondance des actions :

| Action | Direction | Delta `(row, col)` |
|---:|---|---|
| 0 | haut | `(-1, 0)` |
| 1 | bas | `(+1, 0)` |
| 2 | gauche | `(0, -1)` |
| 3 | droite | `(0, +1)` |

## Masque d'action

Taille : **4**.

Le masque vaut `1` si l'action est legale et `0` si elle ferait sortir l'agent de
la grille.

Exemple en position `(0, 0)` :

```text
[0, 1, 0, 1]
```

Cela signifie :

- impossible d'aller en haut ;
- possible d'aller en bas ;
- impossible d'aller a gauche ;
- possible d'aller a droite.

