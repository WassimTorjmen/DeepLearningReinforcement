# TicTacToe — Encoding Vectors

Morpion classique 3x3, deux joueurs (X=1, O=-1). X commence.

## Vecteur d'état

Taille : **27** (fixe). Type : float32.

Chaque case est encodée sur 3 valeurs one-hot `[vide, X, O]`, les 9 cases sont concaténées dans l'ordre :

```
case 0        case 1             case 8
[vide, X, O] [vide, X, O] ... [vide, X, O]
```

Exemple — X au centre, O en haut-gauche :

```
 O | . | .
---+---+---      →  [0,0,1, 1,0,0, 1,0,0, 1,0,0, 0,1,0, 1,0,0, 1,0,0, 1,0,0, 1,0,0]
 . | X | .            case0   case1  case2  case3  case4   case5  case6  case7  case8
---+---+---
 . | . | .
```

On utilise un one-hot plutot que les valeurs brutes (0, 1, -1) pour que le réseau n'ait pas à gérer de valeurs négatives et pour distinguer clairement une case vide d'une absence de signal.

## Vecteur d'action

Taille : **9**. One-hot sur les 9 cases du plateau :

```
 0 | 1 | 2
---+---+---
 3 | 4 | 5
---+---+---
 6 | 7 | 8
```

## Masque d'action

Taille : **9**. Vaut 1 si la case est vide, 0 si elle est occupée.
