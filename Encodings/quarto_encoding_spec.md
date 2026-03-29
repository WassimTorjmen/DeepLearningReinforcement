# Quarto — Encoding Vectors

L'objectif est de transformer l'état du jeu Quarto en un vecteur de nombres que le modèle peut comprendre.
Comme un réseau de neurones ne comprend que des chiffres, on convertit tout le jeu (plateau, pièces, joueur...) en une seule liste de valeurs.

## Vecteur d'état

Chaque état du jeu est représenté par un vecteur de **105 valeurs**.
Ce vecteur contient toutes les informations importantes du jeu à un instant donné.

### Structure du vecteur

| Bloc | Taille | Contenu |
|------|--------|---------|
| Pièce à jouer | 5 | Flag + 4 attributs |
| Plateau | 80 | 16 cases x 5 valeurs |
| Pièces disponibles | 16 | 1 par pièce |
| Phase du jeu | 2 | One-hot |
| Joueur courant | 2 | One-hot |
| **Total** | **105** | |

### Pièce à jouer

- `[1, 0, 0, 0, 0]` → aucune pièce sélectionnée
- `[0, a1, a2, a3, a4]` → pièce avec ses 4 attributs binaires

### Plateau

Chaque case est encodée de la même façon :

- Case vide → `[1, 0, 0, 0, 0]`
- Case occupée → `[0, a1, a2, a3, a4]`

Ca permet de comparer facilement les attributs des pièces (important dans Quarto).

### Pièces disponibles

16 valeurs (une par pièce) :

- `1` → pièce encore disponible
- `0` → déjà utilisée

### Phase du jeu

- `[1, 0]` → choisir une pièce
- `[0, 1]` → placer une pièce

### Joueur courant

- `[1, 0]` → joueur 1
- `[0, 1]` → joueur 2

## Encodage des actions

### Espace d'actions (32 actions)

- `0 → 15` → choisir une pièce
- `16 → 31` → placer une pièce sur le plateau

### Vecteur d'action

Chaque action est transformée en one-hot vector (taille **32**).

### Conversion des actions

- Choisir une pièce → retourne directement son index (0 à 15)
- Placer une pièce → convertit (ligne, colonne) en un nombre entre 16 et 31 via `16 + ligne * 4 + colonne`
