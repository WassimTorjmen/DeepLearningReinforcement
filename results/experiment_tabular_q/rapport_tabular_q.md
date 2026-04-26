# Rapport d'expérimentation — TabularQ (Q-Learning tabulaire)

> Épisodes d'entraînement : 10,000


## Algorithme

Le **Q-Learning tabulaire** stocke une valeur Q(s, a) pour chaque paire (état, action) rencontrée. La politique est ε-greedy :
- Avec probabilité ε : action aléatoire (exploration)
- Sinon : argmax_a Q(s, a) (exploitation)

Mise à jour : `Q(s,a) ← Q(s,a) + α × [r + γ × max Q(s',a') − Q(s,a)]`


## Hyperparamètres par défaut

| Paramètre | Symbole | Valeur | Rôle |
|---|---|---|---|
| Taux d'apprentissage | α | 0.1–0.3 | Vitesse de mise à jour des Q-valeurs |
| Facteur de discount  | γ | 0.99 | Poids des récompenses futures |
| Epsilon initial      | ε | 1.0 | Exploration pure au début |
| Décroissance epsilon | ε_decay | 0.999–0.9999 | Vitesse de transition vers l'exploitation |
| Epsilon minimum      | ε_min | 0.05 | Exploration résiduelle permanente |

## Compatibilité avec les environnements

| Environnement | Compatible | Raison |
|---|---|---|
| LineWorld | ✓ | Espace d'états discret et très petit |
| GridWorld | ✓ | Encodage one-hot → états distincts |
| TicTacToe | ✓ | ~5 478 états théoriques — Q-table compacte |
| Quarto    | ⚠ | Grand espace d'états → couverture partielle, DQN préférable |

## Résultats par checkpoint

### LineWorld

| Épisodes | Score moyen | Longueur moy | Temps/coup (ms) |
|---|---|---|---|
| 1,000 | 1.0000 | 5.00 | 0.0013 |

### GridWorld (5×5)

| Épisodes | Score moyen | Longueur moy | Temps/coup (ms) |
|---|---|---|---|
| 1,000 | 1.0000 | 4.00 | 0.0018 |

### TicTacToe (vs Random)

| Épisodes | Score moyen | Longueur moy | Temps/coup (ms) |
|---|---|---|---|
| 1,000 | 0.8220 | 6.05 | 0.0027 |

### Quarto (vs Random)

| Épisodes | Score moyen | Longueur moy | Temps/coup (ms) |
|---|---|---|---|
| 1,000 | 0.2060 | 21.04 | 0.0050 |

## Score final — Politique gloutonne (ε=0)

| Environnement | Score moyen | Victoires | Longueur | ms/coup | Q-table |
|---|---|---|---|---|---|
| LineWorld | 1.0000 | 100.0% | 5.23 | 0.0012 | 5 |
| GridWorld (5×5) | 1.0000 | 100.0% | 4.30 | 0.0019 | 23 |
| TicTacToe (vs Random) | 0.9180 | 94.4% | 5.91 | 0.0030 | 1,390 |
| Quarto (vs Random) | 0.2340 | 60.6% | 20.52 | 0.0060 | 127,034 |

## Analyse des résultats

### Comportement de l'exploration (ε-greedy)
- Au début : ε=1.0 → exploration pure. L'agent découvre l'espace d'états.
- Progressivement : ε décroît → exploitation de plus en plus prioritaire.
- En fin d'entraînement : ε≈ε_min → quasi-exploitation, évaluation proche de la politique optimale.

### Impact de alpha (taux d'apprentissage)
- α élevé (0.3–0.5) : convergence rapide mais instabilité possible.
- α faible (0.05–0.1) : convergence lente mais plus stable.
- Optimum trouvé par recherche d'hyperparamètres pour chaque env.

### Croissance de la Q-table
- LineWorld / GridWorld : Q-table se stabilise rapidement (espace d'états petit).
- TicTacToe : croissance modérée jusqu'à couvrir ~5k états.
- Quarto : Q-table continue de croître — couverture partielle de l'espace. C'est la limite principale de TabularQ sur cet env.

### Comparaison avec les autres agents
- **vs RandomRollout** : TabularQ est plus rapide à l'inférence (lookup O(1) vs simulations).
- **vs DQN** : TabularQ est optimal sur petits espaces d'états (pas besoin d'approximation).
- **vs AlphaZero** : TabularQ ne généralise pas — chaque état vu pour la première fois obtient Q=0.

### Limites
- Ne généralise pas aux états non vus (contrairement aux réseaux neuronaux).
- Mémoire proportionnelle au nombre d'états visités.
- Inapplicable aux environnements à espace d'états continu.
- Sur Quarto : très grand espace d'états → couverture insuffisante même après 100k épisodes.