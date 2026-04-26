# Rapport d'expérimentation — Tous agents

> Épisodes d'entraînement : 5,000


## Agents étudiés

| Agent | Type | Hyperparamètres principaux |
|---|---|---|
| AlphaZero | Alphazero | hidden_size=256, n_simulations=30, c_puct=1.5, lr=0.001 |

## Score moyen par agent × environnement (dernier checkpoint)

| Agent | LineWorld | GridWorld (5×5) | TicTacToe (vs Random) | Quarto (vs Random) |
|---|---|---|---|---|
| AlphaZero | N/A | N/A | N/A | 0.2160 |

## Temps moyen par coup (ms) — dernier checkpoint

| Agent | LineWorld | GridWorld (5×5) | TicTacToe (vs Random) | Quarto (vs Random) |
|---|---|---|---|---|
| AlphaZero | N/A | N/A | N/A | 1.1443 |

## Résultats détaillés par environnement

### LineWorld

| Agent | Épisodes | Score moyen | Longueur moy | Temps/coup (ms) |
|---|---|---|---|---|
### GridWorld (5×5)

| Agent | Épisodes | Score moyen | Longueur moy | Temps/coup (ms) |
|---|---|---|---|---|
### TicTacToe (vs Random)

| Agent | Épisodes | Score moyen | Longueur moy | Temps/coup (ms) |
|---|---|---|---|---|
### Quarto (vs Random)

| Agent | Épisodes | Score moyen | Longueur moy | Temps/coup (ms) |
|---|---|---|---|---|
| AlphaZero | 1,000 | 0.3020 | 21.78 | 0.4872 |
| AlphaZero | 5,000 | 0.2160 | 22.45 | 1.1443 |


## AlphaZero — Réseau seul vs MCTS

| Environnement | Mode | Score | Victoires | Nuls | Défaites | Longueur | ms/coup |
|---|---|---|---|---|---|---|---|
| Quarto (vs Random) | Réseau | 0.2400 | 61.2% | 1.6% | 37.2% | 21.70 | 1.4505 |
| Quarto (vs Random) | MCTS | 0.3550 | 66.5% | 2.5% | 31.0% | 22.14 | 35.3837 |

## Observations et interprétations

*(À compléter par votre analyse des résultats)*


### RandomRollout
- Agent de référence sans apprentissage. Simule N parties aléatoires par action et choisit la meilleure.
- Performances stables mais coûteux en temps (O(N × profondeur) par coup).
- Bonne baseline pour évaluer l'apport de l'apprentissage.

### MCTS UCT
- Exploration guidée par UCB1. Plus efficace que RandomRollout à iso-simulations.
- Résultats indépendants de l'entraînement : même score à 1k, 10k, 100k épisodes.

### ExpertApprentice
- Distille la politique MCTS dans un réseau neuronal. Phase d'apprentissage supervisé.
- Inférence rapide (réseau seul), mais qualité bornée par l'expert (MCTS à n_simulations fixé).

### MuZero
- Apprend un modèle latent de l'environnement. Planification dans l'espace latent.
- Plus expressif qu'AlphaZero (pas besoin du modèle d'env fourni), mais plus complexe à entraîner.

### MuZeroStochastique
- Étend MuZero avec un encodeur VAE pour modéliser la stochasticité de l'environnement.
- Utile quand l'env est partiellement observable. Sur nos envs déterministes, avantage marginal.

### AlphaZero
- Combine MCTS-PUCT guidé par réseau et auto-jeu. Convergence stable sur les jeux combinatoires.
- Mode réseau seul (inférence rapide) vs MCTS complet (plus fort, plus lent).
- Sur TicTacToe et Quarto, le gain MCTS est significatif après suffisamment d'épisodes.