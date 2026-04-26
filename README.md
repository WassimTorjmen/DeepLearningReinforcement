# Deep Reinforcement Learning — Projet ESGI 5A

Implémentations et benchmarks d'algorithmes de RL classiques et profonds (REINFORCE, REINFORCE+Critic, REINFORCE+Mean Baseline, PPO/A2C, DQN, DDQN, DDQN+ER, DDQN+PER, MCTS, AlphaZero, MuZero, MuZero stochastique, Expert/Apprentice, Tabular Q) sur quatre environnements (LineWorld, GridWorld, TicTacToe, Quarto), avec démos Pygame.

## TL;DR — démarrage en 30 secondes

```bash
pip install torch numpy pygame matplotlib
python Environnements/quarto_gui.py        # joue contre l'agent PPO_A2C
```

## Structure du projet

```
rl_projet/
├── Agents/                # Implémentations des agents (1 fichier par algo)
├── Environnements/        # 4 envs + GUIs Pygame (lineworld_gui, gridworld_gui, tictactoe_gui, quarto_gui)
├── Encodings/             # Encodages d'états spécifiques par env
├── Training/              # Boucles d'entraînement réutilisables
├── Evaluation/            # Fonctions d'évaluation
├── Experimentations/      # Points d'entrée (main.py + experiment_*.py spécialisés)
│   ├── models/<algo>/     # Poids sauvegardés (.pt / .pkl) — généré
│   ├── plots/<algo>/      # Courbes d'entraînement (.png) — généré
│   └── results/<algo>/    # Rapports JSON / Markdown détaillés — généré
├── Benchmarks/            # benchmark_quarto.py (parties algo vs algo)
└── Tests/                 # Tests unitaires
```

## Prérequis

- **Python** 3.9+
- **PyTorch** (CPU ou CUDA — détecté automatiquement)
- **NumPy**
- **Pygame** (démos GUI)
- **Matplotlib** (courbes d'entraînement)

```bash
pip install torch numpy pygame matplotlib
```

> Sur Windows, lancez les commandes depuis un terminal ouvert dans `rl_projet/`. Les chemins relatifs des modèles sont calculés automatiquement.

## Démos GUI

Toutes les GUIs détectent leur `.pt` sous `Experimentations/models/<algo>/`. En l'absence de poids, l'agent joue avec une politique aléatoire (et le message `✗ Modèle non trouvé` s'affiche).

### Quarto

```bash
python Environnements/quarto_gui.py
```

Modèle : [Experimentations/models/ppo_a2c/PPO_A2C_Quarto.pt](Experimentations/models/ppo_a2c/PPO_A2C_Quarto.pt) (PPO/A2C, state=105, actions=32, hidden=128).

Trois modes au démarrage :

| Mode | Description |
|------|-------------|
| 1 | **Humain (J1) vs Agent RL (J2)** — vous jouez contre PPO_A2C |
| 2 | **Agent RL (J1) vs Random (J2)** — démo automatique |
| 3 | **Humain (J1) vs Random (J2)** — vous jouez contre un random |

Contrôles :
- Clic gauche sur une pièce dispo : phase *choose* (donner la pièce à l'adversaire)
- Clic gauche sur une case du plateau : phase *place* (placer la pièce courante)
- `R` : recommencer la partie · `M` : retour au menu

### TicTacToe

```bash
python Environnements/tictactoe_gui.py
```

Modèle : [Experimentations/models/reinforce_critic/REINFORCE_Critic_TicTacToe.pt](Experimentations/models/reinforce_critic/REINFORCE_Critic_TicTacToe.pt).

Deux modes :
1. Humain (X) vs Agent RL (O)
2. Agent RL vs Random (démo)

Contrôles : clic gauche pour placer · `R` : recommencer · `M` : menu.

### GridWorld

```bash
python Environnements/gridworld_gui.py
```

Démo manuelle (humain). Contrôles : flèches `↑ ↓ ← →` pour bouger, `R` pour reset.

### LineWorld

```bash
python Environnements/lineworld_gui.py
```

Démo manuelle (humain). Contrôles : flèches `← →`.

## Entraînement

### Point d'entrée principal — `main.py`

```bash
python Experimentations/main.py
```

[Experimentations/main.py](Experimentations/main.py) contient des blocs `run_experiment(...)` à commenter/décommenter pour choisir l'algo et l'environnement. Chaque exécution :

1. Entraîne pour `num_episodes` épisodes,
2. Évalue le checkpoint sur 500 parties à chaque entrée de `checkpoints=[...]`,
3. Sauvegarde les poids dans `Experimentations/models/<algo>/<Agent>_<Env>.pt`,
4. Trace la courbe `Experimentations/plots/<algo>/<Agent>_<Env>.png` (reward + policy_loss + critic_loss).

#### Sortie console attendue

```
=================================================================
  PPO_A2C  —  Quarto
=================================================================
================================================================
  ENTRAÎNEMENT  (moyennes glissantes sur les 100 derniers épisodes)
================================================================
    Épisodes |   Reward moy |   Policy loss |   Critic loss
----------------------------------------------------------------
       1,000 |       0.0500 |       -0.6431 |        0.4820
      10,000 |       0.4200 |       -0.1520 |        0.1240
     100,000 |       0.7800 |       -0.0210 |        0.0450

=================================================================
  RÉSULTATS  (policy évaluée sur 500 parties, sans exploration)
=================================================================
    Épisodes |  Score moyen |  Longueur moy |   Temps/coup
-----------------------------------------------------------------
       1,000 |       0.0780 |         18.20 |      0.4023ms
      10,000 |       0.3800 |         16.40 |      0.4123ms
     100,000 |       0.7500 |         14.80 |      0.4023ms
Graphique sauvegardé → .../Experimentations/plots/ppo_a2c/PPO_A2C_Quarto.png
Modèle sauvegardé   → .../Experimentations/models/ppo_a2c/PPO_A2C_Quarto.pt
```

> **Note sur la reproductibilité :** aucune seed n'est fixée volontairement (`np.random` / `torch.manual_seed`). Deux exécutions consécutives produisent des courbes différentes. Pour rendre les expériences reproductibles, ajouter `np.random.seed(42)` et `torch.manual_seed(42)` au début de `run_experiment`.

#### Reproduire le modèle Quarto fourni

Bloc PPO/A2C Quarto (déjà actif dans `main.py`) :

```python
run_experiment(
    env=QuartoEnv(), env_name="Quarto",
    train_fn=train_quarto, evaluate_fn=evaluate_quarto,
    agent=PPO_A2C.PPOAgent(105, 32, hidden_size=128),
    agent_name="PPO_A2C",
    num_episodes=100_000, checkpoints=[1_000, 10_000, 100_000],
)
```

#### Tailles d'entrée par environnement

| Env | `state_size` | `num_actions` |
|-----|--------------|---------------|
| LineWorld   | 8   | 2  |
| GridWorld   | 31  | 4  |
| TicTacToe   | 27  | 9  |
| Quarto      | 105 | 32 |

### Points d'entrée spécialisés

Pour les algos plus lourds (MCTS-based / model-based / tabulaires), des scripts dédiés gèrent leurs propres dossiers `models/<algo>/`, `plots/<algo>/` et `results/<algo>/` (rapports JSON et Markdown détaillés) :

| Script | Ce qu'il entraîne |
|--------|-------------------|
| [Experimentations/experiment_alphazero.py](Experimentations/experiment_alphazero.py) | AlphaZero sur les 4 envs |
| [Experimentations/experiment_muzero.py](Experimentations/experiment_muzero.py) | MuZero (déterministe + stochastique) |
| [Experimentations/experiment_agents.py](Experimentations/experiment_agents.py) | Suite complète d'agents (REINFORCE, PPO, DDQN, ExpertApprentice…) avec rapport agrégé |
| [Experimentations/experiment_tabular_q.py](Experimentations/experiment_tabular_q.py) | Tabular Q-learning sur LineWorld/GridWorld |

Chaque script est lancé directement :

```bash
python Experimentations/experiment_alphazero.py
```

## Modèles pré-entraînés fournis

```
Experimentations/models/
├── ppo_a2c/
│   ├── PPO_A2C_LineWorld.pt
│   ├── PPO_A2C_GridWorld.pt
│   ├── PPO_A2C_TicTacToe.pt
│   └── PPO_A2C_Quarto.pt
├── reinforce_critic/
│   └── REINFORCE_Critic_TicTacToe.pt
├── ddqn_er/
│   └── DDQN_ER_LineWorld.pt
├── ddqn_per/                    # plot uniquement (DDQN_PER_TicTacToe.png)
├── alphazero/                   # AlphaZero_{LineWorld,GridWorld,TicTacToe,Quarto}.pt
├── MuZero/                      # MuZero_*.pt
├── MuZeroStochastic/            # MuZeroStochastic_*.pt
├── ExpertApprentice/            # ExpertApprentice_*.pt
└── TabularQ/                    # *.pkl (Q-tables tabulaires)
```

> ⚠️ **Convention de casse hétérogène** : les dossiers générés par `run_experiment(...)` (helper récent) sont en `lowercase`. Les dossiers historiques (AlphaZero, MuZero…) restent en `PascalCase`. Les deux cohabitent sans conflit.

## Benchmarks

```bash
python Benchmarks/benchmark_quarto.py
```

Joue N parties entre deux agents (au choix) et affiche le taux de victoire. Utile pour comparer un agent entraîné à un random ou à un autre algo.

## Architecture — comment les agents communiquent avec les envs

- **Envs** : exposent `reset()`, `step(action)`, `get_actions()`, `encode_state()`, `done`, `winner`, `current_player`.
- **Agents PG (REINFORCE / PPO_A2C / Reinforce_Critic / Reinforce_Mean_Baseline)** : `select_action(state, available)`, `store_reward(r)`, `learn() → (policy_loss, critic_loss?)`.
- **Agents Q (DQN / DDQN / DDQN+ER / DDQN+PER)** : `select_action(state, available)` (ε-greedy), `store(s,a,r,s',done)` ou apprentissage en ligne dans le trainer.
- **Agents model-based / search-based (MCTS / AlphaZero / MuZero / Expert/Apprentice / Random_Rollout / RandomAgent / TabularQ)** : `select_action(env, available)` ou `select_action_mcts(env, available)` — ils ont besoin de l'env pour simuler des rollouts.

L'agent random expose `select_action(env)` ; tous les autres exposent `select_action(state, available)` (PG/Q) ou `select_action(env, available)` (recherche).

## Notes

- Les masques d'actions sont créés sur le **device** de l'agent (`agent.device`) → le code tourne indifféremment sur CPU ou CUDA.
- Si un `.pt` est manquant, le GUI charge l'agent avec ses poids initialisés aléatoirement et l'indique en console (`✗ Modèle non trouvé`).
- L'arborescence `Experimentations/{models,plots,results}/` est créée à la volée si elle n'existe pas.
