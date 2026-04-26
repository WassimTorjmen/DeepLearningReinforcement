# Deep Reinforcement Learning — Projet ESGI 5A

Implémentations et benchmarks d'algorithmes de RL classiques et profonds sur plusieurs environnements (LineWorld, GridWorld, TicTacToe, Quarto).

## Structure du projet

```
rl_projet/
├── Agents/              # Implémentations des agents (REINFORCE, PPO_A2C, DDQN, MCTS, MuZero, AlphaZero, ...)
├── Environnements/      # Environnements + GUIs Pygame (line_world, grid_world, tictactoe, quarto)
├── Encodings/           # Encodages d'états pour les environnements
├── Training/            # Boucles d'entraînement
├── Evaluation/          # Fonctions d'évaluation
├── Experimentations/    # Points d'entrée pour lancer les expériences
├── Benchmarks/          # Scripts de benchmarking
├── Tests/               # Tests
└── *.pt                 # Modèles entraînés sauvegardés
```

## Prérequis

- Python 3.10+
- PyTorch
- NumPy
- Pygame (pour les démos GUI)
- Matplotlib (pour les courbes)
- tqdm

Installation :

```bash
pip install torch numpy pygame matplotlib tqdm
```

## Lancement des démos GUI

Les GUIs se lancent depuis la racine du projet (le dossier `rl_projet/`) afin que les chemins des modèles `.pt` soient résolus correctement.

### Quarto — Humain vs Agent RL / Agent RL vs Random / Humain vs Random

```bash
python Environnements/quarto_gui.py
```

Modèle utilisé : `PPO_A2C_Quarto.pt` (PPO A2C, state_size=105, num_actions=32, hidden_size=128).

Trois modes au démarrage :
1. **Humain (J1) vs Agent RL (J2)** — vous jouez contre PPO_A2C
2. **Agent RL (J1) vs Random (J2)** — démo automatique
3. **Humain (J1) vs Random (J2)** — vous jouez contre un agent aléatoire

Contrôles :
- Clic gauche sur une pièce dispo : choisir la pièce à donner à l'adversaire (phase *choose*)
- Clic gauche sur une case du plateau : placer la pièce courante (phase *place*)
- Touche `R` : recommencer la partie
- Touche `M` : retour au menu

### TicTacToe — Humain vs Agent RL / Agent RL vs Random

```bash
python Environnements/tictactoe_gui.py
```

Modèle utilisé : `REINFORCE_Critic_TicTacToe.pt`.

### GridWorld

```bash
python Environnements/gridworld_gui.py
```

### LineWorld

```bash
python Environnements/lineworld_gui.py
```

## Reproduction des résultats (entraînement)

Le point d'entrée principal est [Experimentations/main.py](Experimentations/main.py). Les blocs d'expériences sont commentés/décommentés à la main pour choisir l'algo et l'environnement à entraîner.

```bash
python Experimentations/main.py
```

À chaque exécution, `run_experiment(...)` :
- entraîne l'agent pendant `num_episodes`,
- sauvegarde des checkpoints `.pt` aux étapes `checkpoints=[...]`,
- évalue l'agent et trace la courbe de récompense (`{Agent}_{Env}.png`).

### Reproduire le modèle Quarto (PPO A2C)

Dans [Experimentations/main.py](Experimentations/main.py), garder décommenté le bloc :

```python
run_experiment(
    env=QuartoEnv(), env_name="Quarto",
    train_fn=train_quarto, evaluate_fn=evaluate_quarto,
    agent=PPO_A2C.PPOAgent(105, 32, hidden_size=128),
    agent_name="PPO_A2C",
    num_episodes=100_000, checkpoints=[1_000, 10_000, 100_000],
)
```

Puis :

```bash
python Experimentations/main.py
```

Le fichier `PPO_A2C_Quarto.pt` est régénéré à la racine du projet et la courbe `PPO_A2C_Quarto.png` est sauvegardée.

### Autres environnements / agents

Pour entraîner sur LineWorld / GridWorld / TicTacToe avec PPO_A2C, REINFORCE, DDQN, MuZero, AlphaZero, etc., décommenter le bloc `run_experiment(...)` correspondant dans [Experimentations/main.py](Experimentations/main.py).

Tailles d'entrée à respecter :
- LineWorld : `state_size=8, num_actions=2`
- GridWorld : `state_size=31, num_actions=4`
- TicTacToe : `state_size=27, num_actions=9`
- Quarto    : `state_size=105, num_actions=32`

## Modèles pré-entraînés fournis

Les `.pt` sont rangés sous `models/<algo_lowercase>/` à la racine du projet :

| Fichier                                              | Algo             | Environnement |
|------------------------------------------------------|------------------|---------------|
| `models/ppo_a2c/PPO_A2C_LineWorld.pt`                | PPO A2C          | LineWorld     |
| `models/ppo_a2c/PPO_A2C_GridWorld.pt`                | PPO A2C          | GridWorld     |
| `models/ppo_a2c/PPO_A2C_TicTacToe.pt`                | PPO A2C          | TicTacToe     |
| `models/ppo_a2c/PPO_A2C_Quarto.pt`                   | PPO A2C          | Quarto        |
| `models/reinforce_critic/REINFORCE_Critic_TicTacToe.pt` | REINFORCE+Critic | TicTacToe |
| `models/ddqn_er/DDQN_ER_LineWorld.pt`                | DDQN + ER        | LineWorld     |
| `models/alphazero/`, `models/MuZero/`, `models/MuZeroStochastic/`, `models/ExpertApprentice/`, `models/TabularQ/` | (pré-existants) | — |

## Notes

- Les GUIs détectent automatiquement le `.pt` à la racine ; un message `✓ Modèle chargé` ou `✗ Modèle non trouvé` est affiché en console au démarrage.
- En cas de modèle absent, l'agent jouera avec ses poids initialisés aléatoirement.
- Sous Windows, lancer les commandes depuis un terminal ouvert dans `rl_projet/`.
