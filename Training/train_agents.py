"""
train_agents.py
===============
Ce fichier est le point d'entrée principal pour entraîner les agents de RL.

Il gère 5 agents différents sur 4 environnements :
  - RandomRollout      → PAS d'entraînement, joue par simulations aléatoires
  - MCTS (UCT)         → PAS d'entraînement, explore un arbre de recherche
  - ExpertApprentice   → Apprentissage SUPERVISÉ : imite un expert MCTS
  - MuZero             → Apprentissage par self-play + planification dans un espace latent
  - MuZeroStochastique → Variante de MuZero avec un encodeur VAE pour la stochasticité

Utilisation :
  python train_agents.py --agent all --env all --episodes 10000
  python train_agents.py --agent muzero --env tictactoe --episodes 50000
  python train_agents.py --agent randomrollout --env all
  python train_agents.py --agent mcts --env all
"""

import sys
import os
import argparse
import time
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")   # Pas d'affichage interactif (on sauvegarde les plots en fichiers)
import matplotlib.pyplot as plt

# Permet d'importer les modules depuis la racine du projet
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ── Import des 4 environnements ──────────────────────────────────────────────
from Environnements.line_world   import LineWorld    # Navigation 1D (6 cases)
from Environnements.grid_world   import GridWorld    # Navigation 2D (grille 5×5)
from Environnements.tictactoe    import TicTacToe    # Morpion 3×3 vs Random
from Environnements.quarto       import QuartoEnv    # Quarto 4×4 vs Random

# ── Import des 6 agents ──────────────────────────────────────────────────────
from Agents.Random_Rollout       import RandomRolloutAgent       # Simule N parties aléatoires par action
from Agents.MCTS                 import MCTSAgent                # Arbre UCT (Upper Confidence Trees)
from Agents.Expert_Apprentice    import ExpertApprenticeAgent    # Réseau entraîné par imitation de MCTS
from Agents.MuZero               import MuZeroAgent              # Modèle latent + MCTS dans l'espace caché
from Agents.Muzerostochastic     import MuZeroStochasticAgent    # MuZero + VAE stochastique
from Agents.Alpha_zero           import AlphaZeroAgent           # MCTS-PUCT + réseau de politique/valeur

# ── Import des fonctions d'entraînement AlphaZero (réutilisées par MuZero) ──
# MuZero utilise la même interface d'entraînement qu'AlphaZero (self-play)
from experiment import (
    train_alphazero_1player,    # Boucle d'entraînement pour envs à 1 joueur (LW, GW)
    train_alphazero_tictactoe,  # Boucle d'entraînement pour TicTacToe
    train_alphazero_quarto,     # Boucle d'entraînement pour Quarto
    evaluate_no_training_1player,    # Éval sans entraînement (RR, MCTS) sur LW/GW
    evaluate_no_training_tictactoe,  # Éval sans entraînement sur TicTacToe
    evaluate_no_training_quarto,     # Éval sans entraînement sur Quarto
)

# ── Dossiers de sortie ───────────────────────────────────────────────────────
BASE     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS   = os.path.join(BASE, "models")   # Sauvegarde des poids des réseaux (.pt)
RESULTS  = os.path.join(BASE, "results")  # Métriques JSON
PLOTS    = os.path.join(BASE, "plots")    # Graphiques PNG
for d in [MODELS, RESULTS, PLOTS]:
    os.makedirs(d, exist_ok=True)  # Crée les dossiers s'ils n'existent pas


# ══════════════════════════════════════════════════════════════════════════════
#  BOUCLES D'ENTRAÎNEMENT — ExpertApprentice
#
#  L'ExpertApprentice suit un paradigme d'apprentissage supervisé :
#    1. L'expert (MCTS interne) joue via select_action_expert()
#       → collecte des paires (état, action_choisie) dans un buffer
#    2. agent.learn() entraîne le réseau à imiter ces paires
#       → CrossEntropy entre la sortie du réseau et l'action de l'expert
#
#  Ce paradigme diffère fondamentalement du self-play (AlphaZero/MuZero) :
#  ici c'est de la distillation de connaissance, pas du RL pur.
# ══════════════════════════════════════════════════════════════════════════════

def _parse_loss(loss_result):
    """
    Utilitaire : agent.learn() peut retourner soit un scalaire (policy_loss),
    soit un tuple (policy_loss, value_loss).
    Cette fonction normalise les deux cas.
    """
    if isinstance(loss_result, tuple):
        return loss_result[0], loss_result[1]   # (policy_loss, value_loss)
    return loss_result, None                     # Juste policy_loss


def train_expert_1player(env, agent, num_episodes, checkpoints, evaluate_fn,
                          window=100, max_steps=200):
    """
    Entraîne ExpertApprentice sur LineWorld ou GridWorld.

    Paramètres :
        env          : instance de l'environnement (LineWorld ou GridWorld)
        agent        : instance d'ExpertApprenticeAgent
        num_episodes : nombre total d'épisodes d'entraînement
        checkpoints  : liste d'épisodes où on évalue la politique (ex: [1000, 10000])
        evaluate_fn  : fonction d'évaluation à appeler aux checkpoints
        window       : fenêtre pour le lissage des courbes (non utilisé ici, passé en arg)
        max_steps    : nombre max de pas avant de forcer la fin d'un épisode

    Retourne :
        all_rewards      : liste des rewards par épisode (pour le graphique)
        all_policy_losses: liste des losses de policy par épisode
        []               : liste vide (pas de value loss pour ExpertApprentice)
        eval_results     : dict {épisode: métriques} aux checkpoints
    """
    all_policy_losses, all_rewards, eval_results = [], [], {}
    checkpoints    = sorted(checkpoints)   # Tri croissant pour parcourir dans l'ordre
    next_check_idx = 0                     # Pointeur vers le prochain checkpoint

    for episode in range(1, num_episodes + 1):
        env.reset()
        episode_reward = 0.0
        steps = 0

        # ── Collecte des données par l'expert ─────────────────────────────
        # L'expert (MCTS interne) joue jusqu'à la fin de l'épisode
        # et stocke chaque (état, action) dans son buffer interne
        while not env.done and steps < max_steps:
            available = env.get_actions()
            # select_action_expert = MCTS joue ET stocke la transition dans le buffer
            action    = agent.select_action_expert(env, available)
            _, reward, _ = env.step(action)
            episode_reward += reward
            steps += 1

        # ── Apprentissage supervisé ────────────────────────────────────────
        # Le réseau s'entraîne à imiter l'expert sur les données collectées
        # learn() = passe forward + backward + optimizer.step()
        policy_loss = agent.learn()
        all_policy_losses.append(policy_loss if policy_loss is not None else 0.0)
        all_rewards.append(episode_reward)

        # ── Évaluation aux checkpoints ─────────────────────────────────────
        # On évalue la POLITIQUE APPRISE (le réseau, pas l'expert) pour mesurer
        # à quel point l'apprentissage progresse
        if next_check_idx < len(checkpoints) and episode == checkpoints[next_check_idx]:
            metrics = evaluate_fn(env, agent)
            eval_results[episode] = metrics
            print(f"  {episode:>9,} épisodes | Score : {metrics['score_moyen']:.4f} | "
                  f"Longueur : {metrics['longueur_moy']:.2f} | Temps/coup : {metrics['temps_coup_ms']:.4f} ms")
            next_check_idx += 1

    return all_rewards, all_policy_losses, [], eval_results


def train_expert_tictactoe(env, agent, num_episodes, checkpoints, evaluate_fn, window=100):
    """
    Entraîne ExpertApprentice sur TicTacToe (vs adversaire aléatoire).

    Structure du jeu alterné :
      - Tour 1 : l'expert joue (et collecte la transition)
      - Tour 2 : adversaire joue au hasard via np.random.choice()
      → L'agent apprend uniquement de ses propres coups (pas de l'adversaire)

    Reward final :
      +1 = victoire, -1 = défaite, 0 = match nul
    """
    all_policy_losses, all_rewards, eval_results = [], [], {}
    checkpoints    = sorted(checkpoints)
    next_check_idx = 0

    for episode in range(1, num_episodes + 1):
        env.reset()
        episode_reward = 0.0

        while not env.done:
            available = env.get_actions()

            # Tour de l'expert (joueur 1) : collecte + joue
            action = agent.select_action_expert(env, available)
            _, _, done, info = env.step(action)
            if done:
                # +1 si victoire de l'agent (winner == 1)
                episode_reward += 1 if info["winner"] == 1 else 0
                break

            # Tour de l'adversaire aléatoire (joueur -1)
            # np.random.choice retourne un tableau → on en prend un élément
            _, _, done, info = env.step(np.random.choice(env.get_actions()))
            if done:
                # -1 si l'adversaire gagne (winner == -1)
                episode_reward += -1 if info["winner"] == -1 else 0
                break

        policy_loss = agent.learn()
        all_policy_losses.append(policy_loss if policy_loss is not None else 0.0)
        all_rewards.append(episode_reward)

        if next_check_idx < len(checkpoints) and episode == checkpoints[next_check_idx]:
            metrics = evaluate_fn(env, agent)
            eval_results[episode] = metrics
            print(f"  {episode:>9,} épisodes | Score : {metrics['score_moyen']:.4f} | "
                  f"Longueur : {metrics['longueur_moy']:.2f} | Temps/coup : {metrics['temps_coup_ms']:.4f} ms")
            next_check_idx += 1

    return all_rewards, all_policy_losses, [], eval_results


def train_expert_quarto(env, agent, num_episodes, checkpoints, evaluate_fn, window=100):
    """
    Entraîne ExpertApprentice sur Quarto (vs adversaire aléatoire).

    Quarto a 2 joueurs identifiés par env.current_player (1 ou 2).
    L'agent joue uniquement en tant que joueur 1.
    Le joueur 2 joue aléatoirement.

    Reward final :
      +1 = victoire joueur 1,  -1 = victoire joueur 2,  0 = match nul
    """
    all_policy_losses, all_rewards, eval_results = [], [], {}
    checkpoints    = sorted(checkpoints)
    next_check_idx = 0

    for episode in range(1, num_episodes + 1):
        env.reset()
        episode_reward = 0.0

        while not env.done:
            available = env.get_actions()
            if env.current_player == 1:
                # L'expert (agent) joue et collecte la transition
                action = agent.select_action_expert(env, available)
            else:
                # L'adversaire joue aléatoirement
                action = int(np.random.choice(available))
            _, _, done, _ = env.step(action)
            if done:
                # env.winner : 1 = agent gagne, 2 = adversaire gagne, 0 = nul
                episode_reward = 1 if env.winner == 1 else (-1 if env.winner == 2 else 0)
                break

        policy_loss = agent.learn()
        all_policy_losses.append(policy_loss if policy_loss is not None else 0.0)
        all_rewards.append(episode_reward)

        if next_check_idx < len(checkpoints) and episode == checkpoints[next_check_idx]:
            metrics = evaluate_fn(env, agent)
            eval_results[episode] = metrics
            print(f"  {episode:>9,} épisodes | Score : {metrics['score_moyen']:.4f} | "
                  f"Longueur : {metrics['longueur_moy']:.2f} | Temps/coup : {metrics['temps_coup_ms']:.4f} ms")
            next_check_idx += 1

    return all_rewards, all_policy_losses, [], eval_results


# ══════════════════════════════════════════════════════════════════════════════
#  FONCTIONS D'ÉVALUATION GÉNÉRIQUES
#
#  Ces fonctions mesurent la performance d'un agent avec sa POLITIQUE FINALE
#  (pas pendant l'entraînement). L'agent joue N parties complètes et on
#  calcule le score moyen, la longueur moyenne et le temps de décision.
#
#  IMPORTANT : "select_action" est l'interface commune à TOUS les agents.
#  Pour les agents avec réseau (MuZero, AlphaZero...) cette méthode
#  utilise le réseau seul (sans MCTS) → inférence rapide.
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_agent_1player(env, agent, n_games=500, max_steps=200):
    """
    Évalue n'importe quel agent sur LineWorld ou GridWorld.

    Ces environnements à 1 joueur sont des problèmes de navigation :
    l'agent doit atteindre un objectif en un minimum de pas.

    Métriques retournées :
        score_moyen   : reward moyen (1 = succès, 0 = échec)
        longueur_moy  : nombre moyen de steps par épisode
        temps_coup_ms : temps moyen de décision par action (ms)
    """
    scores, lengths, times = [], [], []

    for _ in range(n_games):
        env.reset()
        steps = 0
        reward = 0

        while not env.done and steps < max_steps:
            available = env.get_actions()

            # Chronomètre la décision de l'agent uniquement (pas le step)
            t0 = time.perf_counter()
            action = agent.select_action(env, available)   # Interface commune
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000)  # Conversion en millisecondes

            _, reward, _ = env.step(action)
            steps += 1

        scores.append(reward)
        lengths.append(steps)

    return {
        "score_moyen":   round(float(np.mean(scores)), 4),
        "longueur_moy":  round(float(np.mean(lengths)), 2),
        "temps_coup_ms": round(float(np.mean(times)), 4),
    }


def evaluate_agent_tictactoe(env, agent, n_games=500):
    """
    Évalue n'importe quel agent sur TicTacToe (vs adversaire aléatoire).

    Structure d'une partie :
      1. Agent joue (select_action) → on chronomètre
      2. Si la partie est finie → on enregistre le résultat
      3. Adversaire joue aléatoirement
      4. Si la partie est finie → on enregistre le résultat
      5. Retour à 1

    Score : +1 victoire, 0 nul, -1 défaite → score_moyen ∈ [-1, 1]
    """
    scores, lengths, times = [], [], []

    for _ in range(n_games):
        env.reset()
        steps = 0

        while not env.done:
            available = env.get_actions()

            # Tour de l'agent (joueur 1)
            t0 = time.perf_counter()
            action = agent.select_action(env, available)
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000)

            _, _, done, info = env.step(action)
            steps += 1
            if done:
                # info["winner"] : 1 = agent, 0 = nul, -1 = adversaire
                scores.append(1 if info["winner"] == 1 else (0 if info["winner"] == 0 else -1))
                lengths.append(steps)
                break

            # Tour de l'adversaire aléatoire (joueur -1)
            _, _, done, info = env.step(np.random.choice(env.get_actions()))
            steps += 1
            if done:
                scores.append(-1 if info["winner"] == -1 else (0 if info["winner"] == 0 else 1))
                lengths.append(steps)
                break

    return {
        "score_moyen":   round(float(np.mean(scores)), 4),
        "longueur_moy":  round(float(np.mean(lengths)), 2),
        "temps_coup_ms": round(float(np.mean(times)), 4),
    }


def evaluate_agent_quarto(env, agent, n_games=500):
    """
    Évalue n'importe quel agent sur Quarto (vs adversaire aléatoire).

    On chronomètre UNIQUEMENT les décisions du joueur 1 (notre agent).
    Le joueur 2 joue aléatoirement sans chronomètre.

    env.winner : 1 = agent gagne, 2 = adversaire gagne, 0 = nul
    """
    scores, lengths, times = [], [], []

    for _ in range(n_games):
        env.reset()
        steps = 0

        while not env.done:
            available = env.get_actions()
            if env.current_player == 1:
                # Tour de l'agent → on chronomètre
                t0 = time.perf_counter()
                action = agent.select_action(env, available)
                t1 = time.perf_counter()
                times.append((t1 - t0) * 1000)
            else:
                # Tour de l'adversaire aléatoire → pas de chrono
                action = int(np.random.choice(available))

            _, _, done, _ = env.step(action)
            steps += 1
            if done:
                scores.append(1 if env.winner == 1 else (-1 if env.winner == 2 else 0))
                lengths.append(steps)
                break

    return {
        "score_moyen":   round(float(np.mean(scores)), 4),
        "longueur_moy":  round(float(np.mean(lengths)), 2),
        "temps_coup_ms": round(float(np.mean(times)), 4),
    }


# ══════════════════════════════════════════════════════════════════════════════
#  CONFIGURATIONS DES ENVIRONNEMENTS ET DES AGENTS
#
#  Ces dictionnaires centralisent tous les paramètres pour éviter la
#  duplication de code. Chaque combinaison agent×env utilise la même
#  logique mais avec des paramètres différents.
# ══════════════════════════════════════════════════════════════════════════════

ENV_CONFIGS = {
    "lineworld": {
        "env_fn":      lambda: LineWorld(size=6),  # 6 cases, 2 actions (gauche/droite)
        "state_size":  8,                          # Taille du vecteur d'état encodé
        "num_actions": 2,                          # Nombre d'actions possibles
        "train_fn":    "1player",                  # Clé pour dispatcher le bon train_fn
        "eval_fn":     evaluate_agent_1player,     # Fonction d'éval à appeler
        "label":       "LineWorld",
    },
    "gridworld": {
        "env_fn":      lambda: GridWorld(rows=5, cols=5),  # Grille 5×5, 4 directions
        "state_size":  31,
        "num_actions": 4,
        "train_fn":    "1player",
        "eval_fn":     evaluate_agent_1player,
        "label":       "GridWorld",
    },
    "tictactoe": {
        "env_fn":      lambda: TicTacToe(),   # 9 cases, 9 actions possibles max
        "state_size":  27,                    # Encodage 3-canaux (joueur1, joueur2, vide)
        "num_actions": 9,
        "train_fn":    "tictactoe",
        "eval_fn":     evaluate_agent_tictactoe,
        "label":       "TicTacToe",
    },
    "quarto": {
        "env_fn":      lambda: QuartoEnv(),   # 16 pièces, 16 cases → 32 actions possibles
        "state_size":  105,                   # Encodage riche (plateau + pièces restantes)
        "num_actions": 32,
        "train_fn":    "quarto",
        "eval_fn":     evaluate_agent_quarto,
        "label":       "Quarto",
    },
}

AGENT_CONFIGS = {
    "randomrollout": {
        "label":       "RandomRollout",
        "no_training": True,   # Cet agent n'a pas de phase d'entraînement
        # Pour N actions légales, simule 50 parties aléatoires par action et choisit la meilleure
        "fn":          lambda s, a: RandomRolloutAgent(n_rollouts=50),
    },
    "mcts": {
        "label":       "MCTS_UCT",
        "no_training": True,
        # Construit un arbre de recherche avec 100 simulations, guidé par UCB1
        "fn":          lambda s, a: MCTSAgent(n_simulations=100),
    },
    "expertapprentice": {
        "label":       "ExpertApprentice",
        "no_training": False,
        "train_mode":  "expert",   # Utilise les boucles train_expert_*
        # Réseau FC (fully connected) de taille 128 → imite un MCTS à 50 simulations
        "fn":          lambda s, a: ExpertApprenticeAgent(
            state_size=s, num_actions=a, n_simulations=50, hidden_size=128, lr=1e-3
        ),
    },
    "muzero": {
        "label":       "MuZero",
        "no_training": False,
        "train_mode":  "alphazero",  # Réutilise l'interface d'entraînement AlphaZero
        # MuZero apprend 3 réseaux : représentation (h), dynamique (g), prédiction (f)
        # n_simulations=10 : nombre de simulations MCTS dans l'espace latent par décision
        "fn":          lambda s, a: MuZeroAgent(
            state_size=s, num_actions=a, hidden_size=128, n_simulations=10, lr=1e-3
        ),
    },
    "muzero_stochastic": {
        "label":       "MuZeroStochastic",
        "no_training": False,
        "train_mode":  "alphazero",
        # Variante de MuZero avec un encodeur variationnel (VAE)
        # chance_size=8 : dimension de l'espace latent stochastique
        # kl_weight=0.1 : poids de la divergence KL dans la loss totale
        "fn":          lambda s, a: MuZeroStochasticAgent(
            state_size=s, num_actions=a, hidden_size=128, chance_size=8,
            n_simulations=10, lr=1e-3, kl_weight=0.1
        ),
    },
    "alphazero": {
        "label":       "AlphaZero",
        "no_training": False,
        "train_mode":  "alphazero",
        # Hyperparamètres spécifiques à chaque environnement
        # c_puct : constante d'exploration dans la formule UCB de MCTS-PUCT
        # Plus c_puct est grand → plus l'agent explore (vs exploiter le réseau)
        "az_configs": {
            "lineworld": {"hidden_size": 64,  "n_simulations": 20, "c_puct": 1.0, "lr": 3e-4},
            "gridworld": {"hidden_size": 128, "n_simulations": 20, "c_puct": 1.0, "lr": 3e-4},
            "tictactoe": {"hidden_size": 128, "n_simulations": 50, "c_puct": 1.5, "lr": 1e-3},
            "quarto":    {"hidden_size": 256, "n_simulations": 30, "c_puct": 1.5, "lr": 1e-3},
        },
        "fn": lambda s, a: AlphaZeroAgent(
            state_size=s, num_actions=a, hidden_size=128, n_simulations=20,
            lr=3e-4, c_puct=1.0,
        ),
    },
}


# ══════════════════════════════════════════════════════════════════════════════
#  HELPERS GRAPHIQUES
# ══════════════════════════════════════════════════════════════════════════════

def smooth(data, w=100):
    """
    Lissage par moyenne glissante.
    Pour chaque point i, calcule la moyenne des w dernières valeurs.
    Permet de visualiser la tendance malgré le bruit des rewards individuels.
    """
    return [np.mean(data[max(0, i - w): i + 1]) for i in range(len(data))]


def save_plot(rewards, losses, eval_results, agent_label, env_label, plots_dir, window=100):
    """
    Génère et sauvegarde le graphique d'entraînement avec 2 sous-graphiques :
      1. Reward lissé (entraînement) + Score éval aux checkpoints (axe secondaire)
      2. Policy Loss lissée

    L'axe secondaire (orange) montre le score de la POLITIQUE GLOUTONNE
    évaluée aux checkpoints → mesure le vrai progrès de l'apprentissage.
    """
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    fig.suptitle(f"{agent_label} — {env_label}", fontsize=13, fontweight="bold")

    # ── Subplot 1 : Reward d'entraînement ────────────────────────────────
    axes[0].plot(smooth(rewards, window), color="#2196F3", linewidth=1.2)
    if eval_results:
        xs = sorted(eval_results.keys())
        ys = [eval_results[x]["score_moyen"] for x in xs]
        # Axe Y secondaire (côté droit) pour le score d'évaluation
        ax2 = axes[0].twinx()
        ax2.plot(xs, ys, "o--", color="#FF5722", markersize=6, label="Score éval.")
        ax2.set_ylabel("Score éval.", color="#FF5722")
        ax2.legend(loc="lower right", fontsize=9)
    axes[0].set_ylabel(f"Reward lissé ({window} ep)")
    axes[0].set_xlabel("Épisode")
    axes[0].set_title("Reward d'entraînement")
    axes[0].grid(True, alpha=0.3)

    # ── Subplot 2 : Policy Loss ───────────────────────────────────────────
    # La loss doit décroître si l'apprentissage progresse
    axes[1].plot(smooth(losses, window), color="#4CAF50", linewidth=1.0)
    axes[1].set_ylabel("Loss")
    axes[1].set_xlabel("Épisode")
    axes[1].set_title("Policy Loss")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(plots_dir, f"{agent_label}_{env_label}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  → Graphique : {path}")


def print_table(eval_results, agent_label, env_label):
    """Affiche un tableau console des métriques aux checkpoints."""
    print()
    print("=" * 70)
    print(f"  {agent_label}  —  {env_label}")
    print("=" * 70)
    print(f"{'Épisodes':>12} | {'Score moyen':>12} | {'Longueur moy':>13} | {'Temps/coup':>12}")
    print("-" * 70)
    for ep, m in sorted(eval_results.items()):
        print(f"{ep:>12,} | {m['score_moyen']:>12.4f} | {m['longueur_moy']:>13.2f} | {m['temps_coup_ms']:>11.4f}ms")
    print()


# ══════════════════════════════════════════════════════════════════════════════
#  DISPATCHER D'ENTRAÎNEMENT
#
#  Associe chaque combinaison (type_env × mode_agent) à la bonne fonction.
#  Exemples :
#    - MuZero sur TicTacToe → TRAIN_FNS["tictactoe"]["alphazero"] = train_alphazero_tictactoe
#    - ExpertApprentice sur LineWorld → TRAIN_FNS["1player"]["expert"] = train_expert_1player
# ══════════════════════════════════════════════════════════════════════════════

TRAIN_FNS = {
    "1player": {
        "expert":    train_expert_1player,    # ExpertApprentice sur LW/GW
        "alphazero": train_alphazero_1player, # MuZero/AlphaZero sur LW/GW
    },
    "tictactoe": {
        "expert":    train_expert_tictactoe,
        "alphazero": train_alphazero_tictactoe,
    },
    "quarto": {
        "expert":    train_expert_quarto,
        "alphazero": train_alphazero_quarto,
    },
}

# Fonctions d'éval pour agents sans entraînement (RandomRollout, MCTS)
# Ces fonctions appellent select_action() directement sans réseau
EVAL_FNS_NO_TRAINING = {
    "1player":   evaluate_no_training_1player,
    "tictactoe": evaluate_no_training_tictactoe,
    "quarto":    evaluate_no_training_quarto,
}


def run_one(agent_key, env_key, num_episodes, checkpoints, n_eval=500):
    """
    Lance l'entraînement et l'évaluation pour 1 agent × 1 environnement.

    Flux complet :
      1. Crée l'agent et l'environnement
      2. Si agent sans entraînement → évalue directement aux checkpoints
         Si agent avec entraînement → lance la boucle d'entraînement
      3. Affiche les métriques et génère les graphiques
      4. Sauvegarde le modèle (.pt) et les résultats (.json)

    Retourne :
        eval_results : dict {épisode: métriques}
    """
    ecfg  = ENV_CONFIGS[env_key]
    acfg  = AGENT_CONFIGS[agent_key]
    label  = acfg["label"]
    elabel = ecfg["label"]

    # Création des dossiers de sortie spécifiques à cet agent
    agent_dir  = os.path.join(MODELS,  label)
    result_dir = os.path.join(RESULTS, label)
    plot_dir   = os.path.join(PLOTS,   label)
    for d in [agent_dir, result_dir, plot_dir]:
        os.makedirs(d, exist_ok=True)

    env = ecfg["env_fn"]()

    # ── Instanciation de l'agent ──────────────────────────────────────────
    # Cas spécial AlphaZero : hyperparamètres différents selon l'environnement
    if agent_key == "alphazero":
        az_cfg = acfg["az_configs"][env_key]
        agent = AlphaZeroAgent(
            state_size    = ecfg["state_size"],
            num_actions   = ecfg["num_actions"],
            hidden_size   = az_cfg["hidden_size"],
            n_simulations = az_cfg["n_simulations"],
            lr            = az_cfg["lr"],
            c_puct        = az_cfg["c_puct"],
        )
    else:
        # Tous les autres agents : appel direct via lambda
        agent = acfg["fn"](ecfg["state_size"], ecfg["num_actions"])

    eval_fn = ecfg["eval_fn"]

    print(f"\n{'═' * 70}")
    print(f"  {label}  —  {elabel}")
    print(f"{'═' * 70}")

    t0 = time.time()

    # ── CAS 1 : Agents sans entraînement (RandomRollout, MCTS) ───────────
    if acfg["no_training"]:
        print("  (Pas d'entraînement — évaluation directe)\n")
        eval_results = {}
        # On évalue aux mêmes checkpoints pour avoir une table comparable
        # mais le résultat est identique à chaque checkpoint (pas d'apprentissage)
        for cp in checkpoints:
            m = eval_fn(env, agent, n_games=n_eval)
            eval_results[cp] = m
            print(f"  {cp:>9,} épisodes | Score : {m['score_moyen']:.4f} | "
                  f"Longueur : {m['longueur_moy']:.2f} | Temps/coup : {m['temps_coup_ms']:.4f} ms")
        rewards = [0.0]
        losses  = [0.0]

    # ── CAS 2 : Agents avec entraînement ─────────────────────────────────
    else:
        train_mode = acfg["train_mode"]   # "expert" ou "alphazero"
        train_fn   = TRAIN_FNS[ecfg["train_fn"]][train_mode]

        # Pendant l'entraînement d'ExpertApprentice → on évalue avec le réseau apprenti
        # Pendant l'entraînement de MuZero/AlphaZero → on évalue avec select_action (réseau)
        if train_mode == "expert":
            train_eval_fn = eval_fn          # eval réseau apprenti
        else:
            train_eval_fn = EVAL_FNS_NO_TRAINING[ecfg["train_fn"]]  # eval réseau

        kwargs = dict(
            env          = env,
            agent        = agent,
            num_episodes = num_episodes,
            checkpoints  = checkpoints,
            evaluate_fn  = train_eval_fn,
        )
        # max_steps uniquement pour les envs 1-joueur (évite les boucles infinies)
        if ecfg["train_fn"] == "1player":
            kwargs["max_steps"] = 200

        # Appel de la boucle d'entraînement → retourne les courbes et métriques
        rewards, losses, _, eval_results = train_fn(**kwargs)

    elapsed = time.time() - t0
    print(f"\n  Durée : {elapsed:.1f}s ({elapsed / 60:.1f} min)")

    print_table(eval_results, label, elabel)

    # Génère le graphique uniquement si on a des données d'entraînement réelles
    if len(rewards) > 1:
        save_plot(rewards, losses, eval_results, label, elabel, plot_dir)

    # ── Sauvegarde du modèle ──────────────────────────────────────────────
    # Pour les agents sans réseau (RR, MCTS), agent.save() sauvegarde juste les params
    model_path = os.path.join(agent_dir, f"{label}_{elabel}.pt")
    agent.save(model_path)
    print(f"  → Modèle : {model_path}")

    # ── Sauvegarde JSON des métriques ─────────────────────────────────────
    data = {
        "agent":       label,
        "env":         elabel,
        "elapsed_s":   round(elapsed, 1),
        "checkpoints": {str(k): v for k, v in eval_results.items()},
    }
    json_path = os.path.join(result_dir, f"{label}_{elabel}.json")
    with open(json_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  → JSON : {json_path}")

    return eval_results


# ══════════════════════════════════════════════════════════════════════════════
#  POINT D'ENTRÉE
# ══════════════════════════════════════════════════════════════════════════════

AGENTS_LIST = ["randomrollout", "mcts", "expertapprentice", "muzero", "muzero_stochastic", "alphazero"]
ENVS_LIST   = ["lineworld", "gridworld", "tictactoe", "quarto"]


def build_checkpoints(num_episodes):
    """
    Génère automatiquement les checkpoints d'évaluation selon le syllabus :
      1k → 10k → 100k → 1M épisodes (si le nombre total le permet)
    Ajoute toujours le dernier épisode comme checkpoint final.
    """
    candidates = [1_000, 10_000, 100_000, 1_000_000]
    cps = [c for c in candidates if c <= num_episodes]
    if not cps or cps[-1] != num_episodes:
        cps.append(num_episodes)
    return sorted(set(cps))


def main():
    parser = argparse.ArgumentParser(description="Entraînement agents RL — 5 agents × 4 environnements")
    parser.add_argument("--agent",    default="all",
                        choices=["all"] + AGENTS_LIST,
                        help="Agent à entraîner (default: all)")
    parser.add_argument("--env",      default="all",
                        choices=["all"] + ENVS_LIST,
                        help="Environnement cible (default: all)")
    parser.add_argument("--episodes", type=int, default=10_000,
                        help="Épisodes d'entraînement pour agents apprenants (default: 10000)")
    parser.add_argument("--n_eval",   type=int, default=500,
                        help="Parties pour chaque évaluation (default: 500)")
    args = parser.parse_args()

    agents = AGENTS_LIST if args.agent == "all" else [args.agent]
    envs   = ENVS_LIST   if args.env   == "all" else [args.env]
    checkpoints = build_checkpoints(args.episodes)

    print(f"\n{'═' * 70}")
    print(f"  ENTRAÎNEMENT — {len(agents)} agents × {len(envs)} environnements")
    print(f"  Épisodes    : {args.episodes:,}")
    print(f"  Checkpoints : {checkpoints}")
    print(f"{'═' * 70}")

    global_t0  = time.time()
    all_results = {}

    # Double boucle : on entraîne chaque agent sur chaque environnement
    for agent_key in agents:
        all_results[agent_key] = {}
        for env_key in envs:
            result = run_one(agent_key, env_key, args.episodes, checkpoints, args.n_eval)
            all_results[agent_key][env_key] = result

    # ── Résumé global en console ──────────────────────────────────────────
    # Tableau (agent × env) avec le score au dernier checkpoint
    last_cp = max(checkpoints)
    print(f"\n{'═' * 80}")
    print(f"  RÉSUMÉ GLOBAL (checkpoint {last_cp:,} épisodes)")
    print(f"{'═' * 80}")
    header = f"{'Agent':>22}" + "".join(f" | {e[:12]:>12}" for e in envs)
    print(header)
    print("─" * len(header))
    for agent_key in agents:
        row = f"{AGENT_CONFIGS[agent_key]['label']:>22}"
        for env_key in envs:
            res = all_results[agent_key].get(env_key, {})
            m   = res.get(last_cp) or (list(res.values())[-1] if res else None)
            row += f" | {m['score_moyen']:>12.4f}" if m else " | {'N/A':>12}"
        print(row)
    print()

    total = time.time() - global_t0
    print(f"  Durée totale : {total:.0f}s ({total / 60:.1f} min)")


if __name__ == "__main__":
    main()