"""
experiment_agents.py
====================
Expérimentations complètes pour tous les agents sur les 4 environnements.

Ce fichier est le plus complet du projet. Il orchestre :
  1. L'entraînement de chaque agent avec checkpoints au syllabus (1k/10k/100k épisodes)
  2. L'évaluation détaillée à chaque checkpoint
  3. La génération de graphiques (training curves, progression, comparaison)
  4. Pour AlphaZero uniquement : comparaison réseau seul vs MCTS pendant l'éval
  5. L'export de tout (JSON par agent, rapport Markdown global)

Agents couverts :
  - RandomRollout           → éval directe (pas d'entraînement)
  - MCTS UCT                → éval directe (pas d'entraînement)
  - ExpertApprentice        → apprentissage supervisé (imite MCTS)
  - MuZero                  → self-play + planification latente
  - MuZeroStochastique      → MuZero avec espace latent stochastique (VAE)
  - AlphaZero               → MCTS-PUCT guidé par réseau + self-play

Utilisation :
  python experiment_agents.py                                       # tout, 10k épisodes
  python experiment_agents.py --episodes 100000                     # 100k épisodes
  python experiment_agents.py --agent alphazero --env tictactoe     # ciblé
  python experiment_agents.py --skip_no_training                    # saute RR et MCTS
"""

import sys
import os
import argparse
import time
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")   # Rendu sans fenêtre graphique (sauvegarde en fichier)
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ── Import des environnements ─────────────────────────────────────────────────
from Environnements.line_world   import LineWorld
from Environnements.grid_world   import GridWorld
from Environnements.tictactoe    import TicTacToe
from Environnements.quarto       import QuartoEnv

# ── Import des agents ─────────────────────────────────────────────────────────
from Agents.Random_Rollout       import RandomRolloutAgent
from Agents.MCTS                 import MCTSAgent
from Agents.Expert_Apprentice    import ExpertApprenticeAgent
from Agents.MuZero               import MuZeroAgent
from Agents.Muzerostochastic     import MuZeroStochasticAgent
from Agents.Alpha_zero           import AlphaZeroAgent

# ── Import des boucles d'entraînement AlphaZero ───────────────────────────────
# MuZero, MuZeroStochastic et AlphaZero partagent la MÊME interface d'entraînement
# (self-play avec MCTS + mise à jour par loss combinée policy+value)
from experiment import (
    train_alphazero_1player,          # Entraîne sur LineWorld / GridWorld
    train_alphazero_tictactoe,        # Entraîne sur TicTacToe
    train_alphazero_quarto,           # Entraîne sur Quarto
    evaluate_no_training_1player,     # Éval rapide pour agents sans réseau (LW, GW)
    evaluate_no_training_tictactoe,   # Éval rapide pour agents sans réseau (TTT)
    evaluate_no_training_quarto,      # Éval rapide pour agents sans réseau (Quarto)
)

# ── Import des fonctions d'éval détaillée d'AlphaZero ────────────────────────
# Ces fonctions supportent use_mcts=True/False pour comparer les deux modes
from Evaluation.evaluate_alphazero import (
    evaluate_lineworld   as az_eval_lineworld,
    evaluate_gridworld   as az_eval_gridworld,
    evaluate_tictactoe   as az_eval_tictactoe,
    evaluate_quarto      as az_eval_quarto,
)

# ── Import des boucles ExpertApprentice et fonctions d'éval génériques ────────
from Training.train_agents import (
    train_expert_1player,         # Entraîne ExpertApprentice sur LW/GW
    train_expert_tictactoe,       # Entraîne ExpertApprentice sur TicTacToe
    train_expert_quarto,          # Entraîne ExpertApprentice sur Quarto
    evaluate_agent_1player,       # Éval générique (réseau seul) sur LW/GW
    evaluate_agent_tictactoe,     # Éval générique sur TicTacToe
    evaluate_agent_quarto,        # Éval générique sur Quarto
)

# ── Dossiers de sortie ────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR  = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results", "experiment_agents")  # JSONs
PLOTS_DIR   = os.path.join(BASE_DIR, "plots",   "experiment_agents")  # PNGs
for d in [MODELS_DIR, RESULTS_DIR, PLOTS_DIR]:
    os.makedirs(d, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
#  CONFIGURATIONS
# ══════════════════════════════════════════════════════════════════════════════

ENV_CONFIGS = {
    "lineworld": {
        "label":       "LineWorld",
        "state_size":  8,           # Vecteur d'état : 8 valeurs (position + goal encodés)
        "num_actions": 2,           # Gauche ou droite
        "env_fn":      lambda: LineWorld(size=6),
        "train_type":  "1player",   # Clé pour le dispatcher TRAIN_FNS
        "eval_fn":     evaluate_agent_1player,         # Éval avec réseau
        "eval_fn_no_training": evaluate_no_training_1player,  # Éval sans réseau
        "max_steps":   200,         # Limite de pas par épisode (évite les boucles)
    },
    "gridworld": {
        "label":       "GridWorld (5×5)",
        "state_size":  31,          # Encodage one-hot : position (25) + goal (4) + traps (2)
        "num_actions": 4,           # Haut, bas, gauche, droite
        "env_fn":      lambda: GridWorld(rows=5, cols=5),
        "train_type":  "1player",
        "eval_fn":     evaluate_agent_1player,
        "eval_fn_no_training": evaluate_no_training_1player,
        "max_steps":   200,
    },
    "tictactoe": {
        "label":       "TicTacToe (vs Random)",
        "state_size":  27,          # 3 canaux × 9 cases : (joueur1, joueur2, vide)
        "num_actions": 9,           # 9 cases possibles
        "env_fn":      lambda: TicTacToe(),
        "train_type":  "tictactoe",
        "eval_fn":     evaluate_agent_tictactoe,
        "eval_fn_no_training": evaluate_no_training_tictactoe,
        "max_steps":   None,        # La partie s'arrête naturellement (9 coups max)
    },
    "quarto": {
        "label":       "Quarto (vs Random)",
        "state_size":  105,         # Encodage riche : plateau (64) + pièces (16×attributes) + ...
        "num_actions": 32,          # 16 pièces × 16 cases → 32 actions combinées (choisir pièce + placer)
        "env_fn":      lambda: QuartoEnv(),
        "train_type":  "quarto",
        "eval_fn":     evaluate_agent_quarto,
        "eval_fn_no_training": evaluate_no_training_quarto,
        "max_steps":   None,
    },
}

AGENT_CONFIGS = {
    "randomrollout": {
        "label":        "RandomRollout",
        "no_training":  True,     # Pas de boucle d'apprentissage
        "train_mode":   None,
        # Pour chaque action légale, simule 50 parties aléatoires jusqu'au bout
        # et choisit l'action avec le meilleur score moyen simulé
        "fn": lambda s, a, env_key: RandomRolloutAgent(n_rollouts=50),
        "hyperparams": {"n_rollouts": 50},
    },
    "mcts": {
        "label":        "MCTS_UCT",
        "no_training":  True,
        "train_mode":   None,
        # UCT = Upper Confidence Trees : exploration guidée par UCB1
        # 100 simulations = 100 nœuds explorés dans l'arbre par décision
        "fn": lambda s, a, env_key: MCTSAgent(n_simulations=100),
        "hyperparams": {"n_simulations": 100},
    },
    "expertapprentice": {
        "label":        "ExpertApprentice",
        "no_training":  False,
        "train_mode":   "expert",    # Utilise les boucles train_expert_*
        # Réseau FC 128 unités cachées
        # n_simulations=50 : taille de l'expert MCTS qui génère les données d'imitation
        "fn": lambda s, a, env_key: ExpertApprenticeAgent(
            state_size=s, num_actions=a, n_simulations=50, hidden_size=128, lr=1e-3
        ),
        "hyperparams": {"hidden_size": 128, "lr": 1e-3, "n_simulations": 50},
    },
    "muzero": {
        "label":        "MuZero",
        "no_training":  False,
        "train_mode":   "alphazero",   # Réutilise les boucles train_alphazero_*
        # MuZero apprend 3 fonctions :
        #   h() : représentation (observation → état latent)
        #   g() : dynamique (état latent × action → état latent suivant + reward)
        #   f() : prédiction (état latent → politique + valeur)
        # n_simulations=3 : MCTS dans l'espace latent (peu de simulations, plus rapide)
        "fn": lambda s, a, env_key: MuZeroAgent(
            state_size=s, num_actions=a, hidden_size=128, n_simulations=3, lr=1e-3
        ),
        "hyperparams": {"hidden_size": 128, "lr": 1e-3, "n_simulations": 3},
    },
    "muzero_stochastic": {
        "label":        "MuZeroStochastic",
        "no_training":  False,
        "train_mode":   "alphazero",
        # Variante stochastique de MuZero :
        # La fonction dynamique g() intègre un encodeur VAE pour modéliser
        # l'incertitude de l'environnement.
        # chance_size=8 : dimension de l'espace latent stochastique (variables cachées)
        # kl_weight=0.1 : poids de la divergence KL dans la loss (régularisation VAE)
        "fn": lambda s, a, env_key: MuZeroStochasticAgent(
            state_size=s, num_actions=a, hidden_size=128, chance_size=8,
            n_simulations=10, lr=1e-3, kl_weight=0.1
        ),
        "hyperparams": {"hidden_size": 128, "lr": 1e-3, "n_simulations": 10, "chance_size": 8},
    },
    "alphazero": {
        "label":        "AlphaZero",
        "no_training":  False,
        "train_mode":   "alphazero",
        # Hyperparamètres DIFFÉRENTS selon l'environnement :
        # - hidden_size : capacité du réseau (plus grand pour envs complexes)
        # - n_simulations : nombre de simulations MCTS-PUCT par coup (qualité vs vitesse)
        # - c_puct : constante d'exploration dans MCTS-PUCT
        #   formule : UCB = Q(s,a) + c_puct × P(s,a) × sqrt(N(s)) / (1 + N(s,a))
        #   c_puct élevé → plus d'exploration vers les actions peu visitées
        # - lr : taux d'apprentissage
        "az_configs": {
            "lineworld": {"hidden_size": 64,  "n_simulations": 20, "c_puct": 1.0, "lr": 3e-4},
            "gridworld": {"hidden_size": 128, "n_simulations": 20, "c_puct": 1.0, "lr": 3e-4},
            "tictactoe": {"hidden_size": 128, "n_simulations": 50, "c_puct": 1.5, "lr": 1e-3},
            "quarto":    {"hidden_size": 256, "n_simulations": 30, "c_puct": 1.5, "lr": 1e-3},
        },
        "fn": lambda s, a, env_key: None,   # Surchargé dans build_agent()
        "hyperparams": {},                   # Surchargé dans build_agent()
        # Fonctions d'éval dédiées AlphaZero (supportent use_mcts=True)
        "detail_eval_fns": {
            "lineworld": az_eval_lineworld,
            "gridworld": az_eval_gridworld,
            "tictactoe": az_eval_tictactoe,
            "quarto":    az_eval_quarto,
        },
    },
}

# ── Dispatcher des boucles d'entraînement ─────────────────────────────────────
# Associe (type_env, mode_agent) → fonction d'entraînement
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

AGENTS_LIST = ["randomrollout", "mcts", "expertapprentice", "muzero", "muzero_stochastic", "alphazero"]
ENVS_LIST   = ["lineworld", "gridworld", "tictactoe", "quarto"]


# ══════════════════════════════════════════════════════════════════════════════
#  CONSTRUCTION DE L'AGENT
# ══════════════════════════════════════════════════════════════════════════════

def build_agent(agent_key, env_key):
    """
    Instancie l'agent avec les hyperparamètres adaptés à l'environnement.

    Retourne :
        agent       : instance de l'agent prête à l'emploi
        hyperparams : dict des hyperparamètres utilisés (pour le rapport JSON)
    """
    acfg = AGENT_CONFIGS[agent_key]
    ecfg = ENV_CONFIGS[env_key]

    if agent_key == "alphazero":
        # AlphaZero : on récupère les hyperparamètres spécifiques à cet env
        az_cfg = acfg["az_configs"][env_key]
        agent  = AlphaZeroAgent(
            state_size    = ecfg["state_size"],
            num_actions   = ecfg["num_actions"],
            hidden_size   = az_cfg["hidden_size"],
            n_simulations = az_cfg["n_simulations"],
            lr            = az_cfg["lr"],
            c_puct        = az_cfg["c_puct"],
        )
        # Enrichit hyperparams avec state_size et num_actions pour le rapport
        hyperparams = az_cfg.copy()
        hyperparams.update({"state_size": ecfg["state_size"], "num_actions": ecfg["num_actions"]})
    else:
        # Tous les autres : appel via lambda (signature : state_size, num_actions, env_key)
        agent       = acfg["fn"](ecfg["state_size"], ecfg["num_actions"], env_key)
        hyperparams = acfg["hyperparams"].copy()

    return agent, hyperparams


# ══════════════════════════════════════════════════════════════════════════════
#  HELPERS GRAPHIQUES
# ══════════════════════════════════════════════════════════════════════════════

def smooth(data, w=100):
    """
    Lissage par moyenne mobile sur une fenêtre de w épisodes.
    Réduit le bruit des rewards individuels pour visualiser la tendance.
    """
    return [np.mean(data[max(0, i - w): i + 1]) for i in range(len(data))]


def plot_training(rewards, p_losses, v_losses, eval_results,
                  agent_label, env_label, plots_dir, num_episodes, window=100):
    """
    Génère le graphique d'entraînement avec 2 ou 3 sous-graphiques :
      1. Reward lissé (axe gauche) + Score éval checkpoints (axe droit)
      2. Policy Loss (cross-entropy entre la proba du réseau et la cible MCTS)
      3. Value Loss (si disponible : MSE entre valeur prédite et valeur réelle)
         → présent pour AlphaZero/MuZero, absent pour ExpertApprentice

    L'axe secondaire (orange) montre le VRAI progrès : le score mesuré
    sur la politique finale (sans exploration), aux checkpoints du syllabus.
    """
    episodes = np.arange(1, len(rewards) + 1)
    # 3 subplots si on a une value loss non nulle, sinon 2
    n_rows = 3 if (v_losses and any(v != 0 for v in v_losses)) else 2

    fig, axes = plt.subplots(n_rows, 1, figsize=(13, 4 * n_rows))
    fig.suptitle(f"{agent_label} — {env_label}\n({num_episodes:,} épisodes)",
                 fontsize=13, fontweight="bold")

    # ── Subplot 1 : Reward d'entraînement ────────────────────────────────
    axes[0].plot(episodes, smooth(rewards, window), color="#2196F3",
                 linewidth=1.3, label="Reward lissé")
    if eval_results:
        xs = sorted(eval_results.keys())
        ys = [eval_results[x]["score_moyen"] for x in xs]
        # Axe Y secondaire à droite pour le score d'évaluation
        ax0r = axes[0].twinx()
        ax0r.plot(xs, ys, "o--", color="#FF5722", markersize=6, linewidth=1.5,
                  label="Score éval.")
        ax0r.set_ylabel("Score éval. (politique)", color="#FF5722")
        ax0r.tick_params(axis="y", labelcolor="#FF5722")
        ax0r.legend(loc="lower right", fontsize=9)
    axes[0].set_ylabel(f"Reward lissé (fenêtre {window})")
    axes[0].set_xlabel("Épisode")
    axes[0].set_title("Reward d'entraînement")
    axes[0].legend(loc="upper left", fontsize=9)
    axes[0].grid(True, alpha=0.3)

    # ── Subplot 2 : Policy Loss ───────────────────────────────────────────
    # Doit décroître : le réseau apprend à reproduire la distribution MCTS
    axes[1].plot(episodes, smooth(p_losses, window), color="#2196F3", linewidth=1.0)
    axes[1].set_ylabel("Policy Loss")
    axes[1].set_xlabel("Épisode")
    axes[1].set_title("Policy Loss")
    axes[1].grid(True, alpha=0.3)

    # ── Subplot 3 : Value Loss (optionnel) ───────────────────────────────
    # Présent pour AlphaZero et MuZero : erreur de prédiction de la valeur
    if n_rows == 3:
        axes[2].plot(episodes, smooth(v_losses, window), color="#FF5722", linewidth=1.0)
        axes[2].set_ylabel("Value Loss")
        axes[2].set_xlabel("Épisode")
        axes[2].set_title("Value Loss")
        axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    fname = f"{agent_label}_{env_label.split()[0]}_training.png"
    path  = os.path.join(plots_dir, fname)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  → Graphique training : {path}")


def plot_progression(eval_results, agent_label, env_label, plots_dir, metric="score_moyen"):
    """
    Graphique de progression du score aux checkpoints (axe X logarithmique).
    L'axe log est adapté car les checkpoints sont 1k, 10k, 100k (puissances de 10).
    Chaque point est annoté avec la valeur exacte pour faciliter la lecture.
    """
    if not eval_results:
        return
    xs = sorted(eval_results.keys())
    ys = [eval_results[x][metric] for x in xs]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(xs, ys, "o-", color="#2196F3", linewidth=2, markersize=8)
    for x, y in zip(xs, ys):
        ax.annotate(f"{y:.3f}", (x, y), textcoords="offset points",
                    xytext=(5, 6), fontsize=9)
    ax.set_xscale("log")   # Axe X log : 1000, 10000, 100000 équidistants
    ax.set_xlabel("Épisodes d'entraînement (log)")
    ax.set_ylabel(metric.replace("_", " ").title())
    ax.set_title(f"{agent_label} — {env_label}\nÉvolution du {metric} aux checkpoints")
    ax.grid(True, alpha=0.3, which="both")
    plt.tight_layout()
    fname = f"{agent_label}_{env_label.split()[0]}_progression.png"
    path  = os.path.join(plots_dir, fname)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  → Graphique progression : {path}")


def plot_comparison(all_results_for_env, env_key, plots_dir, metric="score_moyen"):
    """
    Graphique barres comparant TOUS les agents sur un environnement donné.
    Chaque barre correspond à un agent, la valeur est le score au dernier checkpoint.

    Ceci est le graphique principal pour le rapport : il montre quel agent
    fonctionne le mieux sur chaque environnement.
    """
    env_label = ENV_CONFIGS[env_key]["label"]
    agents    = [k for k in AGENTS_LIST if k in all_results_for_env]
    labels    = [AGENT_CONFIGS[k]["label"] for k in agents]

    # Récupère le score au dernier checkpoint pour chaque agent
    values = []
    for k in agents:
        checkpoints = all_results_for_env[k].get("checkpoints", {})
        if checkpoints:
            # Dernier checkpoint = max des clés (converties en int)
            last = max(int(ep) for ep in checkpoints)
            values.append(checkpoints[str(last)][metric])
        else:
            # Fallback : final_eval si pas de checkpoints
            values.append(all_results_for_env[k].get("final_eval", {}).get(metric, 0))

    colors = ["#2196F3", "#4CAF50", "#FF9800", "#F44336", "#9C27B0", "#009688"]

    fig, ax = plt.subplots(figsize=(11, 5))
    bars = ax.bar(labels, values, color=colors[:len(labels)],
                  width=0.55, edgecolor="white", linewidth=1.2)
    # Annotation au-dessus de chaque barre
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{val:.3f}", ha="center", va="bottom",
                fontsize=10, fontweight="bold")
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.set_ylabel(metric.replace("_", " ").title(), fontsize=11)
    ax.set_title(f"Comparaison agents — {env_label}\n({metric})", fontsize=12)
    ax.set_ylim(min(-0.15, min(values) - 0.1), max(1.15, max(values) + 0.15))
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    fname = f"comparison_{env_key}_{metric}.png"
    path  = os.path.join(plots_dir, fname)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  → Graphique comparatif : {path}")


# ══════════════════════════════════════════════════════════════════════════════
#  AFFICHAGE CONSOLE
# ══════════════════════════════════════════════════════════════════════════════

def print_checkpoint_table(eval_results, agent_label, env_label):
    """Tableau des métriques aux checkpoints demandés par le syllabus."""
    print()
    print("=" * 70)
    print(f"  {agent_label}  —  {env_label}")
    print("=" * 70)
    print(f"  {'Épisodes':>12} | {'Score moyen':>12} | {'Longueur moy':>13} | {'Temps/coup':>12}")
    print("  " + "-" * 60)
    for ep, m in sorted(eval_results.items()):
        print(f"  {ep:>12,} | {m['score_moyen']:>12.4f} | "
              f"{m['longueur_moy']:>13.2f} | {m['temps_coup_ms']:>11.4f}ms")
    print()


def print_final_metrics(m, agent_label, env_label, mode=""):
    """
    Affiche les métriques finales avec séparateur visuel.
    mode : "Réseau" ou "MCTS" pour distinguer les deux modes d'AlphaZero
    """
    tag = f"  [{mode}]" if mode else ""
    print(f"\n  {'─' * 55}")
    print(f"  {agent_label}{tag}  —  {env_label}")
    print(f"  {'─' * 55}")
    print(f"  Score moyen    : {m['score_moyen']:.4f}")
    print(f"  Longueur moy.  : {m['longueur_moy']:.2f} steps")
    print(f"  Temps/coup     : {m['temps_coup_ms']:.4f} ms")
    if "taux_victoire" in m:
        print(f"  Taux victoire  : {m['taux_victoire'] * 100:.1f}%")
    if "taux_nul" in m:
        print(f"  Taux nul       : {m['taux_nul'] * 100:.1f}%")
    if "taux_defaite" in m:
        print(f"  Taux défaite   : {m['taux_defaite'] * 100:.1f}%")
    if "taux_echec" in m:
        print(f"  Taux échec     : {m['taux_echec'] * 100:.1f}%")


# ══════════════════════════════════════════════════════════════════════════════
#  EXPÉRIMENTATION PRINCIPALE : 1 AGENT × 1 ENVIRONNEMENT
# ══════════════════════════════════════════════════════════════════════════════

def run_experiment(agent_key, env_key, num_episodes, checkpoints, n_eval=500):
    """
    Orchestre l'expérimentation complète pour une combinaison agent × environnement.

    Flux :
      1. build_agent() → instancie l'agent avec les bons hyperparamètres
      2. Si no_training → évalue directement à chaque checkpoint (même résultat)
         Sinon → lance la boucle d'entraînement appropriée
      3. Affiche le tableau checkpoint + métriques finales
      4. Génère les graphiques training + progression
      5. Pour AlphaZero : éval supplémentaire réseau seul vs MCTS
      6. Sauvegarde le modèle + JSON de résultats

    Retourne : result_data (dict JSON-sérialisable)
    """
    acfg       = AGENT_CONFIGS[agent_key]
    ecfg       = ENV_CONFIGS[env_key]
    alabel     = acfg["label"]
    elabel     = ecfg["label"]
    train_type = ecfg["train_type"]   # "1player", "tictactoe" ou "quarto"

    # Création des dossiers spécifiques à cet agent
    agent_plots_dir   = os.path.join(PLOTS_DIR,   alabel)
    agent_results_dir = os.path.join(RESULTS_DIR, alabel)
    agent_models_dir  = os.path.join(MODELS_DIR,  alabel)
    for d in [agent_plots_dir, agent_results_dir, agent_models_dir]:
        os.makedirs(d, exist_ok=True)

    print("\n" + "═" * 70)
    print(f"  EXPÉRIMENTATION  {alabel}  —  {elabel}")
    print(f"  Épisodes : {num_episodes:,}   |   Checkpoints : {checkpoints}")
    print("═" * 70)

    agent, hyperparams = build_agent(agent_key, env_key)
    env = ecfg["env_fn"]()
    t_start = time.time()

    # ── CAS 1 : Agents sans entraînement (RandomRollout, MCTS) ───────────
    if acfg["no_training"]:
        print("  (Pas d'entraînement — évaluation directe)\n")
        # Ces agents n'apprennent pas → même score à tous les checkpoints
        # Utile pour la baseline et la comparaison
        eval_fn = ecfg["eval_fn_no_training"]
        eval_results = {}
        for cp in checkpoints:
            m = eval_fn(env, agent, n_games=n_eval)
            eval_results[cp] = m
            print(f"  {cp:>9,} épisodes | Score : {m['score_moyen']:.4f} | "
                  f"Longueur : {m['longueur_moy']:.2f} | Temps/coup : {m['temps_coup_ms']:.4f} ms")
        rewards  = [0.0]   # Pas de courbe d'entraînement
        p_losses = [0.0]
        v_losses = []
        # Évaluation finale avec la fonction plus complète (victoires/nuls/défaites)
        final_eval = ecfg["eval_fn"](env, agent, n_games=n_eval)

    # ── CAS 2 : Agents avec entraînement ─────────────────────────────────
    else:
        train_mode = acfg["train_mode"]   # "expert" ou "alphazero"
        # Sélection de la bonne boucle d'entraînement dans le dispatcher
        train_fn   = TRAIN_FNS[train_type][train_mode]

        # Choix de la fonction d'éval PENDANT l'entraînement :
        # - ExpertApprentice → éval avec le réseau apprenti (mesure la distillation)
        # - MuZero/AlphaZero → éval rapide avec select_action() (réseau seul)
        if train_mode == "expert":
            train_eval_fn = ecfg["eval_fn"]               # Réseau apprenti
        else:
            train_eval_fn = ecfg["eval_fn_no_training"]   # Réseau seul (rapide)

        train_kwargs = dict(
            env          = env,
            agent        = agent,
            num_episodes = num_episodes,
            checkpoints  = checkpoints,
            evaluate_fn  = train_eval_fn,
        )
        # max_steps seulement pour envs 1-joueur (LineWorld, GridWorld)
        # pour éviter les boucles infinies si l'agent tourne en rond
        if train_type == "1player":
            train_kwargs["max_steps"] = ecfg["max_steps"]

        # Lancement de l'entraînement → retourne les courbes et les métriques checkpoints
        # rewards    : reward par épisode (bruit + exploration)
        # p_losses   : policy loss par épisode (CrossEntropy)
        # v_losses   : value loss par épisode (MSE, vide pour ExpertApprentice)
        # eval_results : dict {épisode: métriques} aux checkpoints
        rewards, p_losses, v_losses, eval_results = train_fn(**train_kwargs)

        # Évaluation finale avec la fonction plus complète
        final_eval = ecfg["eval_fn"](env, agent, n_games=n_eval)

    elapsed = time.time() - t_start
    print(f"\n  Durée : {elapsed:.1f}s  ({elapsed / 60:.1f} min)")

    # ── Affichage + graphiques ─────────────────────────────────────────────
    print_checkpoint_table(eval_results, alabel, elabel)
    print_final_metrics(final_eval, alabel, elabel)

    if len(rewards) > 1:   # Seulement si on a des données d'entraînement réelles
        plot_training(rewards, p_losses, v_losses, eval_results,
                      alabel, elabel, agent_plots_dir, num_episodes)
        plot_progression(eval_results, alabel, elabel, agent_plots_dir)

    # ── CAS SPÉCIAL AlphaZero : Réseau seul vs MCTS ─────────────────────
    # On compare deux modes d'inférence :
    #   - Réseau seul  : argmax du réseau de politique → rapide
    #   - MCTS complet : planification PUCT guidée par le réseau → plus fort, plus lent
    final_eval_mcts = None
    if agent_key == "alphazero":
        detail_fn = acfg["detail_eval_fns"][env_key]
        print(f"\n  Phase MCTS : évaluation avec MCTS complet ({min(n_eval, 200)} parties)...")
        # use_mcts=False → réseau seul (argmax de la politique)
        final_eval_net  = detail_fn(agent, n_games=min(n_eval, 500), use_mcts=False)
        # use_mcts=True  → MCTS-PUCT guidé par le réseau
        final_eval_mcts = detail_fn(agent, n_games=min(n_eval, 200), use_mcts=True)

        # Tableau comparatif réseau vs MCTS
        print(f"\n  {'─' * 55}")
        print(f"  {'Métrique':<25} | {'Réseau seul':>13} | {'MCTS':>10}")
        print(f"  {'─' * 55}")
        for k in ["score_moyen", "longueur_moy", "temps_coup_ms"]:
            print(f"  {k:<25} | {final_eval_net[k]:>13.4f} | {final_eval_mcts[k]:>10.4f}")
        if "taux_victoire" in final_eval_net:
            print(f"  {'taux_victoire':<25} | {final_eval_net['taux_victoire']:>12.1%} | "
                  f"{final_eval_mcts['taux_victoire']:>9.1%}")
        if "taux_nul" in final_eval_net:
            print(f"  {'taux_nul':<25} | {final_eval_net['taux_nul']:>12.1%} | "
                  f"{final_eval_mcts['taux_nul']:>9.1%}")
        if "taux_defaite" in final_eval_net:
            print(f"  {'taux_defaite':<25} | {final_eval_net['taux_defaite']:>12.1%} | "
                  f"{final_eval_mcts['taux_defaite']:>9.1%}")
        print(f"  {'─' * 55}")
        final_eval = final_eval_net   # JSON principal = réseau seul

    # ── Sauvegarde du modèle ──────────────────────────────────────────────
    model_path = os.path.join(agent_models_dir, f"{alabel}_{elabel.split()[0]}.pt")
    agent.save(model_path)
    print(f"\n  → Modèle : {model_path}")

    # ── Sauvegarde JSON des résultats ─────────────────────────────────────
    result_data = {
        "agent":        alabel,
        "env":          elabel,
        "num_episodes": num_episodes,
        "elapsed_s":    round(elapsed, 1),
        "hyperparams":  hyperparams,        # Pour la reproductibilité
        "checkpoints":  {str(k): v for k, v in eval_results.items()},  # Métriques par checkpoint
        "final_eval":   final_eval,         # Évaluation finale (réseau seul)
    }
    if final_eval_mcts is not None:
        result_data["final_eval_mcts"] = final_eval_mcts  # Évaluation finale avec MCTS

    json_path = os.path.join(agent_results_dir, f"{alabel}_{elabel.split()[0]}.json")
    # Si le fichier existe → on le charge
    if os.path.exists(json_path):
        with open(json_path, "r") as f:
            try:
                existing_data = json.load(f)
            except:
                existing_data = []
    else:
        existing_data = []

    # Si c’est pas une liste → on transforme
    if not isinstance(existing_data, list):
        existing_data = [existing_data]

    # On ajoute le nouveau résultat
    existing_data.append(result_data)

    # On sauvegarde
    with open(json_path, "w") as f:
        json.dump(existing_data, f, indent=2)

    print(f"  → JSON (append) : {json_path}")

    return result_data


# ══════════════════════════════════════════════════════════════════════════════
#  GÉNÉRATION DU RAPPORT MARKDOWN GLOBAL
# ══════════════════════════════════════════════════════════════════════════════

def generate_markdown_report(all_results, num_episodes):
    """
    Génère un rapport Markdown complet avec :
      - Tableau des agents et hyperparamètres
      - Table score_moyen par (agent × env) au dernier checkpoint
      - Table temps/coup par (agent × env)
      - Détail par environnement de tous les checkpoints
      - Comparaison réseau vs MCTS pour AlphaZero
      - Section d'observations (à compléter manuellement)

    all_results : { agent_key: { env_key: result_data, ... }, ... }
    """
    lines = [
        "# Rapport d'expérimentation — Tous agents\n",
        f"> Épisodes d'entraînement : {num_episodes:,}\n",
        "",
        "## Agents étudiés\n",
        "| Agent | Type | Hyperparamètres principaux |",
        "|---|---|---|",
    ]

    # Description de chaque agent avec ses hyperparamètres
    for ak in AGENTS_LIST:
        if ak not in all_results:
            continue
        acfg = AGENT_CONFIGS[ak]
        first_env = next(iter(all_results[ak]))
        hp = all_results[ak][first_env].get("hyperparams", {})
        # Filtre state_size et num_actions qui sont des paramètres d'env, pas d'agent
        hp_str = ", ".join(f"{k}={v}" for k, v in hp.items() if k not in ("state_size", "num_actions"))
        agent_type = "Sans entraînement" if acfg["no_training"] else acfg.get("train_mode", "").capitalize()
        lines.append(f"| {acfg['label']} | {agent_type} | {hp_str or '—'} |")

    # Table de synthèse : score moyen par (agent × env)
    lines += [
        "",
        "## Score moyen par agent × environnement (dernier checkpoint)\n",
        "| Agent | " + " | ".join(ENV_CONFIGS[e]["label"] for e in ENVS_LIST) + " |",
        "|---|" + "|".join("---" for _ in ENVS_LIST) + "|",
    ]
    for ak in AGENTS_LIST:
        if ak not in all_results:
            continue
        row = f"| {AGENT_CONFIGS[ak]['label']} |"
        for ek in ENVS_LIST:
            data = all_results[ak].get(ek)
            if data:
                cps = data.get("checkpoints", {})
                if cps:
                    last  = max(int(ep) for ep in cps)
                    score = cps[str(last)]["score_moyen"]
                else:
                    score = data.get("final_eval", {}).get("score_moyen", float("nan"))
                row += f" {score:.4f} |"
            else:
                row += " N/A |"
        lines.append(row)

    # Table de synthèse : temps/coup
    lines += [
        "",
        "## Temps moyen par coup (ms) — dernier checkpoint\n",
        "| Agent | " + " | ".join(ENV_CONFIGS[e]["label"] for e in ENVS_LIST) + " |",
        "|---|" + "|".join("---" for _ in ENVS_LIST) + "|",
    ]
    for ak in AGENTS_LIST:
        if ak not in all_results:
            continue
        row = f"| {AGENT_CONFIGS[ak]['label']} |"
        for ek in ENVS_LIST:
            data = all_results[ak].get(ek)
            if data:
                cps = data.get("checkpoints", {})
                if cps:
                    last = max(int(ep) for ep in cps)
                    t_ms = cps[str(last)]["temps_coup_ms"]
                else:
                    t_ms = data.get("final_eval", {}).get("temps_coup_ms", float("nan"))
                row += f" {t_ms:.4f} |"
            else:
                row += " N/A |"
        lines.append(row)

    # Détail complet par environnement : tous les checkpoints
    lines += ["", "## Résultats détaillés par environnement\n"]
    for ek in ENVS_LIST:
        elabel = ENV_CONFIGS[ek]["label"]
        lines += [f"### {elabel}\n"]
        lines += [
            "| Agent | Épisodes | Score moyen | Longueur moy | Temps/coup (ms) |",
            "|---|---|---|---|---|",
        ]
        for ak in AGENTS_LIST:
            if ak not in all_results or ek not in all_results[ak]:
                continue
            cps    = all_results[ak][ek].get("checkpoints", {})
            alabel = AGENT_CONFIGS[ak]["label"]
            if cps:
                for ep_str, m in sorted(cps.items(), key=lambda x: int(x[0])):
                    lines.append(
                        f"| {alabel} | {int(ep_str):,} | {m['score_moyen']:.4f} | "
                        f"{m['longueur_moy']:.2f} | {m['temps_coup_ms']:.4f} |"
                    )
            lines.append("")

    # Section spéciale AlphaZero : réseau vs MCTS
    az_results = {ek: v for ek, v in all_results.get("alphazero", {}).items()
                  if "final_eval_mcts" in v}
    if az_results:
        lines += [
            "",
            "## AlphaZero — Réseau seul vs MCTS\n",
            "| Environnement | Mode | Score | Victoires | Nuls | Défaites | Longueur | ms/coup |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for ek, data in az_results.items():
            elabel = ENV_CONFIGS[ek]["label"]
            for mode_key, mode_label in [("final_eval", "Réseau"), ("final_eval_mcts", "MCTS")]:
                m     = data[mode_key]
                wins  = f"{m.get('taux_victoire', 0):.1%}"
                draws = f"{m.get('taux_nul',      0):.1%}"
                loss  = f"{m.get('taux_defaite',  0):.1%}"
                lines.append(
                    f"| {elabel} | {mode_label} | {m['score_moyen']:.4f} | "
                    f"{wins} | {draws} | {loss} | {m['longueur_moy']:.2f} | {m['temps_coup_ms']:.4f} |"
                )

    # Section observations et interprétations (à compléter)
    lines += [
        "",
        "## Observations et interprétations\n",
        "*(À compléter par votre analyse des résultats)*\n",
        "",
        "### RandomRollout",
        "- Agent de référence sans apprentissage. Simule N parties aléatoires par action et choisit la meilleure.",
        "- Performances stables mais coûteux en temps (O(N × profondeur) par coup).",
        "- Bonne baseline pour évaluer l'apport de l'apprentissage.",
        "",
        "### MCTS UCT",
        "- Exploration guidée par UCB1. Plus efficace que RandomRollout à iso-simulations.",
        "- Résultats indépendants de l'entraînement : même score à 1k, 10k, 100k épisodes.",
        "",
        "### ExpertApprentice",
        "- Distille la politique MCTS dans un réseau neuronal. Phase d'apprentissage supervisé.",
        "- Inférence rapide (réseau seul), mais qualité bornée par l'expert (MCTS à n_simulations fixé).",
        "",
        "### MuZero",
        "- Apprend un modèle latent de l'environnement. Planification dans l'espace latent.",
        "- Plus expressif qu'AlphaZero (pas besoin du modèle d'env fourni), mais plus complexe à entraîner.",
        "",
        "### MuZeroStochastique",
        "- Étend MuZero avec un encodeur VAE pour modéliser la stochasticité de l'environnement.",
        "- Utile quand l'env est partiellement observable. Sur nos envs déterministes, avantage marginal.",
        "",
        "### AlphaZero",
        "- Combine MCTS-PUCT guidé par réseau et auto-jeu. Convergence stable sur les jeux combinatoires.",
        "- Mode réseau seul (inférence rapide) vs MCTS complet (plus fort, plus lent).",
        "- Sur TicTacToe et Quarto, le gain MCTS est significatif après suffisamment d'épisodes.",
    ]

    md_path = os.path.join(RESULTS_DIR, "rapport_agents.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\n  → Rapport Markdown : {md_path}")
    return md_path


# ══════════════════════════════════════════════════════════════════════════════
#  POINT D'ENTRÉE
# ══════════════════════════════════════════════════════════════════════════════

def build_checkpoints(num_episodes):
    """
    Génère les checkpoints selon le syllabus du projet :
      1 000 → 10 000 → 100 000 → 1 000 000 épisodes
    Garde uniquement ceux qui sont ≤ num_episodes,
    et ajoute toujours num_episodes lui-même.
    """
    candidates = [1_000, 10_000, 100_000, 1_000_000]
    cps = [c for c in candidates if c <= num_episodes]
    if not cps or cps[-1] != num_episodes:
        cps.append(num_episodes)
    return sorted(set(cps))


def main(cli_args=None):
    parser = argparse.ArgumentParser(
        description="Expérimentations complètes — tous agents × 4 environnements"
    )
    parser.add_argument("--agent",    default="all",
                        choices=["all"] + AGENTS_LIST,
                        help="Agent à expérimenter (default: all)")
    parser.add_argument("--env",      default="all",
                        choices=["all"] + ENVS_LIST,
                        help="Environnement cible (default: all)")
    parser.add_argument("--episodes", type=int, default=10_000,
                        help="Épisodes d'entraînement (default: 10000)")
    parser.add_argument("--n_eval",   type=int, default=500,
                        help="Parties pour l'évaluation finale (default: 500)")
    parser.add_argument("--skip_no_training", action="store_true",
                        help="Ignore RandomRollout et MCTS (pas d'entraînement)")
    args = parser.parse_args(cli_args)

    agents = AGENTS_LIST if args.agent == "all" else [args.agent]
    envs   = ENVS_LIST   if args.env   == "all" else [args.env]

    # Option pour sauter les agents sans entraînement (gain de temps si déjà évalués)
    if args.skip_no_training:
        agents = [a for a in agents if not AGENT_CONFIGS[a]["no_training"]]

    checkpoints = build_checkpoints(args.episodes)

    print(f"\n{'═' * 70}")
    print(f"  EXPÉRIMENTATIONS AGENTS")
    print(f"  Agents         : {[AGENT_CONFIGS[a]['label'] for a in agents]}")
    print(f"  Environnements : {[ENV_CONFIGS[e]['label'] for e in envs]}")
    print(f"  Épisodes       : {args.episodes:,}")
    print(f"  Checkpoints    : {checkpoints}")
    print(f"  Éval (parties) : {args.n_eval}")
    print(f"{'═' * 70}")

    all_results = {}
    global_t0   = time.time()

    # Double boucle principale : agents × environnements
    for agent_key in agents:
        all_results[agent_key] = {}
        for env_key in envs:
            data = run_experiment(
                agent_key, env_key,
                num_episodes = args.episodes,
                checkpoints  = checkpoints,
                n_eval       = args.n_eval,
            )
            all_results[agent_key][env_key] = data

    # ── Graphiques comparatifs par environnement ──────────────────────────
    # Un graphique par env montrant les scores de tous les agents
    print("\n  Génération des graphiques comparatifs ...")
    for env_key in envs:
        env_data = {ak: all_results[ak][env_key]
                    for ak in agents if env_key in all_results.get(ak, {})}
        if len(env_data) > 1:   # Au moins 2 agents pour comparer
            plot_comparison(env_data, env_key, PLOTS_DIR, metric="score_moyen")

    # ── Rapport Markdown global ───────────────────────────────────────────
    generate_markdown_report(all_results, args.episodes)

    # ── JSON de synthèse globale ──────────────────────────────────────────
    json_path = os.path.join(RESULTS_DIR, "all_agents_summary.json")
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"  → JSON synthèse : {json_path}")

    # ── Résumé console final ──────────────────────────────────────────────
    last_cp = max(checkpoints)
    total   = time.time() - global_t0
    print(f"\n{'═' * 80}")
    print(f"  RÉSUMÉ GLOBAL — Score moyen au checkpoint {last_cp:,}")
    print(f"{'═' * 80}")
    header = f"  {'Agent':>22}" + "".join(f" | {ENV_CONFIGS[e]['label'][:14]:>14}" for e in envs)
    print(header)
    print("  " + "─" * (len(header) - 2))
    for ak in agents:
        row = f"  {AGENT_CONFIGS[ak]['label']:>22}"
        for ek in envs:
            data = all_results[ak].get(ek, {})
            cps  = data.get("checkpoints", {})
            if cps:
                last_key = str(max(int(k) for k in cps))
                score = cps[last_key]["score_moyen"]
                row  += f" | {score:>14.4f}"
            else:
                row  += f" | {'N/A':>14}"
        print(row)
    print()
    print(f"  Durée totale   : {total:.0f}s ({total / 60:.1f} min)")
    print(f"  Modèles        → {MODELS_DIR}")
    print(f"  Résultats      → {RESULTS_DIR}")
    print(f"  Graphiques     → {PLOTS_DIR}")
    print(f"{'═' * 80}\n")


if __name__ == "__main__":
    main()