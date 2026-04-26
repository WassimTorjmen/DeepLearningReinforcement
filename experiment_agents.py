"""
experiment_agents.py
====================
Expérimentations complètes pour tous les agents sur les 4 environnements.

Agents couverts :
  - RandomRollout           (sans entraînement)
  - MCTS UCT                (sans entraînement)
  - ExpertApprentice        (supervisé depuis MCTS)
  - MuZero                  (self-play + planification latente)
  - MuZeroStochastique      (MuZero + VAE stochastique)
  - AlphaZero               (MCTS-PUCT + self-play)

Pour chaque combinaison agent × environnement :
  - Entraînement avec checkpoints à 1k / 10k / 100k épisodes
  - Métriques syllabus : score moyen, longueur, temps/coup
  - Graphiques d'apprentissage (reward, policy loss, value loss)
  - Graphiques de progression aux checkpoints
  - Comparaison réseau seul vs MCTS (AlphaZero uniquement)
  - Tableau comparatif multi-agents par environnement
  - Export JSON + rapport Markdown

Utilisation :
  python experiment_agents.py                                       # tous agents, tous envs, 10k épisodes
  python experiment_agents.py --episodes 100000                     # 100k épisodes
  python experiment_agents.py --agent alphazero --env tictactoe     # agent + env ciblés
  python experiment_agents.py --agent all --env lineworld           # tous agents sur un env
  python experiment_agents.py --agent randomrollout --env all       # un agent sur tous les envs
  python experiment_agents.py --skip_no_training                    # saute Random/MCTS
"""

import sys
import os
import argparse
import time
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from Environnements.line_world   import LineWorld
from Environnements.grid_world   import GridWorld
from Environnements.tictactoe    import TicTacToe
from Environnements.quarto       import QuartoEnv
from Agents.Random_Rollout       import RandomRolloutAgent
from Agents.MCTS                 import MCTSAgent
from Agents.Expert_Apprentice    import ExpertApprenticeAgent
from Agents.MuZero               import MuZeroAgent
from Agents.Muzerostochastic     import MuZeroStochasticAgent
from Agents.Alpha_zero           import AlphaZeroAgent
from experiment import (
    train_alphazero_1player,
    train_alphazero_tictactoe,
    train_alphazero_quarto,
    evaluate_no_training_1player,
    evaluate_no_training_tictactoe,
    evaluate_no_training_quarto,
)
from Evaluation.evaluate_alphazero import (
    evaluate_lineworld   as az_eval_lineworld,
    evaluate_gridworld   as az_eval_gridworld,
    evaluate_tictactoe   as az_eval_tictactoe,
    evaluate_quarto      as az_eval_quarto,
)
from Training.train_agents import (
    train_expert_1player,
    train_expert_tictactoe,
    train_expert_quarto,
    evaluate_agent_1player,
    evaluate_agent_tictactoe,
    evaluate_agent_quarto,
)

# ── Dossiers de sortie ─────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR  = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results", "experiment_agents")
PLOTS_DIR   = os.path.join(BASE_DIR, "plots",   "experiment_agents")
for d in [MODELS_DIR, RESULTS_DIR, PLOTS_DIR]:
    os.makedirs(d, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
#  Configurations des environnements
# ══════════════════════════════════════════════════════════════════════════════

ENV_CONFIGS = {
    "lineworld": {
        "label":       "LineWorld",
        "state_size":  8,
        "num_actions": 2,
        "env_fn":      lambda: LineWorld(size=6),
        "train_type":  "1player",
        "eval_fn":     evaluate_agent_1player,
        "eval_fn_no_training": evaluate_no_training_1player,
        "max_steps":   200,
    },
    "gridworld": {
        "label":       "GridWorld (5×5)",
        "state_size":  31,
        "num_actions": 4,
        "env_fn":      lambda: GridWorld(rows=5, cols=5),
        "train_type":  "1player",
        "eval_fn":     evaluate_agent_1player,
        "eval_fn_no_training": evaluate_no_training_1player,
        "max_steps":   200,
    },
    "tictactoe": {
        "label":       "TicTacToe (vs Random)",
        "state_size":  27,
        "num_actions": 9,
        "env_fn":      lambda: TicTacToe(),
        "train_type":  "tictactoe",
        "eval_fn":     evaluate_agent_tictactoe,
        "eval_fn_no_training": evaluate_no_training_tictactoe,
        "max_steps":   None,
    },
    "quarto": {
        "label":       "Quarto (vs Random)",
        "state_size":  105,
        "num_actions": 32,
        "env_fn":      lambda: QuartoEnv(),
        "train_type":  "quarto",
        "eval_fn":     evaluate_agent_quarto,
        "eval_fn_no_training": evaluate_no_training_quarto,
        "max_steps":   None,
    },
}


# ══════════════════════════════════════════════════════════════════════════════
#  Configurations des agents
# ══════════════════════════════════════════════════════════════════════════════

AGENT_CONFIGS = {
    "randomrollout": {
        "label":        "RandomRollout",
        "no_training":  True,
        "train_mode":   None,
        "fn": lambda s, a, env_key: RandomRolloutAgent(n_rollouts=50),
        "hyperparams": {"n_rollouts": 50},
    },
    "mcts": {
        "label":        "MCTS_UCT",
        "no_training":  True,
        "train_mode":   None,
        "fn": lambda s, a, env_key: MCTSAgent(n_simulations=100),
        "hyperparams": {"n_simulations": 100},
    },
    "expertapprentice": {
        "label":        "ExpertApprentice",
        "no_training":  False,
        "train_mode":   "expert",
        "fn": lambda s, a, env_key: ExpertApprenticeAgent(
            state_size=s, num_actions=a, n_simulations=50, hidden_size=128, lr=1e-3
        ),
        "hyperparams": {"hidden_size": 128, "lr": 1e-3, "n_simulations": 50},
    },
    "muzero": {
        "label":        "MuZero",
        "no_training":  False,
        "train_mode":   "alphazero",
        "fn": lambda s, a, env_key: MuZeroAgent(
            state_size=s, num_actions=a, hidden_size=128, n_simulations=3, lr=1e-3
        ),
        "hyperparams": {"hidden_size": 128, "lr": 1e-3, "n_simulations": 3},
    },
    "muzero_stochastic": {
        "label":        "MuZeroStochastic",
        "no_training":  False,
        "train_mode":   "alphazero",
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
        # Hyperparamètres spécifiques par environnement (identiques à experiment_alphazero.py)
        "az_configs": {
            "lineworld": {"hidden_size": 64,  "n_simulations": 20, "c_puct": 1.0, "lr": 3e-4},
            "gridworld": {"hidden_size": 128, "n_simulations": 20, "c_puct": 1.0, "lr": 3e-4},
            "tictactoe": {"hidden_size": 128, "n_simulations": 50, "c_puct": 1.5, "lr": 1e-3},
            "quarto":    {"hidden_size": 256, "n_simulations": 30, "c_puct": 1.5, "lr": 1e-3},
        },
        "fn": lambda s, a, env_key: None,  # surchargé dans build_agent()
        "hyperparams": {},                  # surchargé dans build_agent()
        # Fonctions d'évaluation détaillée avec support use_mcts
        "detail_eval_fns": {
            "lineworld": az_eval_lineworld,
            "gridworld": az_eval_gridworld,
            "tictactoe": az_eval_tictactoe,
            "quarto":    az_eval_quarto,
        },
    },
}

# Dispatcher fonctions d'entraînement
TRAIN_FNS = {
    "1player": {
        "expert":    train_expert_1player,
        "alphazero": train_alphazero_1player,
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
#  Construction de l'agent
# ══════════════════════════════════════════════════════════════════════════════

def build_agent(agent_key, env_key):
    """Instancie l'agent avec les bons hyperparamètres pour cet environnement."""
    acfg = AGENT_CONFIGS[agent_key]
    ecfg = ENV_CONFIGS[env_key]

    if agent_key == "alphazero":
        az_cfg = acfg["az_configs"][env_key]
        agent  = AlphaZeroAgent(
            state_size    = ecfg["state_size"],
            num_actions   = ecfg["num_actions"],
            hidden_size   = az_cfg["hidden_size"],
            n_simulations = az_cfg["n_simulations"],
            lr            = az_cfg["lr"],
            c_puct        = az_cfg["c_puct"],
        )
        hyperparams = az_cfg.copy()
        hyperparams.update({"state_size": ecfg["state_size"], "num_actions": ecfg["num_actions"]})
    else:
        agent       = acfg["fn"](ecfg["state_size"], ecfg["num_actions"], env_key)
        hyperparams = acfg["hyperparams"].copy()

    return agent, hyperparams


# ══════════════════════════════════════════════════════════════════════════════
#  Helpers graphiques (même style qu'experiment_alphazero.py)
# ══════════════════════════════════════════════════════════════════════════════

def smooth(data, w=100):
    return [np.mean(data[max(0, i - w): i + 1]) for i in range(len(data))]


def plot_training(rewards, p_losses, v_losses, eval_results,
                  agent_label, env_label, plots_dir, num_episodes, window=100):
    """Graphique training : reward + policy loss + value loss."""
    episodes = np.arange(1, len(rewards) + 1)
    n_rows = 3 if (v_losses and any(v != 0 for v in v_losses)) else 2

    fig, axes = plt.subplots(n_rows, 1, figsize=(13, 4 * n_rows))
    fig.suptitle(f"{agent_label} — {env_label}\n({num_episodes:,} épisodes)",
                 fontsize=13, fontweight="bold")

    # Reward
    axes[0].plot(episodes, smooth(rewards, window), color="#2196F3",
                 linewidth=1.3, label="Reward lissé")
    if eval_results:
        xs = sorted(eval_results.keys())
        ys = [eval_results[x]["score_moyen"] for x in xs]
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

    # Policy loss
    axes[1].plot(episodes, smooth(p_losses, window), color="#2196F3", linewidth=1.0)
    axes[1].set_ylabel("Policy Loss")
    axes[1].set_xlabel("Épisode")
    axes[1].set_title("Policy Loss")
    axes[1].grid(True, alpha=0.3)

    # Value loss (si disponible)
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
    """Graphique progression du score moyen aux checkpoints (axe log)."""
    if not eval_results:
        return
    xs = sorted(eval_results.keys())
    ys = [eval_results[x][metric] for x in xs]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(xs, ys, "o-", color="#2196F3", linewidth=2, markersize=8)
    for x, y in zip(xs, ys):
        ax.annotate(f"{y:.3f}", (x, y), textcoords="offset points",
                    xytext=(5, 6), fontsize=9)
    ax.set_xscale("log")
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
    """Graphique barres comparant tous les agents sur un environnement donné."""
    env_label = ENV_CONFIGS[env_key]["label"]
    agents    = [k for k in AGENTS_LIST if k in all_results_for_env]
    labels    = [AGENT_CONFIGS[k]["label"] for k in agents]

    # Score au dernier checkpoint disponible
    values = []
    for k in agents:
        checkpoints = all_results_for_env[k].get("checkpoints", {})
        if checkpoints:
            last = max(int(ep) for ep in checkpoints)
            values.append(checkpoints[str(last)][metric])
        else:
            values.append(all_results_for_env[k].get("final_eval", {}).get(metric, 0))

    colors = ["#2196F3", "#4CAF50", "#FF9800", "#F44336", "#9C27B0", "#009688"]

    fig, ax = plt.subplots(figsize=(11, 5))
    bars = ax.bar(labels, values, color=colors[:len(labels)],
                  width=0.55, edgecolor="white", linewidth=1.2)
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
#  Affichage console
# ══════════════════════════════════════════════════════════════════════════════

def print_checkpoint_table(eval_results, agent_label, env_label):
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
#  Expérimentation principale : 1 agent × 1 environnement
# ══════════════════════════════════════════════════════════════════════════════

def run_experiment(agent_key, env_key, num_episodes, checkpoints, n_eval=500):
    acfg      = AGENT_CONFIGS[agent_key]
    ecfg      = ENV_CONFIGS[env_key]
    alabel    = acfg["label"]
    elabel    = ecfg["label"]
    train_type = ecfg["train_type"]

    # Dossiers spécifiques à cet agent
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

    # ── Agents sans entraînement ────────────────────────────────────────────
    if acfg["no_training"]:
        print("  (Pas d'entraînement — évaluation directe)\n")
        eval_fn = ecfg["eval_fn_no_training"]
        eval_results = {}
        for cp in checkpoints:
            m = eval_fn(env, agent, n_games=n_eval)
            eval_results[cp] = m
            print(f"  {cp:>9,} épisodes | Score : {m['score_moyen']:.4f} | "
                  f"Longueur : {m['longueur_moy']:.2f} | Temps/coup : {m['temps_coup_ms']:.4f} ms")
        rewards = [0.0]
        p_losses = [0.0]
        v_losses = []
        # Évaluation finale détaillée
        final_eval = ecfg["eval_fn"](env, agent, n_games=n_eval)

    # ── Agents avec entraînement ────────────────────────────────────────────
    else:
        train_mode = acfg["train_mode"]
        train_fn   = TRAIN_FNS[train_type][train_mode]

        # eval_fn pendant le training
        if train_mode == "expert":
            train_eval_fn = ecfg["eval_fn"]
        else:
            train_eval_fn = ecfg["eval_fn_no_training"]

        train_kwargs = dict(
            env          = env,
            agent        = agent,
            num_episodes = num_episodes,
            checkpoints  = checkpoints,
            evaluate_fn  = train_eval_fn,
        )
        if train_type == "1player":
            train_kwargs["max_steps"] = ecfg["max_steps"]

        rewards, p_losses, v_losses, eval_results = train_fn(**train_kwargs)
        final_eval = ecfg["eval_fn"](env, agent, n_games=n_eval)

    elapsed = time.time() - t_start
    print(f"\n  Durée : {elapsed:.1f}s  ({elapsed / 60:.1f} min)")

    # ── Affichage + graphiques ─────────────────────────────────────────────
    print_checkpoint_table(eval_results, alabel, elabel)
    print_final_metrics(final_eval, alabel, elabel)

    if len(rewards) > 1:
        plot_training(rewards, p_losses, v_losses, eval_results,
                      alabel, elabel, agent_plots_dir, num_episodes)
        plot_progression(eval_results, alabel, elabel, agent_plots_dir)

    # ── AlphaZero : évaluation supplémentaire réseau seul vs MCTS ─────────
    final_eval_mcts = None
    if agent_key == "alphazero":
        detail_fn = acfg["detail_eval_fns"][env_key]
        print(f"\n  Phase MCTS : évaluation avec MCTS complet ({min(n_eval, 200)} parties)...")
        final_eval_net  = detail_fn(agent, n_games=min(n_eval, 500), use_mcts=False)
        final_eval_mcts = detail_fn(agent, n_games=min(n_eval, 200), use_mcts=True)

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
        final_eval = final_eval_net  # pour la sauvegarde JSON principale

    # ── Sauvegarde modèle ──────────────────────────────────────────────────
    model_path = os.path.join(agent_models_dir, f"{alabel}_{elabel.split()[0]}.pt")
    agent.save(model_path)
    print(f"\n  → Modèle : {model_path}")

    # ── JSON ───────────────────────────────────────────────────────────────
    result_data = {
        "agent":          alabel,
        "env":            elabel,
        "num_episodes":   num_episodes,
        "elapsed_s":      round(elapsed, 1),
        "hyperparams":    hyperparams,
        "checkpoints":    {str(k): v for k, v in eval_results.items()},
        "final_eval":     final_eval,
    }
    if final_eval_mcts is not None:
        result_data["final_eval_mcts"] = final_eval_mcts

    json_path = os.path.join(agent_results_dir, f"{alabel}_{elabel.split()[0]}.json")
    with open(json_path, "w") as f:
        json.dump(result_data, f, indent=2)
    print(f"  → JSON : {json_path}")

    return result_data


# ══════════════════════════════════════════════════════════════════════════════
#  Génération du rapport Markdown global
# ══════════════════════════════════════════════════════════════════════════════

def generate_markdown_report(all_results, num_episodes):
    """
    all_results = { agent_key: { env_key: result_data, ... }, ... }
    """
    lines = [
        "# Rapport d'expérimentation — Tous agents\n",
        f"> Épisodes d'entraînement : {num_episodes:,}\n",
        "",
        "## Agents étudiés\n",
        "| Agent | Type | Hyperparamètres principaux |",
        "|---|---|---|",
    ]

    for ak in AGENTS_LIST:
        if ak not in all_results:
            continue
        acfg = AGENT_CONFIGS[ak]
        # Récupère les hyperparams du premier env disponible
        first_env = next(iter(all_results[ak]))
        hp = all_results[ak][first_env].get("hyperparams", {})
        hp_str = ", ".join(f"{k}={v}" for k, v in hp.items() if k not in ("state_size", "num_actions"))
        agent_type = "Sans entraînement" if acfg["no_training"] else acfg.get("train_mode", "").capitalize()
        lines.append(f"| {acfg['label']} | {agent_type} | {hp_str or '—'} |")

    # Table score moyen par agent × environnement
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
                    last = max(int(ep) for ep in cps)
                    score = cps[str(last)]["score_moyen"]
                else:
                    score = data.get("final_eval", {}).get("score_moyen", float("nan"))
                row += f" {score:.4f} |"
            else:
                row += " N/A |"
        lines.append(row)

    # Table temps/coup
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

    # Détail par environnement
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
            cps = all_results[ak][ek].get("checkpoints", {})
            alabel = AGENT_CONFIGS[ak]["label"]
            if cps:
                for ep_str, m in sorted(cps.items(), key=lambda x: int(x[0])):
                    lines.append(
                        f"| {alabel} | {int(ep_str):,} | {m['score_moyen']:.4f} | "
                        f"{m['longueur_moy']:.2f} | {m['temps_coup_ms']:.4f} |"
                    )
            lines.append("")

    # Section AlphaZero réseau vs MCTS
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
                m = data[mode_key]
                wins  = f"{m.get('taux_victoire', 0):.1%}"
                draws = f"{m.get('taux_nul',      0):.1%}"
                loss  = f"{m.get('taux_defaite',  0):.1%}"
                lines.append(
                    f"| {elabel} | {mode_label} | {m['score_moyen']:.4f} | "
                    f"{wins} | {draws} | {loss} | {m['longueur_moy']:.2f} | {m['temps_coup_ms']:.4f} |"
                )

    # Observations
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
#  Point d'entrée
# ══════════════════════════════════════════════════════════════════════════════

def build_checkpoints(num_episodes):
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

    # Graphiques comparatifs par environnement
    print("\n  Génération des graphiques comparatifs ...")
    for env_key in envs:
        env_data = {ak: all_results[ak][env_key]
                    for ak in agents if env_key in all_results.get(ak, {})}
        if len(env_data) > 1:
            plot_comparison(env_data, env_key, PLOTS_DIR, metric="score_moyen")

    # Rapport Markdown global
    generate_markdown_report(all_results, args.episodes)

    # JSON global de synthèse
    json_path = os.path.join(RESULTS_DIR, "all_agents_summary.json")
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"  → JSON synthèse : {json_path}")

    # Résumé console final
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
