"""
train_alphazero.py
==================
Entraînement d'AlphaZero sur les 4 environnements :
  - LineWorld
  - GridWorld
  - TicTacToe (vs Random)
  - Quarto    (vs Random)

Utilisation :
  python Training/train_alphazero.py --env all
  python Training/train_alphazero.py --env lineworld
  python Training/train_alphazero.py --env gridworld
  python Training/train_alphazero.py --env tictactoe
  python Training/train_alphazero.py --env quarto
  python Training/train_alphazero.py --env all --episodes 10000
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

# ── Chemin projet ──────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Environnements.line_world  import LineWorld
from Environnements.grid_world  import GridWorld
from Environnements.tictactoe   import TicTacToe
from Environnements.quarto      import QuartoEnv
from Agents.Alpha_zero          import AlphaZeroAgent
from experiment import (
    train_alphazero_1player,
    train_alphazero_tictactoe,
    train_alphazero_quarto,
    evaluate_no_training_1player,
    evaluate_no_training_tictactoe,
    evaluate_no_training_quarto,
)

# ── Dossiers de sauvegarde ─────────────────────────────────────────────────────
MODELS_DIR  = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "alphazero")
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "alphazero")
PLOTS_DIR   = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "plots",   "alphazero")

for d in [MODELS_DIR, RESULTS_DIR, PLOTS_DIR]:
    os.makedirs(d, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
#  Utilitaires de tracé
# ══════════════════════════════════════════════════════════════════════════════

def smooth(data, window=50):
    if len(data) < window:
        return data
    return [np.mean(data[max(0, i - window): i + 1]) for i in range(len(data))]


def plot_and_save(rewards, policy_losses, value_losses, eval_results, env_name, window=100):
    n_plots = 3 if value_losses else 2
    fig, axes = plt.subplots(n_plots, 1, figsize=(12, 4 * n_plots))
    fig.suptitle(f"AlphaZero — {env_name}", fontsize=14, fontweight="bold")

    # Reward
    axes[0].plot(smooth(rewards, window), color="steelblue", linewidth=1.2)
    axes[0].set_ylabel(f"Reward lissé (fenêtre {window})")
    axes[0].set_xlabel("Épisode")
    axes[0].set_title("Reward d'entraînement")
    axes[0].grid(True, alpha=0.3)

    # Points de checkpoint sur le graphe rewards
    if eval_results:
        xs = list(eval_results.keys())
        ys = [eval_results[x]["score_moyen"] for x in xs]
        ax2 = axes[0].twinx()
        ax2.plot(xs, ys, "o--", color="darkorange", label="Score éval.")
        ax2.set_ylabel("Score éval. (500 parties)")
        ax2.legend(loc="lower right")

    # Policy loss
    axes[1].plot(smooth(policy_losses, window), color="steelblue", linewidth=1.0)
    axes[1].set_ylabel("Policy Loss")
    axes[1].set_xlabel("Épisode")
    axes[1].set_title("Policy Loss")
    axes[1].grid(True, alpha=0.3)

    # Value loss
    if value_losses and n_plots == 3:
        axes[2].plot(smooth(value_losses, window), color="darkorange", linewidth=1.0)
        axes[2].set_ylabel("Value Loss")
        axes[2].set_xlabel("Épisode")
        axes[2].set_title("Value Loss")
        axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, f"AlphaZero_{env_name}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  → Graphique sauvegardé : {path}")


def print_table(eval_results, env_name):
    print()
    print("=" * 70)
    print(f"  RÉSULTATS AlphaZero — {env_name}  (éval. sur 500 parties)")
    print("=" * 70)
    print(f"{'Épisodes':>12} | {'Score moyen':>12} | {'Longueur moy':>13} | {'Temps/coup':>12}")
    print("-" * 70)
    for ep, m in sorted(eval_results.items()):
        print(f"{ep:>12,} | {m['score_moyen']:>12.4f} | {m['longueur_moy']:>13.2f} | {m['temps_coup_ms']:>11.4f}ms")
    print()


# ══════════════════════════════════════════════════════════════════════════════
#  Configurations par défaut pour chaque environnement
# ══════════════════════════════════════════════════════════════════════════════

CONFIGS = {
    "lineworld": {
        # LineWorld : état = 8 valeurs, actions = 2 (gauche / droite)
        "state_size":    8,
        "num_actions":   2,
        "hidden_size":   64,
        "lr":            3e-4,
        "n_simulations": 20,
        "c_puct":        1.0,
    },
    "gridworld": {
        # GridWorld 5×5 : état = 31 valeurs, actions = 4
        "state_size":    31,
        "num_actions":   4,
        "hidden_size":   128,
        "lr":            3e-4,
        "n_simulations": 20,
        "c_puct":        1.0,
    },
    "tictactoe": {
        # TicTacToe : état = 27, actions = 9
        "state_size":    27,
        "num_actions":   9,
        "hidden_size":   128,
        "lr":            1e-3,
        "n_simulations": 50,
        "c_puct":        1.5,
    },
    "quarto": {
        # Quarto : état = 105, actions = 32 (16 choose + 16 place)
        "state_size":    105,
        "num_actions":   32,
        "hidden_size":   256,
        "lr":            1e-3,
        "n_simulations": 30,
        "c_puct":        1.5,
    },
}


# ══════════════════════════════════════════════════════════════════════════════
#  Fonctions d'entraînement par environnement
# ══════════════════════════════════════════════════════════════════════════════

def run_lineworld(num_episodes, checkpoints):
    print("\n" + "═" * 70)
    print("  AlphaZero  —  LineWorld")
    print("═" * 70)

    cfg   = CONFIGS["lineworld"]
    env   = LineWorld(size=6)
    agent = AlphaZeroAgent(
        state_size    = cfg["state_size"],
        num_actions   = cfg["num_actions"],
        n_simulations = cfg["n_simulations"],
        hidden_size   = cfg["hidden_size"],
        lr            = cfg["lr"],
        c_puct        = cfg["c_puct"],
    )

    t0 = time.time()
    rewards, p_losses, v_losses, eval_results = train_alphazero_1player(
        env, agent,
        num_episodes = num_episodes,
        checkpoints  = checkpoints,
        evaluate_fn  = evaluate_no_training_1player,
        max_steps    = 200,
    )
    elapsed = time.time() - t0
    print(f"\n  Durée totale : {elapsed:.1f}s")

    print_table(eval_results, "LineWorld")
    plot_and_save(rewards, p_losses, v_losses, eval_results, "LineWorld")

    model_path = os.path.join(MODELS_DIR, "AlphaZero_LineWorld.pt")
    agent.save(model_path)
    print(f"  → Modèle sauvegardé : {model_path}")

    save_results(eval_results, "LineWorld", elapsed)
    return eval_results


def run_gridworld(num_episodes, checkpoints):
    print("\n" + "═" * 70)
    print("  AlphaZero  —  GridWorld (5×5)")
    print("═" * 70)

    cfg   = CONFIGS["gridworld"]
    env   = GridWorld(rows=5, cols=5)
    agent = AlphaZeroAgent(
        state_size    = cfg["state_size"],
        num_actions   = cfg["num_actions"],
        n_simulations = cfg["n_simulations"],
        hidden_size   = cfg["hidden_size"],
        lr            = cfg["lr"],
        c_puct        = cfg["c_puct"],
    )

    t0 = time.time()
    rewards, p_losses, v_losses, eval_results = train_alphazero_1player(
        env, agent,
        num_episodes = num_episodes,
        checkpoints  = checkpoints,
        evaluate_fn  = evaluate_no_training_1player,
        max_steps    = 200,
    )
    elapsed = time.time() - t0
    print(f"\n  Durée totale : {elapsed:.1f}s")

    print_table(eval_results, "GridWorld")
    plot_and_save(rewards, p_losses, v_losses, eval_results, "GridWorld")

    model_path = os.path.join(MODELS_DIR, "AlphaZero_GridWorld.pt")
    agent.save(model_path)
    print(f"  → Modèle sauvegardé : {model_path}")

    save_results(eval_results, "GridWorld", elapsed)
    return eval_results


def run_tictactoe(num_episodes, checkpoints):
    print("\n" + "═" * 70)
    print("  AlphaZero  —  TicTacToe (vs Random)")
    print("═" * 70)

    cfg   = CONFIGS["tictactoe"]
    env   = TicTacToe()
    agent = AlphaZeroAgent(
        state_size    = cfg["state_size"],
        num_actions   = cfg["num_actions"],
        n_simulations = cfg["n_simulations"],
        hidden_size   = cfg["hidden_size"],
        lr            = cfg["lr"],
        c_puct        = cfg["c_puct"],
    )

    t0 = time.time()
    rewards, p_losses, v_losses, eval_results = train_alphazero_tictactoe(
        env, agent,
        num_episodes = num_episodes,
        checkpoints  = checkpoints,
        evaluate_fn  = evaluate_no_training_tictactoe,
    )
    elapsed = time.time() - t0
    print(f"\n  Durée totale : {elapsed:.1f}s")

    print_table(eval_results, "TicTacToe")
    plot_and_save(rewards, p_losses, v_losses, eval_results, "TicTacToe")

    model_path = os.path.join(MODELS_DIR, "AlphaZero_TicTacToe.pt")
    agent.save(model_path)
    print(f"  → Modèle sauvegardé : {model_path}")

    save_results(eval_results, "TicTacToe", elapsed)
    return eval_results


def run_quarto(num_episodes, checkpoints):
    print("\n" + "═" * 70)
    print("  AlphaZero  —  Quarto (vs Random)")
    print("═" * 70)

    cfg   = CONFIGS["quarto"]
    env   = QuartoEnv()
    agent = AlphaZeroAgent(
        state_size    = cfg["state_size"],
        num_actions   = cfg["num_actions"],
        n_simulations = cfg["n_simulations"],
        hidden_size   = cfg["hidden_size"],
        lr            = cfg["lr"],
        c_puct        = cfg["c_puct"],
    )

    t0 = time.time()
    rewards, p_losses, v_losses, eval_results = train_alphazero_quarto(
        env, agent,
        num_episodes = num_episodes,
        checkpoints  = checkpoints,
        evaluate_fn  = evaluate_no_training_quarto,
    )
    elapsed = time.time() - t0
    print(f"\n  Durée totale : {elapsed:.1f}s")

    print_table(eval_results, "Quarto")
    plot_and_save(rewards, p_losses, v_losses, eval_results, "Quarto")

    model_path = os.path.join(MODELS_DIR, "AlphaZero_Quarto.pt")
    agent.save(model_path)
    print(f"  → Modèle sauvegardé : {model_path}")

    save_results(eval_results, "Quarto", elapsed)
    return eval_results


# ══════════════════════════════════════════════════════════════════════════════
#  Sauvegarde JSON des résultats
# ══════════════════════════════════════════════════════════════════════════════

def save_results(eval_results, env_name, elapsed):
    data = {
        "env":          env_name,
        "agent":        "AlphaZero",
        "elapsed_s":    round(elapsed, 1),
        "checkpoints":  {
            str(k): v for k, v in eval_results.items()
        }
    }
    path = os.path.join(RESULTS_DIR, f"AlphaZero_{env_name}.json")
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  → Résultats JSON : {path}")


# ══════════════════════════════════════════════════════════════════════════════
#  Point d'entrée
# ══════════════════════════════════════════════════════════════════════════════

def build_checkpoints(num_episodes):
    """Génère des checkpoints à 1k, 10k, 100k, et au final."""
    candidates = [1_000, 10_000, 100_000, 1_000_000]
    cps = [c for c in candidates if c <= num_episodes]
    if num_episodes not in cps:
        cps.append(num_episodes)
    return sorted(set(cps))


def main():
    parser = argparse.ArgumentParser(description="Entraînement AlphaZero — tous environnements")
    parser.add_argument("--env",      default="all",
                        choices=["all", "lineworld", "gridworld", "tictactoe", "quarto"],
                        help="Environnement cible (default: all)")
    parser.add_argument("--episodes", type=int, default=10_000,
                        help="Nombre total d'épisodes d'entraînement (default: 10000)")
    args = parser.parse_args()

    checkpoints = build_checkpoints(args.episodes)
    print(f"\nCheckpoints : {checkpoints}")
    print(f"Épisodes    : {args.episodes:,}")

    all_results = {}

    if args.env in ("all", "lineworld"):
        all_results["LineWorld"] = run_lineworld(args.episodes, checkpoints)

    if args.env in ("all", "gridworld"):
        all_results["GridWorld"] = run_gridworld(args.episodes, checkpoints)

    if args.env in ("all", "tictactoe"):
        all_results["TicTacToe"] = run_tictactoe(args.episodes, checkpoints)

    if args.env in ("all", "quarto"):
        all_results["Quarto"] = run_quarto(args.episodes, checkpoints)

    # ── Résumé global ──────────────────────────────────────────────────────────
    if len(all_results) > 1:
        print("\n" + "═" * 70)
        print("  RÉSUMÉ GLOBAL — AlphaZero")
        print("═" * 70)
        last_ep = max(checkpoints)
        print(f"{'Environnement':>15} | {'Score final':>12} | {'Longueur':>10} | {'Temps/coup':>12}")
        print("-" * 60)
        for env_name, results in all_results.items():
            if last_ep in results:
                m = results[last_ep]
            else:
                m = list(results.values())[-1]
            print(f"{env_name:>15} | {m['score_moyen']:>12.4f} | {m['longueur_moy']:>10.2f} | {m['temps_coup_ms']:>11.4f}ms")
        print()


if __name__ == "__main__":
    main()
