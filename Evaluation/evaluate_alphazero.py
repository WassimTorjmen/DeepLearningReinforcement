"""
evaluate_alphazero.py
=====================
Tests & évaluation d'AlphaZero sur les 4 environnements.

Charge les modèles sauvegardés et mesure :
  - Score moyen sur N parties
  - Longueur moyenne d'une partie
  - Temps moyen par coup (ms)
  - Distribution victoires / défaites / nuls (envs multi-joueurs)

Utilisation :
  python Evaluation/evaluate_alphazero.py
  python Evaluation/evaluate_alphazero.py --env tictactoe --n_games 1000
  python Evaluation/evaluate_alphazero.py --models_dir /path/to/models
"""

import sys
import os
import argparse
import time
import json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Environnements.line_world import LineWorld
from Environnements.grid_world import GridWorld
from Environnements.tictactoe  import TicTacToe
from Environnements.quarto     import QuartoEnv
from Agents.Alpha_zero         import AlphaZeroAgent

MODELS_DIR  = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "alphazero")
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "alphazero")
os.makedirs(RESULTS_DIR, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
#  Configurations (doit correspondre à train_alphazero.py)
# ══════════════════════════════════════════════════════════════════════════════

CONFIGS = {
    "lineworld": {"state_size": 8,   "num_actions": 2,  "hidden_size": 64,  "n_simulations": 20},
    "gridworld": {"state_size": 31,  "num_actions": 4,  "hidden_size": 128, "n_simulations": 20},
    "tictactoe": {"state_size": 27,  "num_actions": 9,  "hidden_size": 128, "n_simulations": 50},
    "quarto":    {"state_size": 105, "num_actions": 32, "hidden_size": 256, "n_simulations": 30},
}


# ══════════════════════════════════════════════════════════════════════════════
#  Fonctions d'évaluation détaillées
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_lineworld(agent, n_games=500, max_steps=200, use_mcts=False):
    """Évalue AlphaZero sur LineWorld.
    use_mcts=True → utilise MCTS à chaque coup (lent mais plus fort)
    use_mcts=False → utilise directement le réseau (rapide)
    """
    env = LineWorld(size=6)
    scores, lengths, times = [], [], []

    for _ in range(n_games):
        env.reset()
        steps = 0
        reward = 0

        while not env.done and steps < max_steps:
            available = env.get_actions()
            t0 = time.perf_counter()
            if use_mcts:
                action = agent.select_action_mcts(env, available)
                agent.states_buffer.clear()
                agent.pi_buffer.clear()
            else:
                action = agent.select_action(env, available)
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000)
            _, reward, _ = env.step(action)
            steps += 1

        scores.append(reward)
        lengths.append(steps)

    wins  = sum(1 for s in scores if s > 0)
    return {
        "score_moyen":   round(float(np.mean(scores)), 4),
        "longueur_moy":  round(float(np.mean(lengths)), 2),
        "temps_coup_ms": round(float(np.mean(times)), 4),
        "taux_victoire": round(wins / n_games, 4),
        "n_games":       n_games,
        "mode":          "mcts" if use_mcts else "network",
    }


def evaluate_gridworld(agent, n_games=500, max_steps=200, use_mcts=False):
    """Évalue AlphaZero sur GridWorld."""
    env = GridWorld(rows=5, cols=5)
    scores, lengths, times = [], [], []

    for _ in range(n_games):
        env.reset()
        steps = 0
        reward = 0

        while not env.done and steps < max_steps:
            available = env.get_actions()
            t0 = time.perf_counter()
            if use_mcts:
                action = agent.select_action_mcts(env, available)
                agent.states_buffer.clear()
                agent.pi_buffer.clear()
            else:
                action = agent.select_action(env, available)
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000)
            _, reward, _ = env.step(action)
            steps += 1

        scores.append(reward)
        lengths.append(steps)

    wins  = sum(1 for s in scores if s > 0)
    traps = sum(1 for s in scores if s < 0)
    return {
        "score_moyen":   round(float(np.mean(scores)), 4),
        "longueur_moy":  round(float(np.mean(lengths)), 2),
        "temps_coup_ms": round(float(np.mean(times)), 4),
        "taux_victoire": round(wins  / n_games, 4),
        "taux_piege":    round(traps / n_games, 4),
        "n_games":       n_games,
        "mode":          "mcts" if use_mcts else "network",
    }


def evaluate_tictactoe(agent, n_games=500, use_mcts=False):
    """Évalue AlphaZero sur TicTacToe vs Random."""
    env = TicTacToe()
    scores, lengths, times = [], [], []
    wins = draws = losses = 0

    for _ in range(n_games):
        env.reset()
        steps = 0
        result = 0

        while not env.done:
            available = env.get_actions()

            # Tour AlphaZero (joueur 1)
            t0 = time.perf_counter()
            if use_mcts:
                action = agent.select_action_mcts(env, available)
                agent.states_buffer.clear()
                agent.pi_buffer.clear()
            else:
                action = agent.select_action(env, available)
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000)
            _, _, done, info = env.step(action)
            steps += 1
            if done:
                if info["winner"] == 1:   result = 1;  wins   += 1
                elif info["winner"] == 0: result = 0;  draws  += 1
                else:                     result = -1; losses += 1
                break

            # Tour Random (joueur -1)
            _, _, done, info = env.step(np.random.choice(env.get_actions()))
            steps += 1
            if done:
                if info["winner"] == -1:  result = -1; losses += 1
                elif info["winner"] == 0: result = 0;  draws  += 1
                else:                     result = 1;  wins   += 1
                break

        scores.append(result)
        lengths.append(steps)

    return {
        "score_moyen":    round(float(np.mean(scores)), 4),
        "longueur_moy":   round(float(np.mean(lengths)), 2),
        "temps_coup_ms":  round(float(np.mean(times)), 4),
        "taux_victoire":  round(wins   / n_games, 4),
        "taux_nul":       round(draws  / n_games, 4),
        "taux_defaite":   round(losses / n_games, 4),
        "n_games":        n_games,
        "mode":           "mcts" if use_mcts else "network",
    }


def evaluate_quarto(agent, n_games=500, use_mcts=False):
    """Évalue AlphaZero sur Quarto vs Random."""
    env = QuartoEnv()
    scores, lengths, times = [], [], []
    wins = draws = losses = 0

    for _ in range(n_games):
        env.reset()
        steps = 0
        result = 0

        while not env.done:
            available = env.get_actions()

            if env.current_player == 1:
                # Tour AlphaZero
                t0 = time.perf_counter()
                if use_mcts:
                    action = agent.select_action_mcts(env, available)
                    agent.states_buffer.clear()
                    agent.pi_buffer.clear()
                else:
                    action = agent.select_action(env, available)
                t1 = time.perf_counter()
                times.append((t1 - t0) * 1000)
            else:
                # Tour Random
                action = int(np.random.choice(available))

            _, _, done, _ = env.step(action)
            steps += 1
            if done:
                if env.winner == 1:   result = 1;  wins   += 1
                elif env.winner == 0: result = 0;  draws  += 1
                else:                 result = -1; losses += 1
                break

        scores.append(result)
        lengths.append(steps)

    return {
        "score_moyen":   round(float(np.mean(scores)), 4),
        "longueur_moy":  round(float(np.mean(lengths)), 2),
        "temps_coup_ms": round(float(np.mean(times)), 4),
        "taux_victoire": round(wins   / n_games, 4),
        "taux_nul":      round(draws  / n_games, 4),
        "taux_defaite":  round(losses / n_games, 4),
        "n_games":       n_games,
        "mode":          "mcts" if use_mcts else "network",
    }


# ══════════════════════════════════════════════════════════════════════════════
#  Affichage des résultats
# ══════════════════════════════════════════════════════════════════════════════

def print_results(metrics, env_name):
    print()
    print("─" * 60)
    print(f"  AlphaZero — {env_name}  [{metrics['mode']}]  ({metrics['n_games']} parties)")
    print("─" * 60)
    print(f"  Score moyen    : {metrics['score_moyen']:.4f}")
    print(f"  Longueur moy.  : {metrics['longueur_moy']:.2f} steps")
    print(f"  Temps/coup     : {metrics['temps_coup_ms']:.4f} ms")
    if "taux_victoire" in metrics:
        print(f"  Taux victoire  : {metrics['taux_victoire'] * 100:.1f}%")
    if "taux_nul" in metrics:
        print(f"  Taux nul       : {metrics['taux_nul'] * 100:.1f}%")
    if "taux_defaite" in metrics:
        print(f"  Taux défaite   : {metrics['taux_defaite'] * 100:.1f}%")
    if "taux_piege" in metrics:
        print(f"  Taux piège     : {metrics['taux_piege'] * 100:.1f}%")
    print()


# ══════════════════════════════════════════════════════════════════════════════
#  Chargement du modèle
# ══════════════════════════════════════════════════════════════════════════════

def load_agent(env_key, models_dir):
    cfg  = CONFIGS[env_key]
    name = env_key.capitalize() if env_key != "tictactoe" else "TicTacToe"
    name_map = {
        "lineworld": "LineWorld",
        "gridworld": "GridWorld",
        "tictactoe": "TicTacToe",
        "quarto":    "Quarto",
    }
    model_file = os.path.join(models_dir, f"AlphaZero_{name_map[env_key]}.pt")

    agent = AlphaZeroAgent(
        state_size    = cfg["state_size"],
        num_actions   = cfg["num_actions"],
        n_simulations = cfg["n_simulations"],
        hidden_size   = cfg["hidden_size"],
    )

    if os.path.isfile(model_file):
        agent.load(model_file)
        print(f"  ✓ Modèle chargé : {model_file}")
    else:
        print(f"  ⚠ Modèle introuvable : {model_file}")
        print(f"    → Évaluation avec un réseau non entraîné (random)")

    return agent


# ══════════════════════════════════════════════════════════════════════════════
#  Point d'entrée
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Évaluation AlphaZero")
    parser.add_argument("--env",        default="all",
                        choices=["all", "lineworld", "gridworld", "tictactoe", "quarto"])
    parser.add_argument("--n_games",    type=int, default=500, help="Parties d'évaluation")
    parser.add_argument("--use_mcts",   action="store_true",
                        help="Utiliser MCTS pendant l'évaluation (plus lent, plus fort)")
    parser.add_argument("--models_dir", default=MODELS_DIR,
                        help="Dossier contenant les modèles .pt")
    args = parser.parse_args()

    mode_label = "MCTS" if args.use_mcts else "Réseau seul"
    print(f"\n{'=' * 60}")
    print(f"  ÉVALUATION AlphaZero  —  mode: {mode_label}")
    print(f"  Parties : {args.n_games}")
    print(f"{'=' * 60}")

    all_metrics = {}

    if args.env in ("all", "lineworld"):
        agent = load_agent("lineworld", args.models_dir)
        m = evaluate_lineworld(agent, args.n_games, use_mcts=args.use_mcts)
        print_results(m, "LineWorld")
        all_metrics["LineWorld"] = m

    if args.env in ("all", "gridworld"):
        agent = load_agent("gridworld", args.models_dir)
        m = evaluate_gridworld(agent, args.n_games, use_mcts=args.use_mcts)
        print_results(m, "GridWorld")
        all_metrics["GridWorld"] = m

    if args.env in ("all", "tictactoe"):
        agent = load_agent("tictactoe", args.models_dir)
        m = evaluate_tictactoe(agent, args.n_games, use_mcts=args.use_mcts)
        print_results(m, "TicTacToe")
        all_metrics["TicTacToe"] = m

    if args.env in ("all", "quarto"):
        agent = load_agent("quarto", args.models_dir)
        m = evaluate_quarto(agent, args.n_games, use_mcts=args.use_mcts)
        print_results(m, "Quarto")
        all_metrics["Quarto"] = m

    # ── Sauvegarde JSON ────────────────────────────────────────────────────────
    out_path = os.path.join(RESULTS_DIR, "AlphaZero_evaluation.json")
    with open(out_path, "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"  → Résultats sauvegardés : {out_path}")

    # ── Résumé global ──────────────────────────────────────────────────────────
    if len(all_metrics) > 1:
        print(f"\n{'═' * 70}")
        print("  RÉCAPITULATIF")
        print(f"{'═' * 70}")
        print(f"{'Environnement':>15} | {'Score':>8} | {'Victoires':>10} | {'Longueur':>9} | {'ms/coup':>8}")
        print("─" * 70)
        for name, m in all_metrics.items():
            vic = f"{m.get('taux_victoire', 0) * 100:.1f}%"
            print(f"{name:>15} | {m['score_moyen']:>8.4f} | {vic:>10} | {m['longueur_moy']:>9.2f} | {m['temps_coup_ms']:>8.4f}")
        print()


if __name__ == "__main__":
    main()
