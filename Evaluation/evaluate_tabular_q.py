"""
evaluate_tabular_q.py
=====================
Évaluation du TabularQAgent sur les 4 environnements après entraînement.

Charge la Q-table sauvegardée (fichier .pkl) et produit :
  - Score moyen, longueur moy, temps/coup (politique gloutonne, ε=0)
  - Distribution victoires / nuls / défaites (TicTacToe, Quarto)
  - Taux de succès (LineWorld, GridWorld)
  - Statistiques sur la Q-table (taille, couverture, valeurs max/min)
  - Tableau comparatif multi-environnements
  - Export JSON

Utilisation :
  python Evaluation/evaluate_tabular_q.py                       # tous les envs
  python Evaluation/evaluate_tabular_q.py --env tictactoe
  python Evaluation/evaluate_tabular_q.py --env all --n_games 1000
"""

import sys
import os
import argparse
import time
import json
import pickle

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Environnements.line_world import LineWorld
from Environnements.grid_world import GridWorld
from Environnements.tictactoe  import TicTacToe
from Environnements.quarto     import QuartoEnv
from Agents.tabular_q_agent    import TabularQAgent

BASE    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS  = os.path.join(BASE, "models",  "TabularQ")
RESULTS = os.path.join(BASE, "results", "TabularQ")
os.makedirs(RESULTS, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
#  Utilitaires
# ══════════════════════════════════════════════════════════════════════════════

def state_key(raw_state):
    if hasattr(raw_state, "tolist"):
        return tuple(raw_state.tolist())
    return tuple(raw_state)


def load_agent(model_path):
    """Charge la Q-table depuis un fichier pickle et reconstruit l'agent."""
    if not os.path.isfile(model_path):
        raise FileNotFoundError(
            f"Modèle introuvable : {model_path}\n"
            "Lancez d'abord : python Training/train_tabular_q.py"
        )
    with open(model_path, "rb") as f:
        data = pickle.load(f)
    agent = TabularQAgent(
        alpha         = data["alpha"],
        gamma         = data["gamma"],
        epsilon       = 0.0,          # politique gloutonne pour l'évaluation
        epsilon_decay = data["epsilon_decay"],
        epsilon_min   = data["epsilon_min"],
    )
    agent.q_table = data["q_table"]
    print(f"  ✓ Modèle chargé : {model_path}  (Q-table : {len(agent.q_table):,} états)")
    return agent


def q_table_stats(agent):
    """Retourne des statistiques descriptives sur la Q-table."""
    if not agent.q_table:
        return {}
    all_q = [q for state_q in agent.q_table.values() for q in state_q.values()]
    return {
        "n_etats":   len(agent.q_table),
        "n_entries": len(all_q),
        "q_max":     round(max(all_q), 4),
        "q_min":     round(min(all_q), 4),
        "q_mean":    round(float(np.mean(all_q)), 4),
        "q_std":     round(float(np.std(all_q)),  4),
    }


# ══════════════════════════════════════════════════════════════════════════════
#  Fonctions d'évaluation détaillées
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_lineworld(agent, env, n_games=500, max_steps=200):
    """Évaluation complète sur LineWorld (politique gloutonne)."""
    scores, lengths, times = [], [], []

    for _ in range(n_games):
        env.reset()
        steps = 0
        reward = 0
        while not env.done and steps < max_steps:
            s     = state_key(env.encode_state())
            avail = env.get_actions()
            t0    = time.perf_counter()
            a     = agent.choose_action(s, avail)
            times.append((time.perf_counter() - t0) * 1000)
            _, reward, _ = env.step(a)
            steps += 1
        scores.append(reward)
        lengths.append(steps)

    wins = sum(1 for s in scores if s > 0)
    return {
        "score_moyen":   round(float(np.mean(scores)),  4),
        "longueur_moy":  round(float(np.mean(lengths)), 2),
        "temps_coup_ms": round(float(np.mean(times)),   4),
        "taux_succes":   round(wins / n_games, 4),
        "taux_echec":    round(1 - wins / n_games, 4),
        "n_games":       n_games,
    }


def evaluate_gridworld(agent, env, n_games=500, max_steps=200):
    """Évaluation complète sur GridWorld (politique gloutonne)."""
    scores, lengths, times = [], [], []

    for _ in range(n_games):
        env.reset()
        steps = 0
        reward = 0
        while not env.done and steps < max_steps:
            s     = state_key(env.encode_state())
            avail = env.get_actions()
            t0    = time.perf_counter()
            a     = agent.choose_action(s, avail)
            times.append((time.perf_counter() - t0) * 1000)
            _, reward, _ = env.step(a)
            steps += 1
        scores.append(reward)
        lengths.append(steps)

    wins = sum(1 for s in scores if s > 0)
    return {
        "score_moyen":   round(float(np.mean(scores)),  4),
        "longueur_moy":  round(float(np.mean(lengths)), 2),
        "temps_coup_ms": round(float(np.mean(times)),   4),
        "taux_succes":   round(wins / n_games, 4),
        "taux_echec":    round(1 - wins / n_games, 4),
        "n_games":       n_games,
    }


def evaluate_tictactoe(agent, env, n_games=500):
    """Évaluation complète sur TicTacToe vs Random (politique gloutonne)."""
    scores, lengths, times = [], [], []
    wins = draws = losses = 0

    for _ in range(n_games):
        env.reset()
        steps  = 0
        result = 0
        while not env.done:
            s     = state_key(env.encode_state())
            avail = env.get_actions()
            t0    = time.perf_counter()
            a     = agent.choose_action(s, avail)
            times.append((time.perf_counter() - t0) * 1000)
            _, _, done, info = env.step(a)
            steps += 1
            if done:
                if   info["winner"] == 1:  result = 1;  wins   += 1
                elif info["winner"] == 0:  result = 0;  draws  += 1
                else:                      result = -1; losses += 1
                break
            if env.done:
                break
            rand_a = int(np.random.choice(env.get_actions()))
            _, _, done, info = env.step(rand_a)
            steps += 1
            if done:
                if   info["winner"] == -1: result = -1; losses += 1
                elif info["winner"] == 0:  result = 0;  draws  += 1
                else:                      result = 1;  wins   += 1
                break
        scores.append(result)
        lengths.append(steps)

    return {
        "score_moyen":   round(float(np.mean(scores)),  4),
        "longueur_moy":  round(float(np.mean(lengths)), 2),
        "temps_coup_ms": round(float(np.mean(times)),   4),
        "taux_victoire": round(wins   / n_games, 4),
        "taux_nul":      round(draws  / n_games, 4),
        "taux_defaite":  round(losses / n_games, 4),
        "n_games":       n_games,
    }


def evaluate_quarto(agent, env, n_games=500):
    """Évaluation complète sur Quarto vs Random (politique gloutonne)."""
    scores, lengths, times = [], [], []
    wins = draws = losses = 0

    for _ in range(n_games):
        env.reset()
        steps  = 0
        result = 0
        while not env.done:
            avail = env.get_actions()
            if not avail:
                break
            if env.current_player == 1:
                s  = state_key(env.encode_state())
                t0 = time.perf_counter()
                a  = agent.choose_action(s, avail)
                times.append((time.perf_counter() - t0) * 1000)
            else:
                a = int(np.random.choice(avail))
            _, _, done, _ = env.step(a)
            steps += 1
            if done:
                if   env.winner == 1: result = 1;  wins   += 1
                elif env.winner == 0: result = 0;  draws  += 1
                else:                 result = -1; losses += 1
                break
        scores.append(result)
        lengths.append(steps)

    return {
        "score_moyen":   round(float(np.mean(scores)),  4),
        "longueur_moy":  round(float(np.mean(lengths)), 2),
        "temps_coup_ms": round(float(np.mean(times if times else [0])), 4),
        "taux_victoire": round(wins   / n_games, 4),
        "taux_nul":      round(draws  / n_games, 4),
        "taux_defaite":  round(losses / n_games, 4),
        "n_games":       n_games,
    }


EVAL_FNS = {
    "lineworld": evaluate_lineworld,
    "gridworld": evaluate_gridworld,
    "tictactoe": evaluate_tictactoe,
    "quarto":    evaluate_quarto,
}


# ══════════════════════════════════════════════════════════════════════════════
#  Affichage console
# ══════════════════════════════════════════════════════════════════════════════

def print_metrics(m, env_label):
    print(f"\n  {'─' * 57}")
    print(f"  TabularQ  —  {env_label}  ({m['n_games']} parties, ε=0)")
    print(f"  {'─' * 57}")
    print(f"  Score moyen    : {m['score_moyen']:.4f}")
    print(f"  Longueur moy.  : {m['longueur_moy']:.2f} steps")
    print(f"  Temps/coup     : {m['temps_coup_ms']:.4f} ms")
    if "taux_succes" in m:
        print(f"  Taux de succès : {m['taux_succes'] * 100:.1f}%")
        print(f"  Taux d'échec   : {m['taux_echec']  * 100:.1f}%")
    if "taux_victoire" in m:
        print(f"  Taux victoire  : {m['taux_victoire'] * 100:.1f}%")
        print(f"  Taux nul       : {m['taux_nul']      * 100:.1f}%")
        print(f"  Taux défaite   : {m['taux_defaite']  * 100:.1f}%")


def print_q_stats(stats, env_label):
    print(f"\n  Q-table — {env_label}")
    print(f"  {'─' * 45}")
    print(f"  États distincts : {stats['n_etats']:>10,}")
    print(f"  Entrées totales : {stats['n_entries']:>10,}")
    print(f"  Q max           : {stats['q_max']:>10.4f}")
    print(f"  Q min           : {stats['q_min']:>10.4f}")
    print(f"  Q moyen         : {stats['q_mean']:>10.4f}")
    print(f"  Q écart-type    : {stats['q_std']:>10.4f}")


# ══════════════════════════════════════════════════════════════════════════════
#  Configurations
# ══════════════════════════════════════════════════════════════════════════════

ENV_CONFIGS = {
    "lineworld": {
        "label":  "LineWorld",
        "env_fn": lambda: LineWorld(size=6),
        "type":   "1player",
    },
    "gridworld": {
        "label":  "GridWorld (5×5)",
        "env_fn": lambda: GridWorld(rows=5, cols=5),
        "type":   "1player",
    },
    "tictactoe": {
        "label":  "TicTacToe (vs Random)",
        "env_fn": lambda: TicTacToe(),
        "type":   "tictactoe",
    },
    "quarto": {
        "label":  "Quarto (vs Random)",
        "env_fn": lambda: QuartoEnv(),
        "type":   "quarto",
    },
}

ENVS_LIST = ["lineworld", "gridworld", "tictactoe", "quarto"]


# ══════════════════════════════════════════════════════════════════════════════
#  Point d'entrée
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Évaluation TabularQ — 4 environnements"
    )
    parser.add_argument("--env",     default="all",
                        choices=["all"] + ENVS_LIST,
                        help="Environnement cible (default: all)")
    parser.add_argument("--n_games", type=int, default=500,
                        help="Parties d'évaluation (default: 500)")
    args = parser.parse_args()

    envs = ENVS_LIST if args.env == "all" else [args.env]

    print(f"\n{'=' * 70}")
    print(f"  ÉVALUATION TabularQ")
    print(f"  Environnements : {[ENV_CONFIGS[e]['label'] for e in envs]}")
    print(f"  Parties        : {args.n_games} par environnement")
    print(f"{'=' * 70}")

    all_metrics = {}

    for env_key in envs:
        ecfg  = ENV_CONFIGS[env_key]
        label = ecfg["label"]

        # Chargement du modèle
        model_path = os.path.join(MODELS, f"TabularQ_{label.split()[0]}.pkl")
        try:
            agent = load_agent(model_path)
        except FileNotFoundError as e:
            print(f"\n  ⚠ {e}")
            continue

        env     = ecfg["env_fn"]()
        eval_fn = EVAL_FNS[env_key]

        # Évaluation
        t0 = time.time()
        m  = eval_fn(agent, env, n_games=args.n_games)
        print(f"    Terminé en {time.time() - t0:.1f}s")

        print_metrics(m, label)

        # Statistiques Q-table
        stats = q_table_stats(agent)
        print_q_stats(stats, label)

        all_metrics[env_key] = {"metrics": m, "q_table_stats": stats}

    # Tableau comparatif
    if len(all_metrics) > 1:
        print(f"\n{'═' * 90}")
        print("  TABLEAU COMPARATIF — Score moyen  (politique gloutonne, ε=0)")
        print(f"{'═' * 90}")
        header = f"  {'Environnement':>25} | {'Score':>8} | {'Longueur':>10} | " \
                 f"{'ms/coup':>9} | {'Victoires':>10} | {'Q-table':>9}"
        print(header)
        print("  " + "─" * (len(header) - 2))
        for env_key in envs:
            if env_key not in all_metrics:
                continue
            m     = all_metrics[env_key]["metrics"]
            stats = all_metrics[env_key]["q_table_stats"]
            label = ENV_CONFIGS[env_key]["label"]
            win   = m.get("taux_victoire", m.get("taux_succes", float("nan")))
            print(f"  {label:>25} | {m['score_moyen']:>8.4f} | "
                  f"{m['longueur_moy']:>10.2f} | {m['temps_coup_ms']:>8.4f}ms | "
                  f"{win * 100:>9.1f}% | {stats.get('n_etats', 0):>9,}")

    # Sauvegarde JSON
    out_path = os.path.join(RESULTS, "evaluation_tabular_q.json")
    with open(out_path, "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"\n  → Résultats : {out_path}\n")


if __name__ == "__main__":
    main()
