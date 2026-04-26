"""
evaluate_agents.py
==================
Évaluation des 5 agents sur les 4 environnements après entraînement.

Charge les modèles sauvegardés et produit :
  - Score moyen, longueur moy, temps/coup
  - Distribution victoires / nuls / défaites (jeux à 2 joueurs)
  - Tableau comparatif final
  - Export JSON

Utilisation :
  python Evaluation/evaluate_agents.py
  python Evaluation/evaluate_agents.py --agent mcts --env tictactoe --n_games 1000
  python Evaluation/evaluate_agents.py --agent randomrollout --env all
"""

import sys
import os
import argparse
import time
import json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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

BASE     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS   = os.path.join(BASE, "models")
RESULTS  = os.path.join(BASE, "results")
os.makedirs(RESULTS, exist_ok=True)

# ══════════════════════════════════════════════════════════════════════════════
#  Configurations
# ══════════════════════════════════════════════════════════════════════════════

AGENT_CONFIGS = {
    "randomrollout": {
        "label":    "RandomRollout",
        "has_model": False,
        "fn":       lambda s, a: RandomRolloutAgent(n_rollouts=50),
    },
    "mcts": {
        "label":    "MCTS_UCT",
        "has_model": False,
        "fn":       lambda s, a: MCTSAgent(n_simulations=100),
    },
    "expertapprentice": {
        "label":    "ExpertApprentice",
        "has_model": True,
        "fn":       lambda s, a: ExpertApprenticeAgent(
            state_size=s, num_actions=a, n_simulations=50, hidden_size=128, lr=1e-3
        ),
    },
    "muzero": {
        "label":    "MuZero",
        "has_model": True,
        "fn":       lambda s, a: MuZeroAgent(
            state_size=s, num_actions=a, hidden_size=128, n_simulations=10, lr=1e-3
        ),
    },
    "muzero_stochastic": {
        "label":    "MuZeroStochastic",
        "has_model": True,
        "fn":       lambda s, a: MuZeroStochasticAgent(
            state_size=s, num_actions=a, hidden_size=128, chance_size=8,
            n_simulations=10, lr=1e-3
        ),
    },
    "alphazero": {
        "label":    "AlphaZero",
        "has_model": True,
        # Hyperparamètres spécifiques par environnement (même valeurs que train_alphazero.py)
        "az_configs": {
            "lineworld": {"hidden_size": 64,  "n_simulations": 20, "c_puct": 1.0},
            "gridworld": {"hidden_size": 128, "n_simulations": 20, "c_puct": 1.0},
            "tictactoe": {"hidden_size": 128, "n_simulations": 50, "c_puct": 1.5},
            "quarto":    {"hidden_size": 256, "n_simulations": 30, "c_puct": 1.5},
        },
        # fn générique (surchargé dans load_agent pour alphazero)
        "fn":       lambda s, a: AlphaZeroAgent(
            state_size=s, num_actions=a, hidden_size=128, n_simulations=20,
        ),
    },
}

ENV_CONFIGS = {
    "lineworld": {
        "label":      "LineWorld",
        "state_size": 8,
        "num_actions": 2,
        "type":       "1player",
        "env_fn":     lambda: LineWorld(size=6),
    },
    "gridworld": {
        "label":      "GridWorld",
        "state_size": 31,
        "num_actions": 4,
        "type":       "1player",
        "env_fn":     lambda: GridWorld(rows=5, cols=5),
    },
    "tictactoe": {
        "label":      "TicTacToe",
        "state_size": 27,
        "num_actions": 9,
        "type":       "tictactoe",
        "env_fn":     lambda: TicTacToe(),
    },
    "quarto": {
        "label":      "Quarto",
        "state_size": 105,
        "num_actions": 32,
        "type":       "quarto",
        "env_fn":     lambda: QuartoEnv(),
    },
}


# ══════════════════════════════════════════════════════════════════════════════
#  Fonctions d'évaluation détaillées
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_1player(agent, env, n_games=500, max_steps=200, use_mcts=False):
    scores, lengths, times = [], [], []

    for _ in range(n_games):
        env.reset()
        steps = 0
        reward = 0

        while not env.done and steps < max_steps:
            available = env.get_actions()
            t0 = time.perf_counter()
            if use_mcts and hasattr(agent, "select_action_mcts"):
                action = agent.select_action_mcts(env, available)
                if hasattr(agent, "states_buffer"):
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

    wins = sum(1 for s in scores if s > 0)
    return {
        "score_moyen":   round(float(np.mean(scores)), 4),
        "longueur_moy":  round(float(np.mean(lengths)), 2),
        "temps_coup_ms": round(float(np.mean(times)), 4),
        "taux_victoire": round(wins / n_games, 4),
        "taux_echec":    round(1 - wins / n_games, 4),
        "n_games":       n_games,
    }


def evaluate_tictactoe(agent, env, n_games=500, use_mcts=False):
    scores, lengths, times = [], [], []
    wins = draws = losses = 0

    for _ in range(n_games):
        env.reset()
        steps = 0
        result = 0

        while not env.done:
            available = env.get_actions()
            t0 = time.perf_counter()
            if use_mcts and hasattr(agent, "select_action_mcts"):
                action = agent.select_action_mcts(env, available)
                if hasattr(agent, "states_buffer"):
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
        "score_moyen":   round(float(np.mean(scores)), 4),
        "longueur_moy":  round(float(np.mean(lengths)), 2),
        "temps_coup_ms": round(float(np.mean(times)), 4),
        "taux_victoire": round(wins   / n_games, 4),
        "taux_nul":      round(draws  / n_games, 4),
        "taux_defaite":  round(losses / n_games, 4),
        "n_games":       n_games,
    }


def evaluate_quarto(agent, env, n_games=500, use_mcts=False):
    scores, lengths, times = [], [], []
    wins = draws = losses = 0

    for _ in range(n_games):
        env.reset()
        steps = 0
        result = 0

        while not env.done:
            available = env.get_actions()
            if env.current_player == 1:
                t0 = time.perf_counter()
                if use_mcts and hasattr(agent, "select_action_mcts"):
                    action = agent.select_action_mcts(env, available)
                    if hasattr(agent, "states_buffer"):
                        agent.states_buffer.clear()
                        agent.pi_buffer.clear()
                else:
                    action = agent.select_action(env, available)
                t1 = time.perf_counter()
                times.append((t1 - t0) * 1000)
            else:
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
    }


EVAL_FNS = {
    "1player":  evaluate_1player,
    "tictactoe": evaluate_tictactoe,
    "quarto":    evaluate_quarto,
}


# ══════════════════════════════════════════════════════════════════════════════
#  Chargement de modèle
# ══════════════════════════════════════════════════════════════════════════════

def load_agent(agent_key, env_key):
    acfg  = AGENT_CONFIGS[agent_key]
    ecfg  = ENV_CONFIGS[env_key]

    # AlphaZero : instanciation avec hyperparamètres propres à l'environnement
    if agent_key == "alphazero":
        az_cfg = acfg["az_configs"][env_key]
        agent  = AlphaZeroAgent(
            state_size    = ecfg["state_size"],
            num_actions   = ecfg["num_actions"],
            hidden_size   = az_cfg["hidden_size"],
            n_simulations = az_cfg["n_simulations"],
            c_puct        = az_cfg["c_puct"],
        )
    else:
        agent = acfg["fn"](ecfg["state_size"], ecfg["num_actions"])

    if acfg["has_model"]:
        # AlphaZero sauvegarde dans models/alphazero/
        if agent_key == "alphazero":
            model_path = os.path.join(MODELS, "alphazero",
                                      f"AlphaZero_{ecfg['label']}.pt")
        else:
            model_path = os.path.join(MODELS, acfg["label"],
                                      f"{acfg['label']}_{ecfg['label']}.pt")

        if os.path.isfile(model_path):
            agent.load(model_path)
            print(f"  ✓ Modèle chargé : {model_path}")
        else:
            print(f"  ⚠ Modèle introuvable : {model_path}  (réseau non entraîné)")

    return agent


# ══════════════════════════════════════════════════════════════════════════════
#  Affichage
# ══════════════════════════════════════════════════════════════════════════════

def print_metrics(m, agent_label, env_label):
    print(f"\n  {'─' * 55}")
    print(f"  {agent_label}  —  {env_label}  ({m['n_games']} parties)")
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
#  Point d'entrée
# ══════════════════════════════════════════════════════════════════════════════

AGENTS_LIST = ["randomrollout", "mcts", "expertapprentice", "muzero", "muzero_stochastic", "alphazero"]
ENVS_LIST   = ["lineworld", "gridworld", "tictactoe", "quarto"]


def main():
    parser = argparse.ArgumentParser(description="Évaluation des 5 agents")
    parser.add_argument("--agent",   default="all", choices=["all"] + AGENTS_LIST)
    parser.add_argument("--env",     default="all", choices=["all"] + ENVS_LIST)
    parser.add_argument("--n_games", type=int, default=500,
                        help="Parties d'évaluation (default: 500)")
    parser.add_argument("--use_mcts", action="store_true",
                        help="Pour AlphaZero : utiliser MCTS pendant l'éval (plus lent, plus fort)")
    args = parser.parse_args()

    agents = AGENTS_LIST if args.agent == "all" else [args.agent]
    envs   = ENVS_LIST   if args.env   == "all" else [args.env]

    print(f"\n{'=' * 70}")
    print(f"  ÉVALUATION — {len(agents)} agents × {len(envs)} environnements")
    print(f"  Parties : {args.n_games} par combinaison")
    if args.use_mcts:
        print(f"  Mode AlphaZero : MCTS (lent mais plus fort)")
    print(f"{'=' * 70}")

    all_metrics = {}

    for agent_key in agents:
        all_metrics[agent_key] = {}
        for env_key in envs:
            ecfg  = ENV_CONFIGS[env_key]
            acfg  = AGENT_CONFIGS[agent_key]
            agent = load_agent(agent_key, env_key)
            env   = ecfg["env_fn"]()
            eval_fn = EVAL_FNS[ecfg["type"]]

            # Pour AlphaZero, les fonctions d'éval acceptent use_mcts
            if agent_key == "alphazero":
                m = eval_fn(agent, env, n_games=args.n_games, use_mcts=args.use_mcts)
            else:
                m = eval_fn(agent, env, n_games=args.n_games)
            print_metrics(m, acfg["label"], ecfg["label"])
            all_metrics[agent_key][env_key] = m

    # ── Tableau comparatif ─────────────────────────────────────────────────
    print(f"\n{'═' * 90}")
    print("  TABLEAU COMPARATIF — Score moyen")
    print(f"{'═' * 90}")
    header = f"{'Agent':>22}" + "".join(f" | {ENV_CONFIGS[e]['label']:>12}" for e in envs)
    print(header)
    print("─" * len(header))
    for agent_key in agents:
        row = f"{AGENT_CONFIGS[agent_key]['label']:>22}"
        for env_key in envs:
            m = all_metrics[agent_key].get(env_key)
            row += f" | {m['score_moyen']:>12.4f}" if m else f" | {'N/A':>12}"
        print(row)

    print(f"\n{'═' * 90}")
    print("  TABLEAU COMPARATIF — Temps/coup (ms)")
    print(f"{'═' * 90}")
    print(header)
    print("─" * len(header))
    for agent_key in agents:
        row = f"{AGENT_CONFIGS[agent_key]['label']:>22}"
        for env_key in envs:
            m = all_metrics[agent_key].get(env_key)
            row += f" | {m['temps_coup_ms']:>12.4f}" if m else f" | {'N/A':>12}"
        print(row)
    print()

    # ── Sauvegarde JSON ────────────────────────────────────────────────────
    out_path = os.path.join(RESULTS, "comparison_agents.json")
    with open(out_path, "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"  → Résultats : {out_path}")


if __name__ == "__main__":
    main()