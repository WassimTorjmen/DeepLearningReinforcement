"""
train_agents.py
===============
Entraînement de 5 agents sur les 4 environnements du projet :
  - RandomRollout         (pas d'entraînement — évaluation directe)
  - MCTS (UCT)            (pas d'entraînement — évaluation directe)
  - ExpertApprentice      (apprentissage supervisé depuis MCTS)
  - MuZero                (self-play + planification latente)
  - MuZeroStochastique    (MuZero + encodeur stochastique VAE)

Utilisation :
  python Training/train_agents.py --agent all --env all --episodes 10000
  python Training/train_agents.py --agent muzero --env tictactoe --episodes 50000
  python train_agents.py --agent randomrollout --env all
  python Training/train_agents.py --agent mcts --env all
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
from experiment import (
    train_alphazero_1player,
    train_alphazero_tictactoe,
    train_alphazero_quarto,
    evaluate_no_training_1player,
    evaluate_no_training_tictactoe,
    evaluate_no_training_quarto,
)

# ── Dossiers de sortie ─────────────────────────────────────────────────────────
BASE     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS   = os.path.join(BASE, "models")
RESULTS  = os.path.join(BASE, "results")
PLOTS    = os.path.join(BASE, "plots")
for d in [MODELS, RESULTS, PLOTS]:
    os.makedirs(d, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
#  Fonctions d'entraînement pour ExpertApprentice
#  (même logique que train_alphazero_* mais appelle select_action_expert)
# ══════════════════════════════════════════════════════════════════════════════

def _parse_loss(loss_result):
    if isinstance(loss_result, tuple):
        return loss_result[0], loss_result[1]
    return loss_result, None


def train_expert_1player(env, agent, num_episodes, checkpoints, evaluate_fn,
                          window=100, max_steps=200):
    all_policy_losses, all_rewards, eval_results = [], [], {}
    checkpoints    = sorted(checkpoints)
    next_check_idx = 0

    for episode in range(1, num_episodes + 1):
        env.reset()
        episode_reward = 0.0
        steps = 0

        while not env.done and steps < max_steps:
            available = env.get_actions()
            action    = agent.select_action_expert(env, available)
            _, reward, _ = env.step(action)
            episode_reward += reward
            steps += 1

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


def train_expert_tictactoe(env, agent, num_episodes, checkpoints, evaluate_fn, window=100):
    all_policy_losses, all_rewards, eval_results = [], [], {}
    checkpoints    = sorted(checkpoints)
    next_check_idx = 0

    for episode in range(1, num_episodes + 1):
        env.reset()
        episode_reward = 0.0

        while not env.done:
            available = env.get_actions()
            action    = agent.select_action_expert(env, available)
            _, _, done, info = env.step(action)
            if done:
                episode_reward += 1 if info["winner"] == 1 else 0
                break
            _, _, done, info = env.step(np.random.choice(env.get_actions()))
            if done:
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
    all_policy_losses, all_rewards, eval_results = [], [], {}
    checkpoints    = sorted(checkpoints)
    next_check_idx = 0

    for episode in range(1, num_episodes + 1):
        env.reset()
        episode_reward = 0.0

        while not env.done:
            available = env.get_actions()
            if env.current_player == 1:
                action = agent.select_action_expert(env, available)
            else:
                action = int(np.random.choice(available))
            _, _, done, _ = env.step(action)
            if done:
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
#  Évaluation générique (compatible avec agents sans réseau et avec réseau)
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_agent_1player(env, agent, n_games=500, max_steps=200):
    """Évalue n'importe quel agent sur LineWorld / GridWorld."""
    scores, lengths, times = [], [], []

    for _ in range(n_games):
        env.reset()
        steps = 0
        reward = 0

        while not env.done and steps < max_steps:
            available = env.get_actions()
            t0 = time.perf_counter()
            action = agent.select_action(env, available)
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000)
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
    """Évalue n'importe quel agent sur TicTacToe vs Random."""
    scores, lengths, times = [], [], []

    for _ in range(n_games):
        env.reset()
        steps = 0

        while not env.done:
            available = env.get_actions()
            t0 = time.perf_counter()
            action = agent.select_action(env, available)
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000)
            _, _, done, info = env.step(action)
            steps += 1
            if done:
                scores.append(1 if info["winner"] == 1 else (0 if info["winner"] == 0 else -1))
                lengths.append(steps)
                break
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
    """Évalue n'importe quel agent sur Quarto vs Random."""
    scores, lengths, times = [], [], []

    for _ in range(n_games):
        env.reset()
        steps = 0

        while not env.done:
            available = env.get_actions()
            if env.current_player == 1:
                t0 = time.perf_counter()
                action = agent.select_action(env, available)
                t1 = time.perf_counter()
                times.append((t1 - t0) * 1000)
            else:
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
#  Configurations
# ══════════════════════════════════════════════════════════════════════════════

ENV_CONFIGS = {
    "lineworld": {
        "env_fn":      lambda: LineWorld(size=6),
        "state_size":  8,
        "num_actions": 2,
        "train_fn":    "1player",
        "eval_fn":     evaluate_agent_1player,
        "label":       "LineWorld",
    },
    "gridworld": {
        "env_fn":      lambda: GridWorld(rows=5, cols=5),
        "state_size":  31,
        "num_actions": 4,
        "train_fn":    "1player",
        "eval_fn":     evaluate_agent_1player,
        "label":       "GridWorld",
    },
    "tictactoe": {
        "env_fn":      lambda: TicTacToe(),
        "state_size":  27,
        "num_actions": 9,
        "train_fn":    "tictactoe",
        "eval_fn":     evaluate_agent_tictactoe,
        "label":       "TicTacToe",
    },
    "quarto": {
        "env_fn":      lambda: QuartoEnv(),
        "state_size":  105,
        "num_actions": 32,
        "train_fn":    "quarto",
        "eval_fn":     evaluate_agent_quarto,
        "label":       "Quarto",
    },
}

AGENT_CONFIGS = {
    "randomrollout": {
        "label":       "RandomRollout",
        "no_training": True,
        "fn":          lambda s, a: RandomRolloutAgent(n_rollouts=50),
    },
    "mcts": {
        "label":       "MCTS_UCT",
        "no_training": True,
        "fn":          lambda s, a: MCTSAgent(n_simulations=100),
    },
    "expertapprentice": {
        "label":       "ExpertApprentice",
        "no_training": False,
        "train_mode":  "expert",
        "fn":          lambda s, a: ExpertApprenticeAgent(
            state_size=s, num_actions=a, n_simulations=50, hidden_size=128, lr=1e-3
        ),
    },
    "muzero": {
        "label":       "MuZero",
        "no_training": False,
        "train_mode":  "alphazero",   # même interface que alphazero
        "fn":          lambda s, a: MuZeroAgent(
            state_size=s, num_actions=a, hidden_size=128, n_simulations=10, lr=1e-3
        ),
    },
    "muzero_stochastic": {
        "label":       "MuZeroStochastic",
        "no_training": False,
        "train_mode":  "alphazero",
        "fn":          lambda s, a: MuZeroStochasticAgent(
            state_size=s, num_actions=a, hidden_size=128, chance_size=8,
            n_simulations=10, lr=1e-3, kl_weight=0.1
        ),
    },
    "alphazero": {
        "label":       "AlphaZero",
        "no_training": False,
        "train_mode":  "alphazero",
        "az_configs": {
            # Hyperparamètres spécifiques à chaque env (surchargent fn)
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
#  Helpers graphiques
# ══════════════════════════════════════════════════════════════════════════════

def smooth(data, w=100):
    return [np.mean(data[max(0, i - w): i + 1]) for i in range(len(data))]


def save_plot(rewards, losses, eval_results, agent_label, env_label, plots_dir, window=100):
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    fig.suptitle(f"{agent_label} — {env_label}", fontsize=13, fontweight="bold")

    axes[0].plot(smooth(rewards, window), color="#2196F3", linewidth=1.2)
    if eval_results:
        xs = sorted(eval_results.keys())
        ys = [eval_results[x]["score_moyen"] for x in xs]
        ax2 = axes[0].twinx()
        ax2.plot(xs, ys, "o--", color="#FF5722", markersize=6, label="Score éval.")
        ax2.set_ylabel("Score éval.", color="#FF5722")
        ax2.legend(loc="lower right", fontsize=9)
    axes[0].set_ylabel(f"Reward lissé ({window} ep)")
    axes[0].set_xlabel("Épisode")
    axes[0].set_title("Reward d'entraînement")
    axes[0].grid(True, alpha=0.3)

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
#  Dispatcher d'entraînement
# ══════════════════════════════════════════════════════════════════════════════

TRAIN_FNS = {
    "1player":   {
        "expert":    train_expert_1player,
        "alphazero": train_alphazero_1player,
    },
    "tictactoe": {
        "expert":    train_expert_tictactoe,
        "alphazero": train_alphazero_tictactoe,
    },
    "quarto":    {
        "expert":    train_expert_quarto,
        "alphazero": train_alphazero_quarto,
    },
}

EVAL_FNS_NO_TRAINING = {
    "1player":   evaluate_no_training_1player,
    "tictactoe": evaluate_no_training_tictactoe,
    "quarto":    evaluate_no_training_quarto,
}


def run_one(agent_key, env_key, num_episodes, checkpoints, n_eval=500):
    ecfg  = ENV_CONFIGS[env_key]
    acfg  = AGENT_CONFIGS[agent_key]
    label = acfg["label"]
    elabel = ecfg["label"]

    agent_dir  = os.path.join(MODELS,  label)
    result_dir = os.path.join(RESULTS, label)
    plot_dir   = os.path.join(PLOTS,   label)
    for d in [agent_dir, result_dir, plot_dir]:
        os.makedirs(d, exist_ok=True)

    env   = ecfg["env_fn"]()

    # AlphaZero : hyperparamètres différents selon l'environnement
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
        agent = acfg["fn"](ecfg["state_size"], ecfg["num_actions"])

    eval_fn = ecfg["eval_fn"]

    print(f"\n{'═' * 70}")
    print(f"  {label}  —  {elabel}")
    print(f"{'═' * 70}")

    t0 = time.time()

    # ── Agents sans entraînement ────────────────────────────────────────────
    if acfg["no_training"]:
        print("  (Pas d'entraînement — évaluation directe)\n")
        eval_results = {}
        for cp in checkpoints:
            # On évalue plusieurs fois pour simuler les checkpoints
            m = eval_fn(env, agent, n_games=n_eval)
            eval_results[cp] = m
            print(f"  {cp:>9,} épisodes | Score : {m['score_moyen']:.4f} | "
                  f"Longueur : {m['longueur_moy']:.2f} | Temps/coup : {m['temps_coup_ms']:.4f} ms")
        rewards = [0.0]
        losses  = [0.0]

    # ── Agents avec entraînement ────────────────────────────────────────────
    else:
        train_mode = acfg["train_mode"]
        train_fn   = TRAIN_FNS[ecfg["train_fn"]][train_mode]

        # ExpertApprentice utilise son propre evaluate (basé sur le réseau apprenti)
        if train_mode == "expert":
            train_eval_fn = eval_fn
        else:
            train_eval_fn = EVAL_FNS_NO_TRAINING[ecfg["train_fn"]]

        kwargs = dict(
            env          = env,
            agent        = agent,
            num_episodes = num_episodes,
            checkpoints  = checkpoints,
            evaluate_fn  = train_eval_fn,
        )
        if ecfg["train_fn"] == "1player":
            kwargs["max_steps"] = 200

        rewards, losses, _, eval_results = train_fn(**kwargs)

    elapsed = time.time() - t0
    print(f"\n  Durée : {elapsed:.1f}s ({elapsed / 60:.1f} min)")

    print_table(eval_results, label, elabel)

    if len(rewards) > 1:
        save_plot(rewards, losses, eval_results, label, elabel, plot_dir)

    # ── Sauvegarde modèle ──────────────────────────────────────────────────
    model_path = os.path.join(agent_dir, f"{label}_{elabel}.pt")
    agent.save(model_path)
    print(f"  → Modèle : {model_path}")

    # ── JSON résultats ─────────────────────────────────────────────────────
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
#  Point d'entrée
# ══════════════════════════════════════════════════════════════════════════════

AGENTS_LIST = ["randomrollout", "mcts", "expertapprentice", "muzero", "muzero_stochastic", "alphazero"]
ENVS_LIST   = ["lineworld", "gridworld", "tictactoe", "quarto"]


def build_checkpoints(num_episodes):
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

    for agent_key in agents:
        all_results[agent_key] = {}
        for env_key in envs:
            result = run_one(agent_key, env_key, args.episodes, checkpoints, args.n_eval)
            all_results[agent_key][env_key] = result

    # ── Résumé global ──────────────────────────────────────────────────────
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