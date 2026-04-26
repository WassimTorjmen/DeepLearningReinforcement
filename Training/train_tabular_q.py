"""
train_tabular_q.py
==================
Entraînement du TabularQAgent (Q-Learning tabulaire) sur les 4 environnements :
  - LineWorld
  - GridWorld
  - TicTacToe (vs joueur aléatoire)
  - Quarto    (vs joueur aléatoire)

Remarque : TabularQ n'est applicable que si l'espace d'états est discret et fini.
  - LineWorld / GridWorld / TicTacToe : compatible (états one-hot, Q-table de taille raisonnable)
  - Quarto : applicable mais Q-table potentiellement très grande (documenté en sortie)

La conversion état → clé se fait via tuple(state.tolist()) car encode_state() retourne
un numpy.ndarray et la Q-table est un dict Python qui requiert des clés hashables.

Utilisation :
  python Training/train_tabular_q.py                                 # tous les envs
  python Training/train_tabular_q.py --env tictactoe --episodes 50000
  python Training/train_tabular_q.py --env lineworld --episodes 10000
"""

import sys
import os
import argparse
import time
import json
import pickle

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Environnements.line_world import LineWorld
from Environnements.grid_world import GridWorld
from Environnements.tictactoe  import TicTacToe
from Environnements.quarto     import QuartoEnv
from Agents.tabular_q_agent    import TabularQAgent

# ── Dossiers de sortie ────────────────────────────────────────────────────────
BASE    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS  = os.path.join(BASE, "models",  "TabularQ")
RESULTS = os.path.join(BASE, "results", "TabularQ")
PLOTS   = os.path.join(BASE, "plots",   "TabularQ")
for d in [MODELS, RESULTS, PLOTS]:
    os.makedirs(d, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
#  Utilitaire : conversion état numpy → clé hashable
# ══════════════════════════════════════════════════════════════════════════════

def state_key(raw_state):
    """Convertit un numpy.ndarray en tuple hashable pour la Q-table."""
    if hasattr(raw_state, "tolist"):
        return tuple(raw_state.tolist())
    return tuple(raw_state)


# ══════════════════════════════════════════════════════════════════════════════
#  Fonctions d'évaluation (politique gloutonne, epsilon=0)
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_1player(env, agent, n_games=500, max_steps=200):
    """
    Évalue la politique gloutonne sur LineWorld ou GridWorld.
    L'epsilon est mis à 0 le temps de l'évaluation puis restauré.
    """
    saved_eps = agent.epsilon
    agent.epsilon = 0.0
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

    agent.epsilon = saved_eps
    return {
        "score_moyen":   round(float(np.mean(scores)),  4),
        "longueur_moy":  round(float(np.mean(lengths)), 2),
        "temps_coup_ms": round(float(np.mean(times)),   4),
    }


def evaluate_tictactoe(env, agent, n_games=500):
    """Évalue la politique gloutonne sur TicTacToe vs Random."""
    saved_eps = agent.epsilon
    agent.epsilon = 0.0
    scores, lengths, times = [], [], []

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
                result = 1 if info["winner"] == 1 else (0 if info["winner"] == 0 else -1)
                break
            if env.done:
                break
            rand_a = int(np.random.choice(env.get_actions()))
            _, _, done, info = env.step(rand_a)
            steps += 1
            if done:
                result = -1 if info["winner"] == -1 else (0 if info["winner"] == 0 else 1)
                break
        scores.append(result)
        lengths.append(steps)

    agent.epsilon = saved_eps
    return {
        "score_moyen":   round(float(np.mean(scores)),  4),
        "longueur_moy":  round(float(np.mean(lengths)), 2),
        "temps_coup_ms": round(float(np.mean(times)),   4),
    }


def evaluate_quarto(env, agent, n_games=500):
    """Évalue la politique gloutonne sur Quarto vs Random."""
    saved_eps = agent.epsilon
    agent.epsilon = 0.0
    scores, lengths, times = [], [], []

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
                result = 1 if env.winner == 1 else (-1 if env.winner == 2 else 0)
                break
        scores.append(result)
        lengths.append(steps)

    agent.epsilon = saved_eps
    return {
        "score_moyen":   round(float(np.mean(scores)),  4),
        "longueur_moy":  round(float(np.mean(lengths)), 2),
        "temps_coup_ms": round(float(np.mean(times if times else [0])), 4),
    }


# ══════════════════════════════════════════════════════════════════════════════
#  Boucles d'entraînement
# ══════════════════════════════════════════════════════════════════════════════

def train_1player(env, agent, num_episodes, checkpoints, evaluate_fn,
                  max_steps=200):
    """Entraîne TabularQ sur LineWorld ou GridWorld."""
    all_rewards, eval_results = [], {}
    checkpoints    = sorted(checkpoints)
    next_check_idx = 0

    for episode in range(1, num_episodes + 1):
        env.reset()
        ep_reward = 0.0
        steps     = 0

        while not env.done and steps < max_steps:
            s     = state_key(env.encode_state())
            avail = env.get_actions()
            a     = agent.choose_action(s, avail)
            _, reward, done = env.step(a)
            ns    = state_key(env.encode_state())
            n_avail = env.get_actions() if not done else []
            agent.learn(s, a, reward, ns, done, n_avail)
            ep_reward += reward
            steps     += 1

        agent.decay_epsilon()
        all_rewards.append(ep_reward)

        if next_check_idx < len(checkpoints) and episode == checkpoints[next_check_idx]:
            metrics = evaluate_fn(env, agent)
            eval_results[episode] = metrics
            print(f"  {episode:>9,} épisodes | Score : {metrics['score_moyen']:.4f} | "
                  f"Longueur : {metrics['longueur_moy']:.2f} | "
                  f"Temps/coup : {metrics['temps_coup_ms']:.4f} ms | "
                  f"ε : {agent.epsilon:.4f} | Q-table : {len(agent.q_table):,}")
            next_check_idx += 1

    return all_rewards, eval_results


def train_tictactoe(env, agent, num_episodes, checkpoints, evaluate_fn):
    """
    Entraîne TabularQ sur TicTacToe vs Random.
    L'agent joue toujours en premier (joueur 1).
    Reward : +1 victoire, 0 nul, -1 défaite.
    Les transitions intermédiaires reçoivent reward=0.
    """
    all_rewards, eval_results = [], {}
    checkpoints    = sorted(checkpoints)
    next_check_idx = 0

    for episode in range(1, num_episodes + 1):
        env.reset()
        ep_reward = 0.0

        while not env.done:
            s     = state_key(env.encode_state())
            avail = env.get_actions()
            a     = agent.choose_action(s, avail)
            _, _, done, info = env.step(a)

            if done:
                r  = 1 if info["winner"] == 1 else (0 if info["winner"] == 0 else -1)
                ns = state_key(env.encode_state())
                agent.learn(s, a, r, ns, True, [])
                ep_reward = r
                break

            if env.done:
                break

            # Coup adversaire aléatoire
            rand_a = int(np.random.choice(env.get_actions()))
            _, _, done, info = env.step(rand_a)

            if done:
                r  = -1 if info["winner"] == -1 else (0 if info["winner"] == 0 else 1)
                ns = state_key(env.encode_state())
                agent.learn(s, a, r, ns, True, [])
                ep_reward = r
                break

            # Transition non terminale
            ns = state_key(env.encode_state())
            agent.learn(s, a, 0.0, ns, False, env.get_actions())

        agent.decay_epsilon()
        all_rewards.append(ep_reward)

        if next_check_idx < len(checkpoints) and episode == checkpoints[next_check_idx]:
            metrics = evaluate_fn(env, agent)
            eval_results[episode] = metrics
            print(f"  {episode:>9,} épisodes | Score : {metrics['score_moyen']:.4f} | "
                  f"Longueur : {metrics['longueur_moy']:.2f} | "
                  f"Temps/coup : {metrics['temps_coup_ms']:.4f} ms | "
                  f"ε : {agent.epsilon:.4f} | Q-table : {len(agent.q_table):,}")
            next_check_idx += 1

    return all_rewards, eval_results


def train_quarto(env, agent, num_episodes, checkpoints, evaluate_fn):
    """
    Entraîne TabularQ sur Quarto vs Random.
    L'agent joue en tant que joueur 1.
    Reward : +1 victoire, 0 nul/match nul, -1 défaite.
    """
    all_rewards, eval_results = [], {}
    checkpoints    = sorted(checkpoints)
    next_check_idx = 0

    for episode in range(1, num_episodes + 1):
        env.reset()
        ep_reward   = 0.0
        last_s      = None
        last_a      = None
        last_avail  = None

        while not env.done:
            avail = env.get_actions()
            if not avail:
                break

            if env.current_player == 1:
                s = state_key(env.encode_state())
                a = agent.choose_action(s, avail)
                _, _, done, _ = env.step(a)

                if done:
                    r = 1 if env.winner == 1 else (-1 if env.winner == 2 else 0)
                    ns = state_key(env.encode_state())
                    agent.learn(s, a, r, ns, True, [])
                    ep_reward = r
                    break

                # sauvegarde pour apprendre après le coup adversaire
                last_s     = s
                last_a     = a
                last_avail = avail

            else:
                rand_a = int(np.random.choice(avail))
                _, _, done, _ = env.step(rand_a)

                if done:
                    r = 1 if env.winner == 1 else (-1 if env.winner == 2 else 0)
                    if last_s is not None:
                        ns = state_key(env.encode_state())
                        agent.learn(last_s, last_a, r, ns, True, [])
                    ep_reward = r
                    break

                # Apprentissage de la transition après coup adversaire
                if last_s is not None:
                    ns = state_key(env.encode_state())
                    agent.learn(last_s, last_a, 0.0, ns, False, env.get_actions())

        agent.decay_epsilon()
        all_rewards.append(ep_reward)

        if next_check_idx < len(checkpoints) and episode == checkpoints[next_check_idx]:
            metrics = evaluate_fn(env, agent)
            eval_results[episode] = metrics
            print(f"  {episode:>9,} épisodes | Score : {metrics['score_moyen']:.4f} | "
                  f"Longueur : {metrics['longueur_moy']:.2f} | "
                  f"Temps/coup : {metrics['temps_coup_ms']:.4f} ms | "
                  f"ε : {agent.epsilon:.4f} | Q-table : {len(agent.q_table):,}")
            next_check_idx += 1

    return all_rewards, eval_results


# ══════════════════════════════════════════════════════════════════════════════
#  Sauvegarde / chargement (Q-table via pickle)
# ══════════════════════════════════════════════════════════════════════════════

def save_agent(agent, path):
    """Sauvegarde la Q-table et les hyperparamètres dans un fichier pickle."""
    data = {
        "q_table":       agent.q_table,
        "alpha":         agent.alpha,
        "gamma":         agent.gamma,
        "epsilon":       agent.epsilon,
        "epsilon_decay": agent.epsilon_decay,
        "epsilon_min":   agent.epsilon_min,
    }
    with open(path, "wb") as f:
        pickle.dump(data, f)
    print(f"  → Modèle sauvegardé : {path}  (Q-table : {len(agent.q_table):,} états)")


def load_agent(path):
    """Charge une Q-table et reconstruit le TabularQAgent."""
    with open(path, "rb") as f:
        data = pickle.load(f)
    agent = TabularQAgent(
        alpha         = data["alpha"],
        gamma         = data["gamma"],
        epsilon       = data["epsilon"],
        epsilon_decay = data["epsilon_decay"],
        epsilon_min   = data["epsilon_min"],
    )
    agent.q_table = data["q_table"]
    return agent


# ══════════════════════════════════════════════════════════════════════════════
#  Graphiques
# ══════════════════════════════════════════════════════════════════════════════

def smooth(data, w=200):
    return [np.mean(data[max(0, i - w): i + 1]) for i in range(len(data))]


def save_plot(rewards, eval_results, env_label, plots_dir, num_episodes, window=200):
    episodes = np.arange(1, len(rewards) + 1)

    fig, axes = plt.subplots(2, 1, figsize=(13, 8))
    fig.suptitle(f"TabularQ — {env_label}\n({num_episodes:,} épisodes)",
                 fontsize=13, fontweight="bold")

    # Reward d'entraînement
    axes[0].plot(episodes, smooth(rewards, window), color="#2196F3",
                 linewidth=1.2, label=f"Reward lissé (fenêtre {window})")
    if eval_results:
        xs = sorted(eval_results.keys())
        ys = [eval_results[x]["score_moyen"] for x in xs]
        ax2 = axes[0].twinx()
        ax2.plot(xs, ys, "o--", color="#FF5722", markersize=7, linewidth=1.5,
                 label="Score éval. (glouton)")
        ax2.set_ylabel("Score éval. (politique gloutonne)", color="#FF5722")
        ax2.tick_params(axis="y", labelcolor="#FF5722")
        ax2.legend(loc="lower right", fontsize=9)
    axes[0].set_ylabel(f"Reward lissé")
    axes[0].set_xlabel("Épisode")
    axes[0].set_title("Reward d'entraînement (avec exploration ε-greedy)")
    axes[0].legend(loc="upper left", fontsize=9)
    axes[0].grid(True, alpha=0.3)

    # Progression aux checkpoints
    if eval_results:
        xs = sorted(eval_results.keys())
        ys = [eval_results[x]["score_moyen"] for x in xs]
        axes[1].plot(xs, ys, "o-", color="#2196F3", linewidth=2, markersize=8)
        for x, y in zip(xs, ys):
            axes[1].annotate(f"{y:.3f}", (x, y), textcoords="offset points",
                             xytext=(5, 6), fontsize=9)
        axes[1].set_xscale("log")
        axes[1].set_xlabel("Épisodes d'entraînement (log)")
        axes[1].set_ylabel("Score moyen (politique gloutonne)")
        axes[1].set_title("Progression du score moyen aux checkpoints")
        axes[1].axhline(0, color="gray", linestyle="--", linewidth=0.8)
        axes[1].grid(True, alpha=0.3, which="both")

    plt.tight_layout()
    fname = f"TabularQ_{env_label.split()[0]}_training.png"
    path  = os.path.join(plots_dir, fname)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  → Graphique : {path}")


def print_table(eval_results, env_label):
    print()
    print("=" * 72)
    print(f"  TabularQ  —  {env_label}")
    print("=" * 72)
    print(f"  {'Épisodes':>12} | {'Score moyen':>12} | {'Longueur moy':>13} | "
          f"{'Temps/coup':>12}")
    print("  " + "-" * 60)
    for ep, m in sorted(eval_results.items()):
        print(f"  {ep:>12,} | {m['score_moyen']:>12.4f} | "
              f"{m['longueur_moy']:>13.2f} | {m['temps_coup_ms']:>11.4f}ms")
    print()


# ══════════════════════════════════════════════════════════════════════════════
#  Configurations
# ══════════════════════════════════════════════════════════════════════════════

ENV_CONFIGS = {
    "lineworld": {
        "label":     "LineWorld",
        "env_fn":    lambda: LineWorld(size=6),
        "train_fn":  train_1player,
        "eval_fn":   evaluate_1player,
        "note":      "Espace d'états petit (~6 positions) — Q-table très compacte.",
        # Hyperparamètres adaptés : LineWorld est simple, convergence rapide
        "agent_hp":  dict(alpha=0.3, gamma=0.99, epsilon=1.0,
                          epsilon_decay=0.999, epsilon_min=0.05),
    },
    "gridworld": {
        "label":     "GridWorld (5×5)",
        "env_fn":    lambda: GridWorld(rows=5, cols=5),
        "train_fn":  train_1player,
        "eval_fn":   evaluate_1player,
        "note":      "25 cases + encodage one-hot position + goal → Q-table de taille raisonnable.",
        "agent_hp":  dict(alpha=0.2, gamma=0.99, epsilon=1.0,
                          epsilon_decay=0.999, epsilon_min=0.05),
    },
    "tictactoe": {
        "label":     "TicTacToe (vs Random)",
        "env_fn":    lambda: TicTacToe(),
        "train_fn":  train_tictactoe,
        "eval_fn":   evaluate_tictactoe,
        "note":      "~5 478 états distincts — TabularQ parfaitement adapté.",
        "agent_hp":  dict(alpha=0.1, gamma=0.99, epsilon=1.0,
                          epsilon_decay=0.9995, epsilon_min=0.05),
    },
    "quarto": {
        "label":     "Quarto (vs Random)",
        "env_fn":    lambda: QuartoEnv(),
        "train_fn":  train_quarto,
        "eval_fn":   evaluate_quarto,
        "note":      "Espace d'états très grand — Q-table croît rapidement. "
                     "TabularQ sous-optimal sur Quarto : préférer DQN/AlphaZero.",
        "agent_hp":  dict(alpha=0.1, gamma=0.99, epsilon=1.0,
                          epsilon_decay=0.9999, epsilon_min=0.05),
    },
}

ENVS_LIST = ["lineworld", "gridworld", "tictactoe", "quarto"]


def build_checkpoints(num_episodes):
    candidates = [1_000, 10_000, 100_000, 1_000_000]
    cps = [c for c in candidates if c <= num_episodes]
    if not cps or cps[-1] != num_episodes:
        cps.append(num_episodes)
    return sorted(set(cps))


# ══════════════════════════════════════════════════════════════════════════════
#  Runner principal : 1 environnement
# ══════════════════════════════════════════════════════════════════════════════

def run_one(env_key, num_episodes, checkpoints, n_eval=500):
    ecfg   = ENV_CONFIGS[env_key]
    elabel = ecfg["label"]

    print(f"\n{'═' * 70}")
    print(f"  TabularQ  —  {elabel}")
    print(f"  Épisodes : {num_episodes:,}   |   Checkpoints : {checkpoints}")
    print(f"  Note     : {ecfg['note']}")
    print(f"{'═' * 70}")

    env   = ecfg["env_fn"]()
    agent = TabularQAgent(**ecfg["agent_hp"])

    t0 = time.time()

    # Entraînement
    all_rewards, eval_results = ecfg["train_fn"](
        env, agent, num_episodes, checkpoints, ecfg["eval_fn"]
    )

    elapsed = time.time() - t0
    print(f"\n  Durée : {elapsed:.1f}s ({elapsed / 60:.1f} min)")
    print(f"  Q-table finale : {len(agent.q_table):,} états distincts")

    # Évaluation finale (politique gloutonne)
    final_metrics = ecfg["eval_fn"](env, agent, n_games=n_eval)

    print_table(eval_results, elabel)
    print(f"  Évaluation finale ({n_eval} parties, ε=0) :")
    print(f"    Score moyen    : {final_metrics['score_moyen']:.4f}")
    print(f"    Longueur moy.  : {final_metrics['longueur_moy']:.2f} steps")
    print(f"    Temps/coup     : {final_metrics['temps_coup_ms']:.4f} ms")

    # Graphique
    save_plot(all_rewards, eval_results, elabel, PLOTS, num_episodes)

    # Sauvegarde modèle
    model_path = os.path.join(MODELS, f"TabularQ_{elabel.split()[0]}.pkl")
    save_agent(agent, model_path)

    # JSON résultats
    data = {
        "agent":         "TabularQ",
        "env":           elabel,
        "num_episodes":  num_episodes,
        "elapsed_s":     round(elapsed, 1),
        "hyperparams":   ecfg["agent_hp"],
        "q_table_size":  len(agent.q_table),
        "checkpoints":   {str(k): v for k, v in eval_results.items()},
        "final_eval":    final_metrics,
    }
    json_path = os.path.join(RESULTS, f"TabularQ_{elabel.split()[0]}.json")
    with open(json_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  → JSON : {json_path}")

    return data


# ══════════════════════════════════════════════════════════════════════════════
#  Point d'entrée
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Entraînement TabularQ — 4 environnements"
    )
    parser.add_argument("--env",      default="all",
                        choices=["all"] + ENVS_LIST,
                        help="Environnement cible (default: all)")
    parser.add_argument("--episodes", type=int, default=10_000,
                        help="Épisodes d'entraînement (default: 10000)")
    parser.add_argument("--n_eval",   type=int, default=500,
                        help="Parties pour l'évaluation finale (default: 500)")
    args = parser.parse_args()

    envs        = ENVS_LIST if args.env == "all" else [args.env]
    checkpoints = build_checkpoints(args.episodes)

    print(f"\n{'═' * 70}")
    print(f"  ENTRAÎNEMENT TabularQ")
    print(f"  Environnements : {[ENV_CONFIGS[e]['label'] for e in envs]}")
    print(f"  Épisodes       : {args.episodes:,}")
    print(f"  Checkpoints    : {checkpoints}")
    print(f"  Éval (parties) : {args.n_eval}")
    print(f"{'═' * 70}")

    global_t0   = time.time()
    all_results = {}

    for env_key in envs:
        all_results[env_key] = run_one(env_key, args.episodes, checkpoints, args.n_eval)

    # Résumé global
    last_cp = max(checkpoints)
    print(f"\n{'═' * 80}")
    print(f"  RÉSUMÉ GLOBAL TabularQ (checkpoint {last_cp:,} épisodes)")
    print(f"{'═' * 80}")
    header = f"  {'Environnement':>22}" + " | Score moyen | Longueur moy | Temps/coup | Q-table"
    print(header)
    print("  " + "─" * (len(header) - 2))
    for env_key in envs:
        cps = all_results[env_key].get("checkpoints", {})
        m   = cps.get(str(last_cp)) or list(cps.values())[-1] if cps else {}
        qt  = all_results[env_key].get("q_table_size", 0)
        label = ENV_CONFIGS[env_key]["label"]
        if m:
            print(f"  {label:>22} | {m['score_moyen']:>11.4f} | "
                  f"{m['longueur_moy']:>12.2f} | {m['temps_coup_ms']:>10.4f}ms | {qt:>8,}")

    total = time.time() - global_t0
    print(f"\n  Durée totale : {total:.0f}s ({total / 60:.1f} min)")
    print(f"  Modèles      → {MODELS}")
    print(f"  Résultats    → {RESULTS}")
    print(f"  Graphiques   → {PLOTS}")
    print(f"{'═' * 80}\n")


if __name__ == "__main__":
    main()
