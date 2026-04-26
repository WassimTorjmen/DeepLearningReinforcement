"""
experiment_alphazero.py
=======================
Expérimentations complètes AlphaZero pour le rapport.

Pour chaque environnement :
  - Entraînement avec checkpoints à 1k / 10k / 100k épisodes
  - Métriques demandées par le syllabus (score moyen, longueur, temps/coup)
  - Graphiques d'apprentissage
  - Comparaison mode réseau seul vs MCTS à l'évaluation
  - Export JSON + Markdown des résultats

Utilisation :
  python experiment_alphazero.py                    # 100k épisodes
  python experiment_alphazero.py --episodes 10000   # rapide pour tester
  python experiment_alphazero.py --env tictactoe --episodes 50000
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

base = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(base)
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, "Agents"))
sys.path.append(os.path.join(project_root, "Environnements"))

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from Environnements.line_world import LineWorld
from Environnements.grid_world import GridWorld
from Environnements.tictactoe  import TicTacToe
from Environnements.quarto     import QuartoEnv
from Agents.Alpha_zero         import AlphaZeroAgent
from experiment import (
    train_alphazero_1player,
    train_alphazero_tictactoe,
    train_alphazero_quarto,
    evaluate_no_training_1player,
    evaluate_no_training_tictactoe,
    evaluate_no_training_quarto,
)
from Evaluation.evaluate_alphazero import (
    evaluate_lineworld,
    evaluate_gridworld,
    evaluate_tictactoe,
    evaluate_quarto,
)

# ── Dossiers de sortie ─────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR  = os.path.join(BASE_DIR, "models",  "alphazero")
RESULTS_DIR = os.path.join(BASE_DIR, "results", "alphazero")
PLOTS_DIR   = os.path.join(BASE_DIR, "plots",   "alphazero")
for d in [MODELS_DIR, RESULTS_DIR, PLOTS_DIR]:
    os.makedirs(d, exist_ok=True)

# ── Hyperparamètres ────────────────────────────────────────────────────────────
CONFIGS = {
    "lineworld": {
        "state_size": 8,   "num_actions": 2,  "hidden_size": 64,
        "lr": 3e-4, "n_simulations": 20, "c_puct": 1.0,
        "env_fn": lambda: LineWorld(size=6),
        "train_fn": train_alphazero_1player,
        "eval_fn":  evaluate_no_training_1player,
        "detail_eval_fn": evaluate_lineworld,
        "max_steps": 200,
    },
    "gridworld": {
        "state_size": 31,  "num_actions": 4,  "hidden_size": 128,
        "lr": 3e-4, "n_simulations": 20, "c_puct": 1.0,
        "env_fn": lambda: GridWorld(rows=5, cols=5),
        "train_fn": train_alphazero_1player,
        "eval_fn":  evaluate_no_training_1player,
        "detail_eval_fn": evaluate_gridworld,
        "max_steps": 200,
    },
    "tictactoe": {
        "state_size": 27,  "num_actions": 9,  "hidden_size": 128,
        "lr": 1e-3, "n_simulations": 50, "c_puct": 1.5,
        "env_fn": lambda: TicTacToe(),
        "train_fn": train_alphazero_tictactoe,
        "eval_fn":  evaluate_no_training_tictactoe,
        "detail_eval_fn": evaluate_tictactoe,
        "max_steps": None,
    },
    "quarto": {
        "state_size": 105, "num_actions": 32, "hidden_size": 256,
        "lr": 1e-3, "n_simulations": 30, "c_puct": 1.5,
        "env_fn": lambda: QuartoEnv(),
        "train_fn": train_alphazero_quarto,
        "eval_fn":  evaluate_no_training_quarto,
        "detail_eval_fn": evaluate_quarto,
        "max_steps": None,
    },
}

ENV_LABELS = {
    "lineworld": "LineWorld",
    "gridworld": "GridWorld (5×5)",
    "tictactoe": "TicTacToe (vs Random)",
    "quarto":    "Quarto (vs Random)",
}


# ══════════════════════════════════════════════════════════════════════════════
#  Helpers graphiques
# ══════════════════════════════════════════════════════════════════════════════

def smooth(data, w=100):
    return [np.mean(data[max(0, i - w): i + 1]) for i in range(len(data))]


def plot_training(rewards, p_losses, v_losses, eval_results, key, num_episodes, window=100):
    label = ENV_LABELS[key]
    episodes = np.arange(1, len(rewards) + 1)

    fig, axes = plt.subplots(3, 1, figsize=(13, 11))
    fig.suptitle(f"AlphaZero — {label}\n({num_episodes:,} épisodes)", fontsize=13, fontweight="bold")

    # ── Reward ─────────────────────────────────────────────────────────────────
    axes[0].plot(episodes, smooth(rewards, window), color="#2196F3", linewidth=1.3, label="Reward lissé")
    if eval_results:
        xs = list(eval_results.keys())
        ys = [eval_results[x]["score_moyen"] for x in xs]
        ax0r = axes[0].twinx()
        ax0r.plot(xs, ys, "o--", color="#FF5722", markersize=6, linewidth=1.5, label="Score éval.")
        ax0r.set_ylabel("Score éval. (500 parties)", color="#FF5722")
        ax0r.tick_params(axis="y", labelcolor="#FF5722")
        ax0r.legend(loc="lower right", fontsize=9)
    axes[0].set_ylabel(f"Reward lissé (fenêtre {window})")
    axes[0].set_xlabel("Épisode")
    axes[0].set_title("Reward d'entraînement")
    axes[0].legend(loc="upper left", fontsize=9)
    axes[0].grid(True, alpha=0.3)

    # ── Policy Loss ────────────────────────────────────────────────────────────
    axes[1].plot(episodes, smooth(p_losses, window), color="#2196F3", linewidth=1.0)
    axes[1].set_ylabel("Policy Loss")
    axes[1].set_xlabel("Épisode")
    axes[1].set_title("Policy Loss (cross-entropy MCTS vs réseau)")
    axes[1].grid(True, alpha=0.3)

    # ── Value Loss ─────────────────────────────────────────────────────────────
    if v_losses:
        axes[2].plot(episodes, smooth(v_losses, window), color="#FF5722", linewidth=1.0)
        axes[2].set_ylabel("Value Loss")
        axes[2].set_xlabel("Épisode")
        axes[2].set_title("Value Loss (MSE valeur prédite vs reward)")
        axes[2].grid(True, alpha=0.3)
    else:
        axes[2].axis("off")

    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, f"AlphaZero_{key}_training.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  → Graphique training : {path}")


def plot_eval_progression(eval_results, key, metric="score_moyen"):
    label = ENV_LABELS[key]
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
    ax.set_title(f"AlphaZero — {label}\nÉvolution du {metric} aux checkpoints")
    ax.grid(True, alpha=0.3, which="both")
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, f"AlphaZero_{key}_progression.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  → Graphique progression : {path}")


# ══════════════════════════════════════════════════════════════════════════════
#  Expérimentation principale par environnement
# ══════════════════════════════════════════════════════════════════════════════

def run_experiment(key, num_episodes, checkpoints, n_eval=500):
    cfg   = CONFIGS[key]
    label = ENV_LABELS[key]

    print("\n" + "═" * 70)
    print(f"  EXPÉRIMENTATION  AlphaZero — {label}")
    print(f"  Épisodes      : {num_episodes:,}")
    print(f"  Simulations   : {cfg['n_simulations']} / coup (MCTS)")
    print(f"  Hidden size   : {cfg['hidden_size']}")
    print(f"  LR            : {cfg['lr']}")
    print(f"  c_puct        : {cfg['c_puct']}")
    print("═" * 70)

    env   = cfg["env_fn"]()
    agent = AlphaZeroAgent(
        state_size    = cfg["state_size"],
        num_actions   = cfg["num_actions"],
        n_simulations = cfg["n_simulations"],
        hidden_size   = cfg["hidden_size"],
        lr            = cfg["lr"],
        c_puct        = cfg["c_puct"],
    )

    # ── Entraînement ───────────────────────────────────────────────────────────
    print("\n  Phase 1 : Entraînement ...")
    t_start = time.time()

    train_kwargs = dict(
        env          = env,
        agent        = agent,
        num_episodes = num_episodes,
        checkpoints  = checkpoints,
        evaluate_fn  = cfg["eval_fn"],
    )
    if cfg.get("max_steps"):
        train_kwargs["max_steps"] = cfg["max_steps"]

    rewards, p_losses, v_losses, eval_results = cfg["train_fn"](**train_kwargs)

    elapsed = time.time() - t_start
    print(f"\n  Durée entraînement : {elapsed:.1f}s  ({elapsed / 60:.1f} min)")

    # ── Graphiques training ────────────────────────────────────────────────────
    plot_training(rewards, p_losses, v_losses, eval_results, key, num_episodes)
    plot_eval_progression(eval_results, key)

    # ── Sauvegarde modèle ──────────────────────────────────────────────────────
    model_path = os.path.join(MODELS_DIR, f"AlphaZero_{label.split()[0]}.pt")
    agent.save(model_path)
    print(f"  → Modèle : {model_path}")

    # ── Évaluation finale détaillée ────────────────────────────────────────────
    print(f"\n  Phase 2 : Évaluation finale ({n_eval} parties) ...")

    # Mode réseau seul (rapide)
    m_net  = cfg["detail_eval_fn"](agent, n_games=n_eval, use_mcts=False)
    # Mode MCTS complet (plus lent, plus fort)
    m_mcts = cfg["detail_eval_fn"](agent, n_games=min(n_eval, 200), use_mcts=True)

    # ── Affichage résultats ────────────────────────────────────────────────────
    print()
    print(f"  {'─' * 55}")
    print(f"  {'Métrique':<25} | {'Réseau seul':>13} | {'MCTS':>10}")
    print(f"  {'─' * 55}")
    for k in ["score_moyen", "longueur_moy", "temps_coup_ms"]:
        print(f"  {k:<25} | {m_net[k]:>13.4f} | {m_mcts[k]:>10.4f}")
    if "taux_victoire" in m_net:
        print(f"  {'taux_victoire':<25} | {m_net['taux_victoire']:>12.1%} | {m_mcts['taux_victoire']:>9.1%}")
    if "taux_nul" in m_net:
        print(f"  {'taux_nul':<25} | {m_net['taux_nul']:>12.1%} | {m_mcts['taux_nul']:>9.1%}")
    if "taux_defaite" in m_net:
        print(f"  {'taux_defaite':<25} | {m_net['taux_defaite']:>12.1%} | {m_mcts['taux_defaite']:>9.1%}")
    print(f"  {'─' * 55}")

    # ── Table checkpoints (format syllabus) ────────────────────────────────────
    print()
    print(f"  {'Épisodes':>12} | {'Score moyen':>12} | {'Longueur moy':>13} | {'Temps/coup':>12}")
    print(f"  {'─' * 60}")
    for ep in sorted(eval_results.keys()):
        m = eval_results[ep]
        print(f"  {ep:>12,} | {m['score_moyen']:>12.4f} | {m['longueur_moy']:>13.2f} | {m['temps_coup_ms']:>11.4f}ms")

    # ── Résultats JSON ─────────────────────────────────────────────────────────
    result_data = {
        "env":            label,
        "agent":          "AlphaZero",
        "num_episodes":   num_episodes,
        "elapsed_s":      round(elapsed, 1),
        "hyperparams": {
            "state_size":    cfg["state_size"],
            "num_actions":   cfg["num_actions"],
            "hidden_size":   cfg["hidden_size"],
            "lr":            cfg["lr"],
            "n_simulations": cfg["n_simulations"],
            "c_puct":        cfg["c_puct"],
        },
        "checkpoints":      {str(k): v for k, v in eval_results.items()},
        "final_eval_net":   m_net,
        "final_eval_mcts":  m_mcts,
    }
    json_path = os.path.join(RESULTS_DIR, f"AlphaZero_{key}.json")
    with open(json_path, "w") as f:
        json.dump(result_data, f, indent=2)
    print(f"\n  → JSON résultats : {json_path}")

    return result_data


# ══════════════════════════════════════════════════════════════════════════════
#  Génération du rapport Markdown
# ══════════════════════════════════════════════════════════════════════════════

def generate_markdown_report(all_results, num_episodes):
    lines = [
        "# Rapport d'expérimentation — AlphaZero\n",
        f"> Épisodes d'entraînement : {num_episodes:,}\n",
        "",
        "## Hyperparamètres\n",
        "| Environnement | State size | Actions | Hidden | LR | Simulations | c_puct |",
        "|---|---|---|---|---|---|---|",
    ]
    for key, data in all_results.items():
        h = data["hyperparams"]
        lines.append(
            f"| {data['env']} | {h['state_size']} | {h['num_actions']} | "
            f"{h['hidden_size']} | {h['lr']} | {h['n_simulations']} | {h['c_puct']} |"
        )

    lines += ["", "## Résultats aux checkpoints\n"]
    for key, data in all_results.items():
        lines.append(f"### {data['env']}\n")
        lines.append("| Épisodes | Score moyen | Longueur moy | Temps/coup (ms) |")
        lines.append("|---|---|---|---|")
        for ep_str, m in data["checkpoints"].items():
            lines.append(
                f"| {int(ep_str):,} | {m['score_moyen']:.4f} | "
                f"{m['longueur_moy']:.2f} | {m['temps_coup_ms']:.4f} |"
            )
        lines.append("")

    lines += ["", "## Évaluation finale\n"]
    lines += [
        "| Environnement | Mode | Score | Victoires | Nuls | Défaites | Longueur | ms/coup |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for key, data in all_results.items():
        for mode_key, mode_label in [("final_eval_net", "Réseau"), ("final_eval_mcts", "MCTS")]:
            m = data[mode_key]
            wins  = f"{m.get('taux_victoire', 0):.1%}"
            draws = f"{m.get('taux_nul',      0):.1%}"
            loss  = f"{m.get('taux_defaite',  0):.1%}"
            lines.append(
                f"| {data['env']} | {mode_label} | {m['score_moyen']:.4f} | "
                f"{wins} | {draws} | {loss} | {m['longueur_moy']:.2f} | {m['temps_coup_ms']:.4f} |"
            )

    lines += [
        "",
        "## Observations\n",
        "*(À compléter par votre analyse des résultats)*\n",
        "",
        "- **LineWorld** : environnement simple 1-joueur. AlphaZero converge rapidement vers la politique optimale.",
        "- **GridWorld** : espace d'états plus grand avec un piège. Le réseau doit apprendre à éviter le coin bas-droit.",
        "- **TicTacToe** : jeu à somme nulle. AlphaZero via MCTS surpasse vite le joueur random.",
        "- **Quarto** : espace de jeu complexe (32 actions). Les simulations MCTS sont cruciales pour progresser.",
        "",
        "## Interprétation AlphaZero vs autres algos\n",
        "- AlphaZero combine recherche arborescente (MCTS-PUCT) et apprentissage supervisé par auto-jeu.",
        "- La policy head guide les simulations (meilleure exploration), la value head remplace le rollout aléatoire.",
        "- Il converge plus lentement que DQN sur des envs simples, mais monte en qualité sur des jeux complexes.",
        "- L'inconvénient majeur est le coût en temps : chaque coup nécessite N simulations.",
    ]

    md_path = os.path.join(RESULTS_DIR, "AlphaZero_rapport.md")
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


def main():
    parser = argparse.ArgumentParser(description="Expérimentation AlphaZero complète")
    parser.add_argument("--env",      default="all",
                        choices=["all", "lineworld", "gridworld", "tictactoe", "quarto"])
    parser.add_argument("--episodes", type=int, default=100_000,
                        help="Épisodes d'entraînement (default: 100000)")
    parser.add_argument("--n_eval",   type=int, default=500,
                        help="Parties pour l'évaluation finale (default: 500)")
    args = parser.parse_args()

    checkpoints = build_checkpoints(args.episodes)
    targets = ["lineworld", "gridworld", "tictactoe", "quarto"] if args.env == "all" else [args.env]

    print(f"\n{'═' * 70}")
    print(f"  EXPÉRIMENTATION ALPHAZERO")
    print(f"  Environnements : {targets}")
    print(f"  Épisodes       : {args.episodes:,}")
    print(f"  Checkpoints    : {checkpoints}")
    print(f"{'═' * 70}")

    all_results = {}
    global_t0 = time.time()

    for key in targets:
        data = run_experiment(key, args.episodes, checkpoints, n_eval=args.n_eval)
        all_results[key] = data

    # Rapport Markdown
    generate_markdown_report(all_results, args.episodes)

    total = time.time() - global_t0
    print(f"\n{'═' * 70}")
    print(f"  EXPÉRIMENTATION TERMINÉE  —  Durée totale : {total:.0f}s ({total / 60:.1f} min)")
    print(f"  Modèles  → {MODELS_DIR}")
    print(f"  Résultats → {RESULTS_DIR}")
    print(f"  Graphiques → {PLOTS_DIR}")
    print(f"{'═' * 70}\n")


if __name__ == "__main__":
    main()
