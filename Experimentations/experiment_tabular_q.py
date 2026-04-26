"""
experiment_tabular_q.py
=======================
Expérimentations complètes pour le TabularQAgent sur les 4 environnements.

Pour chaque environnement :
  1. Impact des hyperparamètres (alpha, gamma, epsilon_decay)
  2. Entraînement complet avec checkpoints 1k / 10k / 100k épisodes
  3. Métriques du syllabus : score moyen, longueur moy, temps/coup
  4. Graphiques training : reward lissé + score aux checkpoints
  5. Graphique comparatif des hyperparamètres
  6. Évolution de la taille de la Q-table
  7. Export JSON + rapport Markdown

Utilisation :
  python experiment_tabular_q.py                           # tous envs, 10k épisodes
  python experiment_tabular_q.py --episodes 100000         # 100k épisodes
  python experiment_tabular_q.py --env tictactoe           # env ciblé
  python experiment_tabular_q.py --skip_hp_search          # saute la recherche d'hyperparams
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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from Environnements.line_world import LineWorld
from Environnements.grid_world import GridWorld
from Environnements.tictactoe  import TicTacToe
from Environnements.quarto     import QuartoEnv
from Agents.tabular_q_agent    import TabularQAgent

from Training.train_tabular_q import (
    train_1player, train_tictactoe, train_quarto,
    evaluate_1player, evaluate_tictactoe, evaluate_quarto,
    save_agent, state_key, smooth,
)
from Evaluation.evaluate_tabular_q import (
    evaluate_lineworld, evaluate_gridworld,
    evaluate_tictactoe as eval_ttt_full,
    evaluate_quarto    as eval_quarto_full,
    q_table_stats, print_metrics, print_q_stats,
)

# ── Dossiers de sortie ────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR  = os.path.join(BASE_DIR, "models",   "TabularQ")
RESULTS_DIR = os.path.join(BASE_DIR, "results",  "experiment_tabular_q")
PLOTS_DIR   = os.path.join(BASE_DIR, "plots",    "experiment_tabular_q")
for d in [MODELS_DIR, RESULTS_DIR, PLOTS_DIR]:
    os.makedirs(d, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
#  Configurations des environnements
# ══════════════════════════════════════════════════════════════════════════════

ENV_CONFIGS = {
    "lineworld": {
        "label":     "LineWorld",
        "env_fn":    lambda: LineWorld(size=6),
        "train_fn":  train_1player,
        "eval_fn":   evaluate_1player,
        "eval_fn_full": evaluate_lineworld,
        "default_hp": dict(alpha=0.3, gamma=0.99, epsilon=1.0,
                           epsilon_decay=0.999, epsilon_min=0.05),
        "note": "Espace d'états très petit. Convergence rapide attendue.",
    },
    "gridworld": {
        "label":     "GridWorld (5×5)",
        "env_fn":    lambda: GridWorld(rows=5, cols=5),
        "train_fn":  train_1player,
        "eval_fn":   evaluate_1player,
        "eval_fn_full": evaluate_gridworld,
        "default_hp": dict(alpha=0.2, gamma=0.99, epsilon=1.0,
                           epsilon_decay=0.999, epsilon_min=0.05),
        "note": "25 positions encodées en one-hot. Q-table de taille modérée.",
    },
    "tictactoe": {
        "label":     "TicTacToe (vs Random)",
        "env_fn":    lambda: TicTacToe(),
        "train_fn":  train_tictactoe,
        "eval_fn":   evaluate_tictactoe,
        "eval_fn_full": eval_ttt_full,
        "default_hp": dict(alpha=0.1, gamma=0.99, epsilon=1.0,
                           epsilon_decay=0.9995, epsilon_min=0.05),
        "note": "~5 478 états théoriques. TabularQ parfaitement adapté.",
    },
    "quarto": {
        "label":     "Quarto (vs Random)",
        "env_fn":    lambda: QuartoEnv(),
        "train_fn":  train_quarto,
        "eval_fn":   evaluate_quarto,
        "eval_fn_full": eval_quarto_full,
        "default_hp": dict(alpha=0.1, gamma=0.99, epsilon=1.0,
                           epsilon_decay=0.9999, epsilon_min=0.05),
        "note": "Espace d'états très grand. Q-table croît vite : couverture partielle.",
    },
}

ENVS_LIST = ["lineworld", "gridworld", "tictactoe", "quarto"]


# ══════════════════════════════════════════════════════════════════════════════
#  Grilles d'hyperparamètres pour la recherche
# ══════════════════════════════════════════════════════════════════════════════

HP_GRIDS = {
    "lineworld": [
        {"alpha": 0.1,  "epsilon_decay": 0.999},
        {"alpha": 0.3,  "epsilon_decay": 0.999},
        {"alpha": 0.5,  "epsilon_decay": 0.999},
        {"alpha": 0.3,  "epsilon_decay": 0.995},
    ],
    "gridworld": [
        {"alpha": 0.1,  "epsilon_decay": 0.999},
        {"alpha": 0.2,  "epsilon_decay": 0.999},
        {"alpha": 0.3,  "epsilon_decay": 0.999},
        {"alpha": 0.2,  "epsilon_decay": 0.9995},
    ],
    "tictactoe": [
        {"alpha": 0.05, "epsilon_decay": 0.9999},
        {"alpha": 0.1,  "epsilon_decay": 0.9995},
        {"alpha": 0.2,  "epsilon_decay": 0.9990},
        {"alpha": 0.1,  "epsilon_decay": 0.9990},
    ],
    "quarto": [
        {"alpha": 0.05, "epsilon_decay": 0.99995},
        {"alpha": 0.1,  "epsilon_decay": 0.9999},
        {"alpha": 0.2,  "epsilon_decay": 0.9999},
        {"alpha": 0.1,  "epsilon_decay": 0.9995},
    ],
}

# Épisodes réduits pour la recherche HP (plus rapide)
HP_SEARCH_EPISODES = {
    "lineworld": 5_000,
    "gridworld": 5_000,
    "tictactoe": 20_000,
    "quarto":    20_000,
}


# ══════════════════════════════════════════════════════════════════════════════
#  Graphiques
# ══════════════════════════════════════════════════════════════════════════════

def plot_training_full(rewards, eval_results, qtable_sizes, env_label,
                       plots_dir, num_episodes, window=200):
    """
    Graphique d'entraînement avec 3 sous-graphiques :
      1. Reward lissé + score éval aux checkpoints
      2. Progression du score moyen (axe log)
      3. Évolution de la taille de la Q-table
    """
    episodes = np.arange(1, len(rewards) + 1)
    fig, axes = plt.subplots(3, 1, figsize=(13, 12))
    fig.suptitle(f"TabularQ — {env_label}\n({num_episodes:,} épisodes)",
                 fontsize=13, fontweight="bold")

    # --- Subplot 1 : Reward d'entraînement ---
    axes[0].plot(episodes, smooth(rewards, window), color="#2196F3",
                 linewidth=1.2, label=f"Reward lissé (fenêtre {window})")
    if eval_results:
        xs  = sorted(eval_results.keys())
        ys  = [eval_results[x]["score_moyen"] for x in xs]
        ax2 = axes[0].twinx()
        ax2.plot(xs, ys, "o--", color="#FF5722", markersize=7, linewidth=1.5,
                 label="Score éval. (ε=0)")
        ax2.set_ylabel("Score éval. (politique gloutonne)", color="#FF5722")
        ax2.tick_params(axis="y", labelcolor="#FF5722")
        ax2.legend(loc="lower right", fontsize=9)
    axes[0].set_ylabel("Reward lissé")
    axes[0].set_xlabel("Épisode")
    axes[0].set_title("Reward d'entraînement (avec exploration ε-greedy)")
    axes[0].legend(loc="upper left", fontsize=9)
    axes[0].grid(True, alpha=0.3)

    # --- Subplot 2 : Progression score (axe log) ---
    if eval_results:
        xs = sorted(eval_results.keys())
        ys = [eval_results[x]["score_moyen"] for x in xs]
        axes[1].plot(xs, ys, "o-", color="#2196F3", linewidth=2, markersize=8)
        for x, y in zip(xs, ys):
            axes[1].annotate(f"{y:.3f}", (x, y), textcoords="offset points",
                             xytext=(5, 6), fontsize=9)
        axes[1].set_xscale("log")
        axes[1].axhline(0, color="gray", linestyle="--", linewidth=0.8)
        axes[1].set_ylabel("Score moyen (politique gloutonne)")
        axes[1].set_xlabel("Épisodes d'entraînement (log)")
        axes[1].set_title("Progression du score aux checkpoints")
        axes[1].grid(True, alpha=0.3, which="both")

    # --- Subplot 3 : Taille Q-table ---
    if qtable_sizes:
        xs_qt = [s[0] for s in qtable_sizes]
        ys_qt = [s[1] for s in qtable_sizes]
        axes[2].plot(xs_qt, ys_qt, color="#4CAF50", linewidth=1.5)
        axes[2].fill_between(xs_qt, ys_qt, alpha=0.15, color="#4CAF50")
        axes[2].set_ylabel("Nombre d'états dans la Q-table")
        axes[2].set_xlabel("Épisode")
        axes[2].set_title("Croissance de la Q-table (couverture de l'espace d'états)")
        axes[2].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{int(x):,}"))
        axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    fname = f"TabularQ_{env_label.split()[0]}_training.png"
    path  = os.path.join(plots_dir, fname)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  → Graphique training : {path}")


def plot_hp_comparison(hp_results, env_label, plots_dir):
    """
    Graphique comparatif des combinaisons d'hyperparamètres.
    Barres horizontales triées par score final.
    """
    if not hp_results:
        return
    labels = [r["label"] for r in hp_results]
    scores = [r["score_final"] for r in hp_results]

    # Tri par score décroissant
    order  = np.argsort(scores)[::-1]
    labels = [labels[i] for i in order]
    scores = [scores[i] for i in order]
    colors = ["#2196F3" if i == 0 else "#90CAF9" for i in range(len(scores))]

    fig, ax = plt.subplots(figsize=(11, max(4, len(labels) * 0.9)))
    bars = ax.barh(labels, scores, color=colors, edgecolor="white")
    for bar, score in zip(bars, scores):
        ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
                f"{score:.4f}", va="center", fontsize=9)
    ax.set_xlabel("Score moyen (politique gloutonne)")
    ax.set_title(f"TabularQ — {env_label}\nComparaison des hyperparamètres "
                 f"(alpha × epsilon_decay)")
    ax.axvline(0, color="gray", linewidth=0.8, linestyle="--")
    ax.set_xlim(min(scores) - 0.1, max(scores) + 0.15)
    ax.grid(True, alpha=0.3, axis="x")
    plt.tight_layout()
    fname = f"TabularQ_{env_label.split()[0]}_hp_comparison.png"
    path  = os.path.join(plots_dir, fname)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  → Graphique HP : {path}")


def plot_global_comparison(all_results, plots_dir):
    """Heatmap des scores finaux : environnement × configuration."""
    envs   = [e for e in ENVS_LIST if e in all_results]
    if not envs:
        return
    labels = [ENV_CONFIGS[e]["label"].split()[0] for e in envs]
    scores = [all_results[e]["final_eval"]["score_moyen"] for e in envs]
    times  = [all_results[e]["final_eval"]["temps_coup_ms"] for e in envs]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("TabularQ — Résumé global", fontsize=13, fontweight="bold")

    colors = ["#2196F3", "#4CAF50", "#FF9800", "#9C27B0"]
    bars0 = axes[0].bar(labels, scores, color=colors[:len(labels)],
                        width=0.5, edgecolor="white")
    for bar, val in zip(bars0, scores):
        axes[0].text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + 0.01,
                     f"{val:.3f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    axes[0].axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
    axes[0].set_ylabel("Score moyen (politique gloutonne, ε=0)")
    axes[0].set_title("Score moyen par environnement")
    axes[0].set_ylim(min(-0.1, min(scores) - 0.1), max(1.1, max(scores) + 0.15))
    axes[0].grid(True, axis="y", alpha=0.3)

    bars1 = axes[1].bar(labels, times, color=colors[:len(labels)],
                        width=0.5, edgecolor="white")
    for bar, val in zip(bars1, times):
        axes[1].text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + max(times) * 0.01,
                     f"{val:.4f}ms", ha="center", va="bottom", fontsize=9)
    axes[1].set_ylabel("Temps moyen par coup (ms)")
    axes[1].set_title("Temps de décision par environnement")
    axes[1].grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    path = os.path.join(plots_dir, "TabularQ_global_comparison.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  → Graphique global : {path}")


# ══════════════════════════════════════════════════════════════════════════════
#  Affichage console
# ══════════════════════════════════════════════════════════════════════════════

def print_checkpoint_table(eval_results, env_label):
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
#  Entraînement avec suivi de la Q-table
# ══════════════════════════════════════════════════════════════════════════════

def train_with_qtable_tracking(env, agent, train_fn, eval_fn, num_episodes,
                                checkpoints, track_every=1000):
    """
    Lance l'entraînement en enregistrant la taille de la Q-table
    toutes les `track_every` épisodes.
    """
    qtable_sizes = []  # [(episode, taille), ...]
    checkpoints_done = set()
    eval_results = {}
    all_rewards  = []
    checkpoints  = sorted(checkpoints)
    next_check_idx = 0

    # On ré-implémente la boucle d'entraînement pour intercepter qtable_size
    # en appelant directement la bonne fonction de train avec des mini-runs
    step = 0
    ep   = 0

    # On utilise le train_fn en mode "partiel" avec des sous-checkpoints
    sub_cps = list(range(track_every, num_episodes + 1, track_every))
    if not sub_cps or sub_cps[-1] != num_episodes:
        sub_cps.append(num_episodes)
    sub_cps = sorted(set(sub_cps))

    prev_ep = 0
    for sub_cp in sub_cps:
        delta = sub_cp - prev_ep
        # Checkpoints intermédiaires du syllabus dans ce segment
        seg_cps = [c for c in checkpoints if prev_ep < c <= sub_cp]

        rewards_seg, er_seg = train_fn(
            env, agent, delta, seg_cps, eval_fn
        )
        all_rewards.extend(rewards_seg)
        eval_results.update(er_seg)

        qtable_sizes.append((sub_cp, len(agent.q_table)))
        prev_ep = sub_cp

    return all_rewards, eval_results, qtable_sizes


# ══════════════════════════════════════════════════════════════════════════════
#  Recherche d'hyperparamètres
# ══════════════════════════════════════════════════════════════════════════════

def hp_search(env_key, hp_search_episodes, n_eval=200):
    """
    Teste différentes combinaisons (alpha, epsilon_decay) sur un env donné.
    Retourne la meilleure config et les résultats pour le graphique.
    """
    ecfg   = ENV_CONFIGS[env_key]
    elabel = ecfg["label"]
    grid   = HP_GRIDS[env_key]
    base_hp = ecfg["default_hp"].copy()

    print(f"\n  [Recherche HP] {elabel}  ({hp_search_episodes:,} épisodes / config)")
    print(f"  {'alpha':>7} | {'ε_decay':>9} | {'Score final':>12} | "
          f"{'Q-table':>8}")
    print("  " + "-" * 45)

    hp_results = []
    best_score = float("-inf")
    best_hp    = base_hp

    for hp_override in grid:
        hp = {**base_hp, **hp_override}
        env   = ecfg["env_fn"]()
        agent = TabularQAgent(**hp)
        # Checkpoints vides pendant la recherche (pas d'éval intermédiaire)
        rewards, _ = ecfg["train_fn"](env, agent, hp_search_episodes, [], ecfg["eval_fn"])
        # Évaluation finale
        m = ecfg["eval_fn"](env, agent, n_games=n_eval)

        label = f"α={hp['alpha']}  ε_decay={hp['epsilon_decay']}"
        print(f"  {hp['alpha']:>7.3f} | {hp['epsilon_decay']:>9.5f} | "
              f"{m['score_moyen']:>12.4f} | {len(agent.q_table):>8,}")

        hp_results.append({
            "label":       label,
            "hp":          hp,
            "score_final": m["score_moyen"],
            "q_size":      len(agent.q_table),
        })

        if m["score_moyen"] > best_score:
            best_score = m["score_moyen"]
            best_hp    = hp

    best_label = f"α={best_hp['alpha']}  ε_decay={best_hp['epsilon_decay']}"
    print(f"\n  ✓ Meilleure config : {best_label}  →  Score = {best_score:.4f}")
    return best_hp, hp_results


# ══════════════════════════════════════════════════════════════════════════════
#  Expérimentation principale : 1 environnement
# ══════════════════════════════════════════════════════════════════════════════

def run_experiment(env_key, num_episodes, checkpoints, n_eval=500,
                   skip_hp_search=False):
    ecfg   = ENV_CONFIGS[env_key]
    elabel = ecfg["label"]

    env_plots_dir   = os.path.join(PLOTS_DIR,   elabel.split()[0])
    env_results_dir = os.path.join(RESULTS_DIR, elabel.split()[0])
    for d in [env_plots_dir, env_results_dir]:
        os.makedirs(d, exist_ok=True)

    print("\n" + "═" * 70)
    print(f"  EXPÉRIMENTATION TabularQ  —  {elabel}")
    print(f"  Épisodes : {num_episodes:,}   |   Checkpoints : {checkpoints}")
    print(f"  Note     : {ecfg['note']}")
    print("═" * 70)

    # ── 1. Recherche d'hyperparamètres ─────────────────────────────────────
    if skip_hp_search:
        best_hp    = ecfg["default_hp"].copy()
        hp_results = []
        print(f"\n  (Recherche HP ignorée — hyperparamètres par défaut)")
    else:
        best_hp, hp_results = hp_search(
            env_key,
            hp_search_episodes = HP_SEARCH_EPISODES[env_key],
            n_eval             = min(200, n_eval),
        )
        # Graphique comparatif HP
        plot_hp_comparison(hp_results, elabel, env_plots_dir)

    # ── 2. Entraînement complet avec les meilleurs HP ──────────────────────
    print(f"\n  [Entraînement complet]  α={best_hp['alpha']}  "
          f"ε_decay={best_hp['epsilon_decay']}  {num_episodes:,} épisodes")

    env   = ecfg["env_fn"]()
    agent = TabularQAgent(**best_hp)
    t_start = time.time()

    all_rewards, eval_results, qtable_sizes = train_with_qtable_tracking(
        env, agent,
        train_fn     = ecfg["train_fn"],
        eval_fn      = ecfg["eval_fn"],
        num_episodes = num_episodes,
        checkpoints  = checkpoints,
        track_every  = max(1000, num_episodes // 50),
    )

    elapsed = time.time() - t_start
    print(f"\n  Durée : {elapsed:.1f}s ({elapsed / 60:.1f} min)")

    # ── 3. Évaluation finale détaillée ─────────────────────────────────────
    print(f"\n  [Évaluation finale]  {n_eval} parties  ε=0  (politique gloutonne)")
    final_eval = ecfg["eval_fn_full"](agent, env, n_games=n_eval)
    stats      = q_table_stats(agent)

    print_checkpoint_table(eval_results, elabel)
    print_metrics(final_eval, elabel)
    print_q_stats(stats, elabel)

    # ── 4. Graphiques ──────────────────────────────────────────────────────
    plot_training_full(all_rewards, eval_results, qtable_sizes,
                       elabel, env_plots_dir, num_episodes)

    # ── 5. Sauvegarde modèle ───────────────────────────────────────────────
    model_path = os.path.join(MODELS_DIR, f"TabularQ_{elabel.split()[0]}.pkl")
    save_agent(agent, model_path)

    # ── 6. JSON résultats ──────────────────────────────────────────────────
    result_data = {
        "agent":         "TabularQ",
        "env":           elabel,
        "num_episodes":  num_episodes,
        "elapsed_s":     round(elapsed, 1),
        "hyperparams":   best_hp,
        "hp_search":     hp_results,
        "checkpoints":   {str(k): v for k, v in eval_results.items()},
        "final_eval":    final_eval,
        "q_table_stats": stats,
    }
    json_path = os.path.join(env_results_dir, f"TabularQ_{elabel.split()[0]}.json")
    with open(json_path, "w") as f:
        json.dump(result_data, f, indent=2)
    print(f"\n  → JSON : {json_path}")

    return result_data


# ══════════════════════════════════════════════════════════════════════════════
#  Rapport Markdown global
# ══════════════════════════════════════════════════════════════════════════════

def generate_markdown_report(all_results, num_episodes):
    lines = [
        "# Rapport d'expérimentation — TabularQ (Q-Learning tabulaire)\n",
        f"> Épisodes d'entraînement : {num_episodes:,}\n",
        "",
        "## Algorithme\n",
        "Le **Q-Learning tabulaire** stocke une valeur Q(s, a) pour chaque paire "
        "(état, action) rencontrée. La politique est ε-greedy :",
        "- Avec probabilité ε : action aléatoire (exploration)",
        "- Sinon : argmax_a Q(s, a) (exploitation)",
        "",
        "Mise à jour : `Q(s,a) ← Q(s,a) + α × [r + γ × max Q(s',a') − Q(s,a)]`\n",
        "",
        "## Hyperparamètres par défaut\n",
        "| Paramètre | Symbole | Valeur | Rôle |",
        "|---|---|---|---|",
        "| Taux d'apprentissage | α | 0.1–0.3 | Vitesse de mise à jour des Q-valeurs |",
        "| Facteur de discount  | γ | 0.99 | Poids des récompenses futures |",
        "| Epsilon initial      | ε | 1.0 | Exploration pure au début |",
        "| Décroissance epsilon | ε_decay | 0.999–0.9999 | Vitesse de transition vers l'exploitation |",
        "| Epsilon minimum      | ε_min | 0.05 | Exploration résiduelle permanente |",
        "",
        "## Compatibilité avec les environnements\n",
        "| Environnement | Compatible | Raison |",
        "|---|---|---|",
        "| LineWorld | ✓ | Espace d'états discret et très petit |",
        "| GridWorld | ✓ | Encodage one-hot → états distincts |",
        "| TicTacToe | ✓ | ~5 478 états théoriques — Q-table compacte |",
        "| Quarto    | ⚠ | Grand espace d'états → couverture partielle, DQN préférable |",
        "",
        "## Résultats par checkpoint\n",
    ]

    for env_key in ENVS_LIST:
        if env_key not in all_results:
            continue
        data   = all_results[env_key]
        elabel = ENV_CONFIGS[env_key]["label"]
        lines += [
            f"### {elabel}\n",
            "| Épisodes | Score moyen | Longueur moy | Temps/coup (ms) |",
            "|---|---|---|---|",
        ]
        for ep_str, m in sorted(data["checkpoints"].items(), key=lambda x: int(x[0])):
            lines.append(
                f"| {int(ep_str):,} | {m['score_moyen']:.4f} | "
                f"{m['longueur_moy']:.2f} | {m['temps_coup_ms']:.4f} |"
            )
        lines.append("")

    # Score final comparatif
    lines += [
        "## Score final — Politique gloutonne (ε=0)\n",
        "| Environnement | Score moyen | Victoires | Longueur | ms/coup | Q-table |",
        "|---|---|---|---|---|---|",
    ]
    for env_key in ENVS_LIST:
        if env_key not in all_results:
            continue
        data   = all_results[env_key]
        m      = data["final_eval"]
        stats  = data.get("q_table_stats", {})
        elabel = ENV_CONFIGS[env_key]["label"]
        win    = m.get("taux_victoire", m.get("taux_succes", float("nan")))
        lines.append(
            f"| {elabel} | {m['score_moyen']:.4f} | {win:.1%} | "
            f"{m['longueur_moy']:.2f} | {m['temps_coup_ms']:.4f} | "
            f"{stats.get('n_etats', 0):,} |"
        )

    # Analyse et interprétation
    lines += [
        "",
        "## Analyse des résultats\n",
        "### Comportement de l'exploration (ε-greedy)",
        "- Au début : ε=1.0 → exploration pure. L'agent découvre l'espace d'états.",
        "- Progressivement : ε décroît → exploitation de plus en plus prioritaire.",
        "- En fin d'entraînement : ε≈ε_min → quasi-exploitation, "
          "évaluation proche de la politique optimale.",
        "",
        "### Impact de alpha (taux d'apprentissage)",
        "- α élevé (0.3–0.5) : convergence rapide mais instabilité possible.",
        "- α faible (0.05–0.1) : convergence lente mais plus stable.",
        "- Optimum trouvé par recherche d'hyperparamètres pour chaque env.",
        "",
        "### Croissance de la Q-table",
        "- LineWorld / GridWorld : Q-table se stabilise rapidement "
          "(espace d'états petit).",
        "- TicTacToe : croissance modérée jusqu'à couvrir ~5k états.",
        "- Quarto : Q-table continue de croître — couverture partielle "
          "de l'espace. C'est la limite principale de TabularQ sur cet env.",
        "",
        "### Comparaison avec les autres agents",
        "- **vs RandomRollout** : TabularQ est plus rapide à l'inférence "
          "(lookup O(1) vs simulations).",
        "- **vs DQN** : TabularQ est optimal sur petits espaces d'états "
          "(pas besoin d'approximation).",
        "- **vs AlphaZero** : TabularQ ne généralise pas — chaque état "
          "vu pour la première fois obtient Q=0.",
        "",
        "### Limites",
        "- Ne généralise pas aux états non vus (contrairement aux réseaux neuronaux).",
        "- Mémoire proportionnelle au nombre d'états visités.",
        "- Inapplicable aux environnements à espace d'états continu.",
        "- Sur Quarto : très grand espace d'états → couverture insuffisante "
          "même après 100k épisodes.",
    ]

    md_path = os.path.join(RESULTS_DIR, "rapport_tabular_q.md")
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
    parser = argparse.ArgumentParser(
        description="Expérimentations complètes TabularQ — 4 environnements"
    )
    parser.add_argument("--env",      default="all",
                        choices=["all"] + ENVS_LIST,
                        help="Environnement cible (default: all)")
    parser.add_argument("--episodes", type=int, default=10_000,
                        help="Épisodes d'entraînement (default: 10000)")
    parser.add_argument("--n_eval",   type=int, default=500,
                        help="Parties pour l'évaluation finale (default: 500)")
    parser.add_argument("--skip_hp_search", action="store_true",
                        help="Ignore la recherche d'hyperparamètres (plus rapide)")
    args = parser.parse_args()

    envs        = ENVS_LIST if args.env == "all" else [args.env]
    checkpoints = build_checkpoints(args.episodes)

    print(f"\n{'═' * 70}")
    print(f"  EXPÉRIMENTATIONS TabularQ")
    print(f"  Environnements : {[ENV_CONFIGS[e]['label'] for e in envs]}")
    print(f"  Épisodes       : {args.episodes:,}")
    print(f"  Checkpoints    : {checkpoints}")
    print(f"  Éval (parties) : {args.n_eval}")
    print(f"  Recherche HP   : {'non' if args.skip_hp_search else 'oui'}")
    print(f"{'═' * 70}")

    global_t0   = time.time()
    all_results = {}

    for env_key in envs:
        data = run_experiment(
            env_key,
            num_episodes   = args.episodes,
            checkpoints    = checkpoints,
            n_eval         = args.n_eval,
            skip_hp_search = args.skip_hp_search,
        )
        all_results[env_key] = data

    # Graphique comparatif global
    if len(all_results) > 1:
        print("\n  Génération du graphique comparatif global ...")
        plot_global_comparison(all_results, PLOTS_DIR)

    # Rapport Markdown
    generate_markdown_report(all_results, args.episodes)

    # JSON global
    json_path = os.path.join(RESULTS_DIR, "tabular_q_summary.json")
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"  → JSON synthèse : {json_path}")

    # Résumé console final
    last_cp = max(checkpoints)
    total   = time.time() - global_t0
    print(f"\n{'═' * 80}")
    print(f"  RÉSUMÉ GLOBAL TabularQ — Score moyen au checkpoint {last_cp:,}")
    print(f"{'═' * 80}")
    header = f"  {'Environnement':>22} | Score moyen | Longueur | ms/coup | Q-table"
    print(header)
    print("  " + "─" * (len(header) - 2))
    for env_key in envs:
        data   = all_results.get(env_key, {})
        cps    = data.get("checkpoints", {})
        stats  = data.get("q_table_stats", {})
        label  = ENV_CONFIGS[env_key]["label"]
        if cps:
            last_key = str(max(int(k) for k in cps))
            m = cps[last_key]
            print(f"  {label:>22} | {m['score_moyen']:>11.4f} | "
                  f"{m['longueur_moy']:>8.2f} | {m['temps_coup_ms']:>7.4f}ms | "
                  f"{stats.get('n_etats', 0):>8,}")

    print(f"\n  Durée totale : {total:.0f}s ({total / 60:.1f} min)")
    print(f"  Modèles      → {MODELS_DIR}")
    print(f"  Résultats    → {RESULTS_DIR}")
    print(f"  Graphiques   → {PLOTS_DIR}")
    print(f"{'═' * 80}\n")


if __name__ == "__main__":
    main()
