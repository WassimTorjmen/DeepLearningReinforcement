"""
evaluate_agents.py
==================
Évaluation des agents sur les 4 environnements APRÈS entraînement.

Ce fichier est distinct de train_agents.py :
  - train_agents.py  → entraîne + évalue en cours d'entraînement (politique en apprentissage)
  - evaluate_agents.py → charge un modèle sauvegardé + évalue la POLITIQUE FINALE (ε=0, greedy)

Il produit :
  - Score moyen, longueur moyenne, temps de décision par coup
  - Distribution victoires / nuls / défaites (TicTacToe, Quarto)
  - Taux de succès (LineWorld, GridWorld)
  - Tableau comparatif final de tous les agents
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

# ── Chemins des dossiers ──────────────────────────────────────────────────────
BASE     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS   = os.path.join(BASE, "models")   # Contient les poids sauvegardés (.pt)
RESULTS  = os.path.join(BASE, "results")  # Destination des JSON de résultats
os.makedirs(RESULTS, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
#  CONFIGURATIONS
#
#  Ces dictionnaires définissent comment instancier chaque agent et
#  chaque environnement. On les centralise ici pour éviter la duplication.
# ══════════════════════════════════════════════════════════════════════════════

AGENT_CONFIGS = {
    "randomrollout": {
        "label":     "RandomRollout",
        "has_model": False,   # Pas de poids à charger (algorithme pur, sans réseau)
        # Instancie avec 50 rollouts : pour chaque action, simule 50 parties aléatoires
        "fn":        lambda s, a: RandomRolloutAgent(n_rollouts=50),
    },
    "mcts": {
        "label":     "MCTS_UCT",
        "has_model": False,   # Pas de réseau : construit l'arbre à chaque décision
        # 100 simulations = 100 nœuds explorés dans l'arbre UCT par coup
        "fn":        lambda s, a: MCTSAgent(n_simulations=100),
    },
    "expertapprentice": {
        "label":     "ExpertApprentice",
        "has_model": True,    # Possède un réseau de politique entraîné (distillation MCTS)
        "fn":        lambda s, a: ExpertApprenticeAgent(
            state_size=s, num_actions=a, n_simulations=50, hidden_size=128, lr=1e-3
        ),
    },
    "muzero": {
        "label":     "MuZero",
        "has_model": True,    # 3 réseaux appris : représentation, dynamique, prédiction
        "fn":        lambda s, a: MuZeroAgent(
            state_size=s, num_actions=a, hidden_size=128, n_simulations=10, lr=1e-3
        ),
    },
    "muzero_stochastic": {
        "label":     "MuZeroStochastic",
        "has_model": True,    # MuZero + encodeur VAE stochastique
        "fn":        lambda s, a: MuZeroStochasticAgent(
            state_size=s, num_actions=a, hidden_size=128, chance_size=8,
            n_simulations=10, lr=1e-3
        ),
    },
    "alphazero": {
        "label":     "AlphaZero",
        "has_model": True,    # Réseau de politique + valeur entraîné par self-play
        # Hyperparamètres DIFFÉRENTS selon l'environnement (doivent correspondre
        # exactement à ce qui a été utilisé à l'entraînement pour charger correctement)
        "az_configs": {
            "lineworld": {"hidden_size": 64,  "n_simulations": 20, "c_puct": 1.0},
            "gridworld": {"hidden_size": 128, "n_simulations": 20, "c_puct": 1.0},
            "tictactoe": {"hidden_size": 128, "n_simulations": 50, "c_puct": 1.5},
            "quarto":    {"hidden_size": 256, "n_simulations": 30, "c_puct": 1.5},
        },
        # fn générique (utilisé uniquement si l'env n'est pas dans az_configs)
        "fn":        lambda s, a: AlphaZeroAgent(
            state_size=s, num_actions=a, hidden_size=128, n_simulations=20,
        ),
    },
}

ENV_CONFIGS = {
    "lineworld": {
        "label":       "LineWorld",
        "state_size":  8,
        "num_actions": 2,
        "type":        "1player",     # Détermine quelle fonction d'éval appeler
        "env_fn":      lambda: LineWorld(size=6),
    },
    "gridworld": {
        "label":       "GridWorld",
        "state_size":  31,
        "num_actions": 4,
        "type":        "1player",
        "env_fn":      lambda: GridWorld(rows=5, cols=5),
    },
    "tictactoe": {
        "label":       "TicTacToe",
        "state_size":  27,
        "num_actions": 9,
        "type":        "tictactoe",
        "env_fn":      lambda: TicTacToe(),
    },
    "quarto": {
        "label":       "Quarto",
        "state_size":  105,
        "num_actions": 32,
        "type":        "quarto",
        "env_fn":      lambda: QuartoEnv(),
    },
}


# ══════════════════════════════════════════════════════════════════════════════
#  FONCTIONS D'ÉVALUATION DÉTAILLÉES
#
#  Différence clé avec train_agents.py :
#  - Ici on mesure plus de métriques (victoires/nuls/défaites)
#  - use_mcts=True : pour AlphaZero, utilise MCTS pendant l'éval
#    → sélectionne les actions via select_action_mcts() au lieu de select_action()
#    → plus lent mais plus fort (planification complète vs réseau seul)
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_1player(agent, env, n_games=500, max_steps=200, use_mcts=False):
    """
    Évaluation sur LineWorld ou GridWorld.

    Paramètre use_mcts :
      - False (défaut) : select_action() → réseau seul, inférence rapide
      - True (AlphaZero) : select_action_mcts() → MCTS guidé par le réseau

    Après select_action_mcts(), on vide les buffers internes de l'agent
    (states_buffer, pi_buffer) pour éviter de polluer la prochaine décision.

    Métriques additionnelles vs train_agents.py :
      taux_victoire : proportion d'épisodes réussis (reward > 0)
      taux_echec    : proportion d'épisodes échoués
    """
    scores, lengths, times = [], [], []

    for _ in range(n_games):
        env.reset()
        steps = 0
        reward = 0

        while not env.done and steps < max_steps:
            available = env.get_actions()
            t0 = time.perf_counter()

            # Sélection de l'action selon le mode (réseau seul ou MCTS complet)
            if use_mcts and hasattr(agent, "select_action_mcts"):
                action = agent.select_action_mcts(env, available)
                # Nettoyage des buffers : indispensable entre les épisodes
                if hasattr(agent, "states_buffer"):
                    agent.states_buffer.clear()
                    agent.pi_buffer.clear()
            else:
                action = agent.select_action(env, available)

            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000)  # ms
            _, reward, _ = env.step(action)
            steps += 1

        scores.append(reward)
        lengths.append(steps)

    wins = sum(1 for s in scores if s > 0)
    return {
        "score_moyen":   round(float(np.mean(scores)), 4),
        "longueur_moy":  round(float(np.mean(lengths)), 2),
        "temps_coup_ms": round(float(np.mean(times)), 4),
        "taux_victoire": round(wins / n_games, 4),          # Métrique additionnelle
        "taux_echec":    round(1 - wins / n_games, 4),      # Métrique additionnelle
        "n_games":       n_games,
    }


def evaluate_tictactoe(agent, env, n_games=500, use_mcts=False):
    """
    Évaluation sur TicTacToe (agent = joueur 1, adversaire = random).

    Pour chaque partie :
      - Tour agent  → chronomètre + sélection d'action
      - Vérif fin   → enregistre victoire/défaite/nul
      - Tour random → np.random.choice() (pas de chrono)
      - Vérif fin   → enregistre victoire/défaite/nul

    Retourne 3 taux distincts : victoire, nul, défaite (en plus du score moyen)
    Ce découpage est important pour analyser le comportement de l'agent
    (préfère-t-il éviter les défaites ou maximiser les victoires ?)
    """
    scores, lengths, times = [], [], []
    wins = draws = losses = 0  # Compteurs pour les 3 issues possibles

    for _ in range(n_games):
        env.reset()
        steps = 0
        result = 0

        while not env.done:
            available = env.get_actions()

            # Tour de l'agent (joueur 1)
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
                # Lecture du gagnant depuis info["winner"]
                if info["winner"] == 1:   result = 1;  wins   += 1
                elif info["winner"] == 0: result = 0;  draws  += 1
                else:                     result = -1; losses += 1
                break

            # Tour de l'adversaire aléatoire (joueur -1)
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
    """
    Évaluation sur Quarto (agent = joueur 1, adversaire = random joueur 2).

    Différence avec TicTacToe : le joueur courant est identifié par env.current_player
    (valeur 1 ou 2) → on conditionne la sélection d'action sur cette valeur.

    Le chronomètre ne tourne que pour l'agent (joueur 1), pas pour l'adversaire.

    env.winner : 1 = agent gagne, 2 = adversaire gagne, 0 = match nul
    """
    scores, lengths, times = [], [], []
    wins = draws = losses = 0

    for _ in range(n_games):
        env.reset()
        steps = 0
        result = 0

        while not env.done:
            available = env.get_actions()
            if env.current_player == 1:
                # Tour de l'agent → chrono + sélection intelligente
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
                # Tour de l'adversaire aléatoire → int() pour convertir numpy → Python
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


# Dispatcher : associe chaque type d'environnement à sa fonction d'évaluation
EVAL_FNS = {
    "1player":   evaluate_1player,
    "tictactoe": evaluate_tictactoe,
    "quarto":    evaluate_quarto,
}


# ══════════════════════════════════════════════════════════════════════════════
#  CHARGEMENT DE MODÈLE
#
#  Pour les agents avec réseau, on charge les poids depuis le fichier .pt
#  sauvegardé lors de l'entraînement.
#  Si le fichier n'existe pas → l'agent utilise ses poids initiaux (random)
#  et on affiche un avertissement.
# ══════════════════════════════════════════════════════════════════════════════

def load_agent(agent_key, env_key):
    """
    Instancie l'agent et charge ses poids sauvegardés si disponibles.

    Logique de chemin :
      - AlphaZero → models/alphazero/AlphaZero_<Env>.pt
      - Autres    → models/<AgentLabel>/<AgentLabel>_<Env>.pt

    Retourne : instance de l'agent (avec ou sans poids chargés)
    """
    acfg  = AGENT_CONFIGS[agent_key]
    ecfg  = ENV_CONFIGS[env_key]

    # ── Instanciation avec les bons hyperparamètres ───────────────────────
    if agent_key == "alphazero":
        # AlphaZero a des hyperparamètres différents par environnement
        # Il faut utiliser exactement les mêmes que lors de l'entraînement
        # sinon l'architecture du réseau ne correspond pas au fichier .pt
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

    # ── Chargement des poids (seulement pour les agents avec réseau) ──────
    if acfg["has_model"]:
        # Construction du chemin selon la convention de nommage de train_agents.py
        if agent_key == "alphazero":
            model_path = os.path.join(MODELS, "alphazero",
                                      f"AlphaZero_{ecfg['label']}.pt")
        else:
            model_path = os.path.join(MODELS, acfg["label"],
                                      f"{acfg['label']}_{ecfg['label']}.pt")

        if os.path.isfile(model_path):
            agent.load(model_path)   # Charge les poids PyTorch
            print(f"  ✓ Modèle chargé : {model_path}")
        else:
            # Pas de modèle → réseau non entraîné (poids aléatoires)
            # L'évaluation sera mauvaise mais ne plante pas
            print(f"  ⚠ Modèle introuvable : {model_path}  (réseau non entraîné)")

    return agent


# ══════════════════════════════════════════════════════════════════════════════
#  AFFICHAGE CONSOLE
# ══════════════════════════════════════════════════════════════════════════════

def print_metrics(m, agent_label, env_label):
    """
    Affiche les métriques d'une évaluation de manière lisible.
    Affiche conditionnellement les métriques disponibles selon l'environnement :
      - taux_victoire/taux_nul/taux_defaite → jeux adversariaux (TicTacToe, Quarto)
      - taux_echec → environnements 1 joueur (LineWorld, GridWorld)
    """
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
#  POINT D'ENTRÉE
# ══════════════════════════════════════════════════════════════════════════════

AGENTS_LIST = ["randomrollout", "mcts", "expertapprentice", "muzero", "muzero_stochastic", "alphazero"]
ENVS_LIST   = ["lineworld", "gridworld", "tictactoe", "quarto"]


def main():
    parser = argparse.ArgumentParser(description="Évaluation des 5 agents")
    parser.add_argument("--agent",    default="all", choices=["all"] + AGENTS_LIST)
    parser.add_argument("--env",      default="all", choices=["all"] + ENVS_LIST)
    parser.add_argument("--n_games",  type=int, default=500,
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

    # Double boucle : évalue chaque agent sur chaque environnement
    for agent_key in agents:
        all_metrics[agent_key] = {}
        for env_key in envs:
            ecfg    = ENV_CONFIGS[env_key]
            acfg    = AGENT_CONFIGS[agent_key]
            agent   = load_agent(agent_key, env_key)
            env     = ecfg["env_fn"]()
            # Sélection de la bonne fonction d'éval selon le type d'environnement
            eval_fn = EVAL_FNS[ecfg["type"]]

            # AlphaZero peut recevoir use_mcts=True pour activer le MCTS pendant l'éval
            if agent_key == "alphazero":
                m = eval_fn(agent, env, n_games=args.n_games, use_mcts=args.use_mcts)
            else:
                m = eval_fn(agent, env, n_games=args.n_games)

            print_metrics(m, acfg["label"], ecfg["label"])
            all_metrics[agent_key][env_key] = m

    # ── Tableau comparatif scores ─────────────────────────────────────────
    # Permet de comparer visuellement tous les agents en un coup d'œil
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

    # ── Tableau comparatif temps/coup ─────────────────────────────────────
    # Crucial pour comprendre le compromis qualité/vitesse de chaque agent
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

    # ── Sauvegarde JSON ───────────────────────────────────────────────────
    # Le JSON contient toutes les métriques pour exploitation ultérieure
    # (graphiques, rapport, comparaison avec d'autres runs)
    out_path = os.path.join(RESULTS, "comparison_agents.json")
    with open(out_path, "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"  → Résultats : {out_path}")


if __name__ == "__main__":
    main()