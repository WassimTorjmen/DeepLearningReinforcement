"""
MCTS (Monte Carlo Tree Search) avec UCT.
Construit un arbre de recherche : à chaque décision, fait n_simulations
itérations Sélection → Expansion → Simulation → Backpropagation.
Plus efficace que Random Rollout car réutilise les résultats passés.
"""

import numpy as np
import copy
import math
import time


# ╔═════════════════════════════════════════════════════════════╗
# ║  NOEUD DE L'ARBRE MCTS                                     ║
# ╚═════════════════════════════════════════════════════════════╝

class MCTSNode:
    """
    Un noeud dans l'arbre MCTS.

    Chaque noeud représente un état du jeu et stocke :
        - l'action qui a mené à cet état
        - le nombre de fois qu'on l'a visité
        - la somme des scores obtenus depuis ce noeud
        - ses enfants (états suivants)
        - son parent
    """

    def __init__(self, action=None, parent=None):
        self.action   = action   # action qui a mené à ce noeud
        self.parent   = parent   # noeud parent
        self.children = []       # noeuds enfants

        self.visits = 0    # nombre de fois visité
        self.value  = 0.0  # somme des scores obtenus

    def is_fully_expanded(self, available_actions):
        """Vrai si tous les enfants possibles ont été créés."""
        return len(self.children) == len(available_actions)

    def uct_score(self, c=math.sqrt(2)):
        """
        Formule UCT pour choisir quel noeud explorer.

        UCT = score_moyen + C * sqrt(log(N_parent) / N_enfant)
                ↑                        ↑
           exploitation              exploration

        - score_moyen élevé → noeud qui a bien marché
        - sqrt(log(N)/n) élevé → noeud peu exploré
        """
        if self.visits == 0:
            return float('inf')   # noeud jamais visité → priorité absolue

        exploitation = self.value / self.visits
        exploration  = c * math.sqrt(math.log(self.parent.visits) / self.visits)
        return exploitation + exploration

    def best_child(self):
        """Retourne l'enfant avec le score UCT le plus élevé."""
        return max(self.children, key=lambda n: n.uct_score())

    def best_action_child(self):
        """Retourne l'enfant le plus visité (utilisé pour choisir l'action finale)."""
        return max(self.children, key=lambda n: n.visits)


# ╔═════════════════════════════════════════════════════════════╗
# ║  AGENT MCTS                                                 ║
# ╚═════════════════════════════════════════════════════════════╝

class MCTSAgent:
    """
    Monte Carlo Tree Search avec UCT.

    Pour chaque décision, on effectue `n_simulations` itérations de :
        1. SELECTION  : descend dans l'arbre via UCT
        2. EXPANSION  : ajoute un nouveau noeud non exploré
        3. SIMULATION : joue aléatoirement jusqu'à la fin
        4. BACKPROP   : remonte le résultat dans l'arbre

    Plus fort que RandomRollout car il mémorise les résultats
    des simulations précédentes dans l'arbre.

    Paramètres :
        n_simulations : nombre d'itérations par décision
        c             : constante UCT (équilibre exploration/exploitation)
    """

    def __init__(self, n_simulations=100, c=math.sqrt(2)):
        self.n_simulations = n_simulations
        self.c             = c

    def select_action(self, env, available_actions):
        """
        Lance n_simulations itérations MCTS et retourne la meilleure action.
        """
        root = MCTSNode()   # noeud racine = état courant

        for _ in range(self.n_simulations):

            # ── 1. SELECTION ────────────────────────────────────
            # on descend dans l'arbre en suivant UCT
            node    = root
            sim_env = copy.deepcopy(env)

            while node.is_fully_expanded(sim_env.get_actions()) and node.children:
                node   = node.best_child()
                sim_env.step(node.action)
                if sim_env.done:
                    break

            # ── 2. EXPANSION ─────────────────────────────────────
            # on ajoute un enfant non encore exploré
            if not sim_env.done:
                actions_tried = [child.action for child in node.children]
                actions_left  = [a for a in sim_env.get_actions() if a not in actions_tried]

                if actions_left:
                    action   = int(np.random.choice(actions_left))
                    child    = MCTSNode(action=action, parent=node)
                    node.children.append(child)
                    node = child
                    sim_env.step(action)

            # ── 3. SIMULATION ────────────────────────────────────
            # on joue aléatoirement jusqu'à la fin depuis ce noeud
            while not sim_env.done:
                available = sim_env.get_actions()
                if not available:
                    break
                sim_env.step(int(np.random.choice(available)))

            score = self._get_score(sim_env)

            # ── 4. BACKPROPAGATION ───────────────────────────────
            # on remonte le score dans tous les noeuds visités
            while node is not None:
                node.visits += 1
                node.value  += score
                node         = node.parent

        # on retourne l'action de l'enfant le plus visité
        if root.children:
            return root.best_action_child().action
        return int(np.random.choice(available_actions))

    def _get_score(self, env):
        """Retourne le score final de la simulation."""
        if hasattr(env, 'winner'):
            if env.winner == 1:                    return 1.0
            if env.winner == 2 or env.winner == -1: return -1.0
            return 0.0
        if hasattr(env, 'agent_position') and env.done:
            if env.agent_position == env.goal_position: return 1.0
            return 0.0
        return 0.0

    # méthodes vides pour compatibilité avec experiment.py
    def store_reward(self, reward):
        pass

    def learn(self):
        return 0.0

    def save(self, path):
        pass

    def load(self, path):
        pass