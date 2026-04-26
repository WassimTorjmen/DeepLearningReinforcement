import numpy as np
import copy
import math
import time
import torch
import torch.nn as nn
import torch.optim as optim


# ╔═════════════════════════════════════════════════════════════╗
# ║  EXPERT  —  MCTS (identique à mcts.py)                     ║
# ╚═════════════════════════════════════════════════════════════╝

class MCTSNode:
    def __init__(self, action=None, parent=None):
        self.action   = action
        self.parent   = parent
        self.children = []
        self.visits   = 0
        self.value    = 0.0

    def is_fully_expanded(self, available_actions):
        return len(self.children) == len(available_actions)

    def uct_score(self, c=math.sqrt(2)):
        if self.visits == 0:
            return float('inf')
        exploitation = self.value / self.visits
        exploration  = c * math.sqrt(math.log(self.parent.visits) / self.visits)
        return exploitation + exploration

    def best_child(self):
        return max(self.children, key=lambda n: n.uct_score())

    def best_action_child(self):
        return max(self.children, key=lambda n: n.visits)


class MCTSExpert:
    """L'expert — utilise MCTS pour choisir la meilleure action."""

    def __init__(self, n_simulations=100):
        self.n_simulations = n_simulations

    def select_action(self, env, available_actions):
        root = MCTSNode()

        for _ in range(self.n_simulations):
            node    = root
            sim_env = copy.deepcopy(env)

            # sélection
            while node.is_fully_expanded(sim_env.get_actions()) and node.children:
                node = node.best_child()
                sim_env.step(node.action)
                if sim_env.done:
                    break

            # expansion
            if not sim_env.done:
                actions_tried = [child.action for child in node.children]
                actions_left  = [a for a in sim_env.get_actions() if a not in actions_tried]
                if actions_left:
                    action = int(np.random.choice(actions_left))
                    child  = MCTSNode(action=action, parent=node)
                    node.children.append(child)
                    node = child
                    sim_env.step(action)

            # simulation
            while not sim_env.done:
                available = sim_env.get_actions()
                if not available:
                    break
                sim_env.step(int(np.random.choice(available)))

            # score
            score = self._get_score(sim_env)

            # backprop
            while node is not None:
                node.visits += 1
                node.value  += score
                node         = node.parent

        if root.children:
            return root.best_action_child().action
        return int(np.random.choice(available_actions))

    def _get_score(self, env):
        if hasattr(env, 'winner'):
            if env.winner == 1:                     return 1.0
            if env.winner == 2 or env.winner == -1: return -1.0
            return 0.0
        if hasattr(env, 'agent_position') and env.done:
            return 1.0 if env.agent_position == env.goal_position else 0.0
        return 0.0


# ╔═════════════════════════════════════════════════════════════╗
# ║  APPRENTI  —  réseau qui imite l'expert                    ║
# ╚═════════════════════════════════════════════════════════════╝

class ApprenticeNetwork(nn.Module):
    """
    Réseau de l'apprenti.

    Entrée  : vecteur d'état encode_state()
    Sortie  : probabilité sur chaque action (comme PolicyNetwork)

    Le réseau apprend à prédire l'action choisie par MCTS.
    """
    def __init__(self, state_size, num_actions, hidden_size=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_actions),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        return self.net(x)


class ExpertApprenticeAgent:
    """
    Expert Apprentice.

    Phase 1 — Collecte de données :
        L'expert (MCTS) joue et on enregistre (état, action_expert).

    Phase 2 — Apprentissage :
        Le réseau apprenti apprend à prédire l'action de l'expert
        via une Cross Entropy Loss (classification supervisée).

    Après entraînement, l'apprenti joue seul sans MCTS.

    Paramètres :
        state_size     : taille du vecteur d'état
        num_actions    : nombre d'actions possibles
        n_simulations  : simulations MCTS par coup (qualité de l'expert)
        hidden_size    : taille du réseau apprenti
        lr             : learning rate
    """

    def __init__(self, state_size, num_actions, n_simulations=50,
                 hidden_size=64, lr=1e-3):
        self.num_actions  = num_actions
        self.device       = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # l'expert MCTS
        self.expert       = MCTSExpert(n_simulations=n_simulations)

        # le réseau apprenti
        self.apprentice   = ApprenticeNetwork(state_size, num_actions, hidden_size).to(self.device)
        self.optimizer    = optim.Adam(self.apprentice.parameters(), lr=lr)

        # Cross Entropy Loss : mesure l'écart entre la prédiction et l'action de l'expert
        self.loss_fn      = nn.CrossEntropyLoss()

        # buffer de données (état, action_expert)
        self.states_buffer  = []
        self.actions_buffer = []

    def select_action_expert(self, env, available_actions):
        """
        L'expert (MCTS) choisit une action ET on la stocke avec l'état.
        Utilisé pendant la phase d'entraînement.
        """
        state  = env.encode_state()
        action = self.expert.select_action(env, available_actions)

        # on stocke le couple (état, action_expert) pour l'apprentissage
        self.states_buffer.append(state)
        self.actions_buffer.append(action)

        return action

    def select_action(self, state, available_actions):
        """
        L'apprenti (réseau) choisit une action.
        Utilisé pendant l'évaluation (sans MCTS).
        """
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        probs   = self.apprentice(state_t).squeeze(0)

        # masque les actions illégales
        mask    = torch.zeros(self.num_actions, device=self.device)
        mask[available_actions] = 1.0
        probs   = probs * mask
        probs   = probs / (probs.sum() + 1e-8)

        return int(probs.argmax().item())

    def store_reward(self, reward):
        """Pas utilisé — l'apprentissage est supervisé, pas par récompense."""
        pass

    def learn(self):
        """
        Entraîne l'apprenti sur les données collectées par l'expert.

        Loss = CrossEntropy(prédiction_apprenti, action_expert)
        → le réseau apprend à prédire exactement l'action de MCTS.
        """
        if not self.states_buffer:
            return 0.0

        states  = torch.FloatTensor(self.states_buffer).to(self.device)
        actions = torch.LongTensor(self.actions_buffer).to(self.device)

        # prédiction de l'apprenti sur tous les états collectés
        logits = self.apprentice(states)     # (N, num_actions)

        # loss : à quel point l'apprenti s'éloigne de l'expert
        loss = self.loss_fn(logits, actions)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.apprentice.parameters(), max_norm=1.0)
        self.optimizer.step()

        # vide le buffer après apprentissage
        self.states_buffer  = []
        self.actions_buffer = []

        return loss.item()

    def save(self, path):
        torch.save(self.apprentice.state_dict(), path)

    def load(self, path):
        self.apprentice.load_state_dict(torch.load(path, map_location=self.device))
        self.apprentice.eval()