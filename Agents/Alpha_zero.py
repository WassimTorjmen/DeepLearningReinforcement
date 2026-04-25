import numpy as np
import copy
import math
import torch
import torch.nn as nn
import torch.optim as optim


class AlphaZeroNetwork(nn.Module):
    def __init__(self, state_size, num_actions, hidden_size=128):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_size, num_actions),
            nn.Softmax(dim=-1)
        )
        self.value_head = nn.Sequential(
            nn.Linear(hidden_size, 1),
            nn.Tanh()
        )

    def forward(self, x):
        trunk  = self.trunk(x)
        policy = self.policy_head(trunk)
        value  = self.value_head(trunk).squeeze(-1)
        return policy, value


class AlphaZeroNode:
    def __init__(self, action=None, parent=None, prior=0.0):
        self.action   = action
        self.parent   = parent
        self.children = []
        self.visits   = 0
        self.value    = 0.0
        self.prior    = prior

    def is_fully_expanded(self, available_actions):
        return len(self.children) == len(available_actions)

    def puct_score(self, c=1.0):
        if self.visits == 0:
            q = 0.0
        else:
            q = self.value / self.visits
        u = c * self.prior * math.sqrt(self.parent.visits + 1) / (1 + self.visits)
        return q + u

    def best_child(self):
        return max(self.children, key=lambda n: n.puct_score())

    def best_action_child(self):
        return max(self.children, key=lambda n: n.visits)


class AlphaZeroAgent:
    """
    AlphaZero.

    Différences avec Expert Apprentice :
    1. Le réseau guide MCTS via les probabilités a priori (PUCT)
    2. La simulation aléatoire est remplacée par V(s) du réseau
    3. Le réseau s'améliore en boucle (self-play)
    """

    def __init__(self, state_size, num_actions, n_simulations=50,
                 hidden_size=128, lr=1e-3, c_puct=1.0):
        self.num_actions   = num_actions
        self.n_simulations = n_simulations
        self.c_puct        = c_puct
        self.device        = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.network       = AlphaZeroNetwork(state_size, num_actions, hidden_size).to(self.device)
        self.optimizer     = optim.Adam(self.network.parameters(), lr=lr)
        self.states_buffer  = []
        self.pi_buffer      = []
        self.rewards_buffer = []

    def _get_policy_value(self, env, available_actions):
        state   = env.encode_state()
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            policy, value = self.network(state_t)
        policy = policy.squeeze(0).cpu().numpy()
        mask   = np.zeros(self.num_actions)
        mask[available_actions] = 1.0
        policy = policy * mask
        if policy.sum() > 0:
            policy = policy / policy.sum()
        else:
            policy[available_actions] = 1.0 / len(available_actions)
        return policy, value.item()

    def _mcts(self, env, available_actions):
        root   = AlphaZeroNode()
        policy, _ = self._get_policy_value(env, available_actions)
        for action in available_actions:
            child = AlphaZeroNode(action=action, parent=root, prior=policy[action])
            root.children.append(child)

        for _ in range(self.n_simulations):
            node    = root
            sim_env = copy.deepcopy(env)

            while node.children and not sim_env.done:
                node = node.best_child()
                sim_env.step(node.action)

            if sim_env.done:
                score = self._get_score(sim_env)
            else:
                av = sim_env.get_actions()
                if av:
                    _, score        = self._get_policy_value(sim_env, av)
                    policy_child, _ = self._get_policy_value(sim_env, av)
                    for action in av:
                        child = AlphaZeroNode(action=action, parent=node, prior=policy_child[action])
                        node.children.append(child)
                else:
                    score = 0.0

            while node is not None:
                node.visits += 1
                node.value  += score
                node         = node.parent

        pi = np.zeros(self.num_actions)
        for child in root.children:
            pi[child.action] = child.visits
        if pi.sum() > 0:
            pi = pi / pi.sum()
        return pi

    def select_action(self, state_or_env, available_actions):
        """
        Utilisé à l'évaluation.
        CORRIGÉ : accepte soit un vecteur d'état, soit un env directement.
        """
        # si on reçoit un env, on encode l'état
        if hasattr(state_or_env, 'encode_state'):
            state = state_or_env.encode_state()
        else:
            state = state_or_env

        state_t = torch.FloatTensor(np.array(state)).unsqueeze(0).to(self.device)
        with torch.no_grad():
            policy, _ = self.network(state_t)
        policy = policy.squeeze(0)
        mask   = torch.zeros(self.num_actions)
        mask[available_actions] = 1.0
        policy = policy * mask
        policy = policy / (policy.sum() + 1e-8)
        return int(policy.argmax().item())

    def select_action_mcts(self, env, available_actions):
        """Utilisé pendant l'entraînement — MCTS guidé par le réseau."""
        state = env.encode_state()
        pi    = self._mcts(env, available_actions)
        self.states_buffer.append(state)
        self.pi_buffer.append(pi)
        action = int(np.random.choice(self.num_actions, p=pi))
        return action

    def store_reward(self, reward):
        n = len(self.states_buffer) - len(self.rewards_buffer)
        self.rewards_buffer.extend([reward] * n)

    def learn(self):
        if not self.states_buffer or len(self.states_buffer) != len(self.rewards_buffer):
            self.states_buffer  = []
            self.pi_buffer      = []
            self.rewards_buffer = []
            return 0.0, 0.0

        # CORRIGÉ : conversion numpy avant FloatTensor pour éviter le warning
        states  = torch.FloatTensor(np.array(self.states_buffer)).to(self.device)
        pi_mcts = torch.FloatTensor(np.array(self.pi_buffer)).to(self.device)
        rewards = torch.FloatTensor(np.array(self.rewards_buffer)).to(self.device)

        policy_pred, value_pred = self.network(states)
        policy_loss = -(pi_mcts * torch.log(policy_pred + 1e-8)).sum(dim=1).mean()
        value_loss  = nn.MSELoss()(value_pred, rewards)
        loss        = policy_loss + value_loss

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=1.0)
        self.optimizer.step()

        self.states_buffer  = []
        self.pi_buffer      = []
        self.rewards_buffer = []

        return policy_loss.item(), value_loss.item()

    def _get_score(self, env):
        if hasattr(env, 'winner'):
            if env.winner == 1:                     return 1.0
            if env.winner == 2 or env.winner == -1: return -1.0
            return 0.0
        if hasattr(env, 'agent_position') and env.done:
            return 1.0 if env.agent_position == env.goal_position else 0.0
        return 0.0

    def save(self, path):
        torch.save(self.network.state_dict(), path)

    def load(self, path):
        self.network.load_state_dict(torch.load(path, map_location=self.device))
        self.network.eval()