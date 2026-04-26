import numpy as np
import math
import torch
import torch.nn as nn
import torch.optim as optim


class RepresentationNetwork(nn.Module):
    """Encode l'état réel en état latent."""
    def __init__(self, state_size, hidden_size=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_size, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, hidden_size), nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)


class DynamicsNetwork(nn.Module):
    """Prédit l'état latent suivant et la récompense sans l'env."""
    def __init__(self, hidden_size, num_actions):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_size + num_actions, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, hidden_size), nn.ReLU(),
        )
        self.next_state_head = nn.Linear(hidden_size, hidden_size)
        self.reward_head     = nn.Linear(hidden_size, 1)

    def forward(self, h, a_onehot):
        x = self.net(torch.cat([h, a_onehot], dim=-1))
        return self.next_state_head(x), self.reward_head(x).squeeze(-1)


class PredictionNetwork(nn.Module):
    """Prédit politique + valeur depuis l'état latent."""
    def __init__(self, hidden_size, num_actions):
        super().__init__()
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_size, num_actions), nn.Softmax(dim=-1)
        )
        self.value_head = nn.Sequential(
            nn.Linear(hidden_size, 1), nn.Tanh()
        )

    def forward(self, h):
        return self.policy_head(h), self.value_head(h).squeeze(-1)


class MuZeroAgent:
    """
    MuZero — version simplifiée sans arbre récursif.

    Au lieu d'un arbre MCTS profond, on fait une recherche à UN niveau :
    pour chaque action possible, on simule n_simulations fois en utilisant
    DynamicsNetwork et on garde l'action avec la meilleure valeur estimée.

    Différences avec AlphaZero :
    1. 3 réseaux : representation, dynamics, prediction
    2. Pas de deepcopy(env) — DynamicsNetwork simule le futur
    3. reward loss en plus de policy loss et value loss
    """

    def __init__(self, state_size, num_actions, hidden_size=64,
                 n_simulations=10, lr=1e-3):
        self.num_actions   = num_actions
        self.hidden_size   = hidden_size
        self.n_simulations = n_simulations
        self.device        = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.representation = RepresentationNetwork(state_size, hidden_size).to(self.device)
        self.dynamics       = DynamicsNetwork(hidden_size, num_actions).to(self.device)
        self.prediction     = PredictionNetwork(hidden_size, num_actions).to(self.device)

        self.optimizer = optim.Adam(
            list(self.representation.parameters()) +
            list(self.dynamics.parameters()) +
            list(self.prediction.parameters()),
            lr=lr
        )

        self.states_buffer  = []
        self.actions_buffer = []
        self.rewards_buffer = []
        self.pi_buffer      = []

    def _encode_action(self, action):
        a = torch.zeros(1, self.num_actions).to(self.device)
        a[0, action] = 1.0
        return a

    def _plan(self, h, available_actions):
        """
        Planification à un niveau :
        Pour chaque action disponible, on utilise DynamicsNetwork
        pour prédire l'état suivant et on évalue avec PredictionNetwork.
        On retourne la distribution sur les actions (comme π_mcts).
        """
        scores = np.zeros(self.num_actions)

        with torch.no_grad():
            for action in available_actions:
                a_onehot        = self._encode_action(action)
                next_h, reward  = self.dynamics(h, a_onehot)
                _, value        = self.prediction(next_h)
                scores[action]  = reward.item() + value.item()

        # on garde seulement les actions disponibles
        pi = np.zeros(self.num_actions)
        for a in available_actions:
            pi[a] = max(scores[a], 0) + 1e-8
        pi = pi / pi.sum()

        return pi

    def select_action(self, state_or_env, available_actions):
        """Évaluation — le réseau joue sans planification."""
        if hasattr(state_or_env, 'encode_state'):
            state = state_or_env.encode_state()
        else:
            state = state_or_env

        state_t = torch.FloatTensor(np.array(state)).unsqueeze(0).to(self.device)
        with torch.no_grad():
            h         = self.representation(state_t)
            policy, _ = self.prediction(h)
        policy = policy.squeeze(0)
        mask   = torch.zeros(self.num_actions).to(self.device)
        mask[available_actions] = 1.0
        policy = policy * mask
        policy = policy / (policy.sum() + 1e-8)
        return int(policy.argmax().item())

    def select_action_mcts(self, env, available_actions):
        """Entraînement — planification avec DynamicsNetwork."""
        state   = env.encode_state()
        state_t = torch.FloatTensor(np.array(state)).unsqueeze(0).to(self.device)

        with torch.no_grad():
            h = self.representation(state_t)

        pi = self._plan(h, available_actions)

        self.states_buffer.append(state)
        self.pi_buffer.append(pi)

        action = int(np.random.choice(self.num_actions, p=pi))
        self.actions_buffer.append(action)
        return action

    def store_reward(self, reward):
        n = len(self.states_buffer) - len(self.rewards_buffer)
        self.rewards_buffer.extend([reward] * n)

    def learn(self):
        if not self.states_buffer or len(self.states_buffer) != len(self.rewards_buffer):
            self.states_buffer  = []
            self.actions_buffer = []
            self.rewards_buffer = []
            self.pi_buffer      = []
            return 0.0, 0.0

        states  = torch.FloatTensor(np.array(self.states_buffer)).to(self.device)
        pi_mcts = torch.FloatTensor(np.array(self.pi_buffer)).to(self.device)
        rewards = torch.FloatTensor(np.array(self.rewards_buffer)).to(self.device)
        actions = torch.LongTensor(self.actions_buffer).to(self.device)

        h                       = self.representation(states)
        policy_pred, value_pred = self.prediction(h)

        # policy loss : imiter la planification
        policy_loss = -(pi_mcts * torch.log(policy_pred + 1e-8)).sum(dim=1).mean()
        # value loss : prédire le résultat final
        value_loss  = nn.MSELoss()(value_pred, rewards)
        # reward loss : DynamicsNetwork prédit les récompenses
        a_onehot = torch.zeros(len(actions), self.num_actions).to(self.device)
        a_onehot.scatter_(1, actions.unsqueeze(1), 1.0)
        _, reward_pred = self.dynamics(h.detach(), a_onehot)
        reward_loss    = nn.MSELoss()(reward_pred, rewards)

        loss = policy_loss + value_loss + reward_loss

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.representation.parameters()) +
            list(self.dynamics.parameters()) +
            list(self.prediction.parameters()),
            max_norm=1.0
        )
        self.optimizer.step()

        self.states_buffer  = []
        self.actions_buffer = []
        self.rewards_buffer = []
        self.pi_buffer      = []

        return policy_loss.item(), value_loss.item()

    def save(self, path):
        torch.save({
            "representation": self.representation.state_dict(),
            "dynamics":       self.dynamics.state_dict(),
            "prediction":     self.prediction.state_dict(),
        }, path)

    def load(self, path):
        data = torch.load(path, map_location=self.device)
        self.representation.load_state_dict(data["representation"])
        self.dynamics.load_state_dict(data["dynamics"])
        self.prediction.load_state_dict(data["prediction"])