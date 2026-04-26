import numpy as np
import time
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt


# Identique au REINFORCE
class PolicyNetwork(nn.Module):
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


class ReinforceAgentMeanBaseline:
    """
    Agent REINFORCE with Mean Baseline.

    Différence avec REINFORCE classique :
        REINFORCE classique     : returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        REINFORCE mean baseline : returns = returns - returns.mean()
    """
    def __init__(self, state_size, num_actions, gamma=0.99, lr=1e-3, hidden_size=64):
        self.num_actions     = num_actions
        self.gamma           = gamma
        self.device          = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy          = PolicyNetwork(state_size, num_actions, hidden_size).to(self.device)
        # CORRIGÉ : lr plus petit pour éviter explosion des gradients sur envs complexes
        self.optimizer       = optim.Adam(self.policy.parameters(), lr=min(lr, 1e-4))
        self.saved_log_probs = []
        self.rewards         = []

    # Identique au REINFORCE
    def select_action(self, state, available_actions):
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        probs   = self.policy(state_t).squeeze(0)

        # CORRIGÉ : vérifie que le réseau ne produit pas de nan
        if torch.isnan(probs).any():
            self.policy.apply(self._reset_weights)
            probs = self.policy(state_t).squeeze(0)

        mask    = torch.zeros(self.num_actions)
        mask[available_actions] = 1.0
        probs   = probs * mask
        probs   = probs / (probs.sum() + 1e-8)
        dist    = torch.distributions.Categorical(probs)
        action  = dist.sample()
        self.saved_log_probs.append(dist.log_prob(action))
        return int(action.item())

    @staticmethod
    def _reset_weights(m):
        """Réinitialise les poids si nan détecté."""
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)

    # Identique au REINFORCE
    def store_reward(self, reward):
        self.rewards.append(reward)

    def learn(self):
        # sécurité : réinitialise si état incohérent
        if len(self.saved_log_probs) == 0 or len(self.rewards) != len(self.saved_log_probs):
            self.saved_log_probs = []
            self.rewards         = []
            return 0.0

        G       = 0
        returns = []
        for r in reversed(self.rewards):
            G = r + self.gamma * G
            returns.insert(0, G)

        returns = torch.FloatTensor(returns).to(self.device)

        # DIFFÉRENCE : soustrait uniquement la moyenne (pas de division par std)
        if len(returns) > 1:
            returns = returns - returns.mean()

        loss = torch.stack(
            [-lp * G_t for lp, G_t in zip(self.saved_log_probs, returns)]
        ).sum()

        # CORRIGÉ : vérifie que la loss n'est pas nan avant de rétropropager
        if torch.isnan(loss):
            self.saved_log_probs = []
            self.rewards         = []
            return 0.0

        self.optimizer.zero_grad()
        loss.backward()
        # gradient clipping pour éviter l'explosion des gradients
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=1.0)
        self.optimizer.step()

        self.saved_log_probs = []
        self.rewards         = []

        return loss.item()

    # Identique au REINFORCE
    def save(self, path):
        torch.save(self.policy.state_dict(), path)

    def load(self, path):
        self.policy.load_state_dict(torch.load(path, map_location=self.device))
        self.policy.eval()