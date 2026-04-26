import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


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
    REINFORCE with Mean Baseline.
    Différence : returns = returns - returns.mean()  (pas de division par std)
    """
    def __init__(self, state_size, num_actions, gamma=0.99, lr=1e-3, hidden_size=64):
        self.num_actions     = num_actions
        self.gamma           = gamma
        self.device          = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy          = PolicyNetwork(state_size, num_actions, hidden_size).to(self.device)
        self.optimizer       = optim.Adam(self.policy.parameters(), lr=min(lr, 1e-4))
        self.saved_log_probs = []
        self.rewards         = []

    def _reset_weights(self):
        for layer in self.policy.net:
            if hasattr(layer, 'weight'):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    def select_action(self, state, available_actions):
        state_array = np.array(state, dtype=np.float32)

        # état corrompu → action aléatoire
        if np.isnan(state_array).any():
            self.saved_log_probs.append(torch.tensor(0.0, requires_grad=True))
            return int(np.random.choice(available_actions))

        state_t = torch.FloatTensor(state_array).unsqueeze(0).to(self.device)
        probs   = self.policy(state_t).squeeze(0)

        # nan détecté → reset et recalcule
        if torch.isnan(probs).any():
            self._reset_weights()
            probs = self.policy(state_t).squeeze(0)

        mask  = torch.zeros(self.num_actions, device=self.device)
        mask[available_actions] = 1.0
        probs = probs * mask
        total = probs.sum()

        # dernier check avant Categorical — si encore nan ou sum=0 → aléatoire
        if torch.isnan(total) or total.item() < 1e-8:
            self.saved_log_probs.append(torch.tensor(0.0, requires_grad=True))
            return int(np.random.choice(available_actions))

        probs = probs / total
        dist  = torch.distributions.Categorical(probs)
        action = dist.sample()
        self.saved_log_probs.append(dist.log_prob(action))
        return int(action.item())

    def store_reward(self, reward):
        self.rewards.append(reward)

    def learn(self):
        if len(self.saved_log_probs) == 0 or len(self.rewards) != len(self.saved_log_probs):
            self.saved_log_probs = []
            self.rewards         = []
            return 0.0

        G, returns = 0, []
        for r in reversed(self.rewards):
            G = r + self.gamma * G
            returns.insert(0, G)

        returns = torch.FloatTensor(returns).to(self.device)
        if len(returns) > 1:
            returns = returns - returns.mean()

        loss = torch.stack(
            [-lp * G_t for lp, G_t in zip(self.saved_log_probs, returns)]
        ).sum()

        if torch.isnan(loss):
            self.saved_log_probs = []
            self.rewards         = []
            return 0.0

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=1.0)
        self.optimizer.step()

        self.saved_log_probs = []
        self.rewards         = []
        return loss.item()

    def save(self, path):
        torch.save(self.policy.state_dict(), path)

    def load(self, path):
        self.policy.load_state_dict(torch.load(path, map_location=self.device))
        self.policy.eval()