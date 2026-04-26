import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


# Identique au REINFORCE Critic
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


# Identique au REINFORCE Critic
class ValueNetwork(nn.Module):
    def __init__(self, state_size, hidden_size=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


class PPOAgent:
    """
    PPO A2C style.

    Différences avec REINFORCE Critic :
    1. On sauvegarde les ANCIENNES probabilités avant la mise à jour
    2. On calcule un ratio  new_prob / old_prob
    3. On coupe ce ratio entre [1-clip_eps, 1+clip_eps]  ← c'est le PPO clip
    4. On apprend à chaque step (A2C) et non à la fin de l'épisode

    Paramètres nouveaux :
        clip_eps : marge autorisée pour le ratio (typiquement 0.2)
    """
    def __init__(self, state_size, num_actions, gamma=0.99, lr=1e-3,
                 hidden_size=64, clip_eps=0.2):  # NOUVEAU : clip_eps

        self.num_actions = num_actions
        self.gamma       = gamma
        self.clip_eps    = clip_eps  # NOUVEAU

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Identique au REINFORCE Critic
        self.policy           = PolicyNetwork(state_size, num_actions, hidden_size).to(self.device)
        self.critic           = ValueNetwork(state_size, hidden_size).to(self.device)
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)

        # mémoire de l'épisode
        self.saved_log_probs = []
        self.saved_old_probs = []  # NOUVEAU : on sauvegarde les anciennes probas
        self.rewards         = []
        self.saved_states    = []

    def select_action(self, state, available_actions):
        """
        Identique au REINFORCE Critic SAUF qu'on sauvegarde aussi
        la probabilité de l'action choisie AVANT la mise à jour (old_prob).
        """
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        probs   = self.policy(state_t).squeeze(0)

        mask = torch.zeros(self.num_actions, device=self.device)
        #Old version
        #mask    = torch.zeros(self.num_actions)
        mask[available_actions] = 1.0
        probs   = probs * mask
        probs   = probs / (probs.sum() + 1e-8)

        dist    = torch.distributions.Categorical(probs)
        action  = dist.sample()

        self.saved_log_probs.append(dist.log_prob(action))
        # NOUVEAU : on sauvegarde la proba de l'action choisie avant mise à jour
        self.saved_old_probs.append(probs[action].detach())
        self.saved_states.append(state)

        return int(action.item())

    def store_reward(self, reward):
        self.rewards.append(reward)

    def learn(self):
        if len(self.saved_log_probs) == 0 or len(self.rewards) != len(self.saved_log_probs):
            self.saved_log_probs = []
            self.saved_old_probs = []
            self.rewards         = []
            self.saved_states    = []
            return 0.0, 0.0

        # ── Calcul des retours G_t ── identique au REINFORCE Critic
        G       = 0
        returns = []
        for r in reversed(self.rewards):
            G = r + self.gamma * G
            returns.insert(0, G)
        returns = torch.FloatTensor(returns).to(self.device)

        # ── Calcul des valeurs et advantages ── identique au REINFORCE Critic
        #states_np = np.asarray(self.saved_states, dtype=np.float32)
        #states_t = torch.as_tensor(states_np, dtype=torch.float32, device=self.device)
        states_t = torch.as_tensor(
        np.asarray(self.saved_states, dtype=np.float32),
        device=self.device
        )





        #states_t   = torch.FloatTensor(self.saved_states).to(self.device)
        values     = self.critic(states_t)
        advantages = returns - values.detach()

        # ── NOUVEAU : calcul du ratio new_prob / old_prob ────────
        # new_probs : probabilités APRÈS la mise à jour (calculées maintenant)
        new_probs  = torch.exp(torch.stack(self.saved_log_probs))
        old_probs  = torch.stack(self.saved_old_probs)
        ratio      = new_probs / (old_probs + 1e-8)

        # ── NOUVEAU : PPO clip ────────────────────────────────────
        # on calcule deux termes et on prend le minimum (le plus pessimiste)
        # terme 1 : ratio * advantage  (mise à jour normale)
        # terme 2 : ratio clipé * advantage  (mise à jour limitée)
        clipped_ratio = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps)
        policy_loss   = -torch.min(
            ratio * advantages,
            clipped_ratio * advantages
        ).sum()

        # ── Loss Critic ── identique au REINFORCE Critic
        critic_loss = nn.MSELoss()(values, returns)

        # ── Mise à jour ── identique au REINFORCE Critic
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=1.0)
        self.policy_optimizer.step()

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)
        self.critic_optimizer.step()

        self.saved_log_probs = []
        self.saved_old_probs = []
        self.rewards         = []
        self.saved_states    = []

        return policy_loss.item(), critic_loss.item()

    def save(self, path):
        torch.save({
            "policy": self.policy.state_dict(),
            "critic": self.critic.state_dict(),
        }, path)

    def load(self, path):
        data = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(data["policy"])
        self.critic.load_state_dict(data["critic"])
        self.policy.eval()
        self.critic.eval()