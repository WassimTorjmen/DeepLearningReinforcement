import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


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


# NOUVEAU : réseau Critic qui estime V(s)
class ValueNetwork(nn.Module):
    """
    Le Critic — estime la valeur d'un état V(s).

    Contrairement au PolicyNetwork qui sort une proba par action,
    ce réseau sort UNE SEULE valeur : l'estimation du retour futur
    depuis l'état s.

    Entrée  : vecteur d'état
    Sortie  : un scalaire V(s)
    """
    def __init__(self, state_size, hidden_size=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)   # une seule sortie : V(s)
            # pas d'activation : V(s) peut être n'importe quel réel
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)   # (batch,)


class ReinforceAgentCritic:
    """
    REINFORCE with Baseline Learned by a Critic.

    Différences avec REINFORCE classique :
        1. Un deuxième réseau (Critic) apprend V(s)
        2. La baseline est V(s_t) au lieu de la moyenne des retours
        3. On entraîne aussi le Critic à chaque fin d'épisode
    """
    def __init__(self, state_size, num_actions, gamma=0.99, lr=1e-3, hidden_size=64):
        self.num_actions     = num_actions
        self.gamma           = gamma
        self.device          = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Identique au REINFORCE
        self.policy          = PolicyNetwork(state_size, num_actions, hidden_size).to(self.device)
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=lr)

        # NOUVEAU : réseau Critic et son optimiseur
        self.critic          = ValueNetwork(state_size, hidden_size).to(self.device)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)

        self.saved_log_probs = []
        self.rewards         = []
        self.saved_states    = []   # NOUVEAU : on sauvegarde les états pour le Critic

    # Identique au REINFORCE sauf qu'on sauvegarde aussi l'état
    def select_action(self, state, available_actions):
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        probs   = self.policy(state_t).squeeze(0)
        mask    = torch.zeros(self.num_actions)
        mask[available_actions] = 1.0
        probs   = probs * mask
        probs   = probs / (probs.sum() + 1e-8)
        dist    = torch.distributions.Categorical(probs)
        action  = dist.sample()
        self.saved_log_probs.append(dist.log_prob(action))
        self.saved_states.append(state)   # NOUVEAU : on mémorise l'état
        return int(action.item())

    # Identique au REINFORCE
    def store_reward(self, reward):
        self.rewards.append(reward)

    def learn(self):
        if len(self.saved_log_probs) == 0 or len(self.rewards) != len(self.saved_log_probs):
            self.saved_log_probs = []
            self.rewards         = []
            self.saved_states    = []
            return 0.0, 0.0

        # ── Calcul des retours G_t ── identique au REINFORCE ────
        G       = 0
        returns = []
        for r in reversed(self.rewards):
            G = r + self.gamma * G
            returns.insert(0, G)
        returns = torch.FloatTensor(returns).to(self.device)

        # NOUVEAU : calcul de V(s_t) pour chaque état de l'épisode
        states_t = torch.FloatTensor(self.saved_states).to(self.device)
        values   = self.critic(states_t)   # V(s_t) pour chaque step

        # NOUVEAU : advantage = G_t - V(s_t)
        # C'est la différence entre ce qu'on a obtenu et ce qu'on attendait
        # REINFORCE classique  : advantage = (G_t - mean) / std
        # REINFORCE mean baseline : advantage = G_t - mean
        # REINFORCE critic     : advantage = G_t - V(s_t)
        advantages = returns - values.detach()   # detach : on ne propage pas dans le critic ici

        # ── Loss de la politique ─────────────────────────────────
        # identique au REINFORCE mais avec l'advantage à la place des returns
        policy_loss = torch.stack(
            [-lp * adv for lp, adv in zip(self.saved_log_probs, advantages)]
        ).sum()

        # NOUVEAU : loss du Critic = MSE entre V(s_t) et G_t
        # Le Critic apprend à prédire le retour futur réel
        critic_loss = nn.MSELoss()(values, returns)

        # ── Mise à jour de la politique ──────────────────────────
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=1.0)
        self.policy_optimizer.step()

        # NOUVEAU : mise à jour du Critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)
        self.critic_optimizer.step()

        self.saved_log_probs = []
        self.rewards         = []
        self.saved_states    = []   # NOUVEAU : vide les états

        return policy_loss.item(), critic_loss.item()

    def save(self, path):
        torch.save({
            "policy": self.policy.state_dict(),
            "critic": self.critic.state_dict(),   # NOUVEAU : sauvegarde aussi le critic
        }, path)

    def load(self, path):
        data = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(data["policy"])
        self.critic.load_state_dict(data["critic"])
        self.policy.eval()
        self.critic.eval()