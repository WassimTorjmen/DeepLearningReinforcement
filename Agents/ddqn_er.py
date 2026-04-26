import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random


# Identique au DQN/DDQN
class QNetwork(nn.Module):
    def __init__(self, state_size, num_actions, hidden_size=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_actions)
        )

    def forward(self, x):
        return self.net(x)


# ╔═════════════════════════════════════════════════════════════╗
# ║  EXPERIENCE REPLAY BUFFER                                  ║
# ╚═════════════════════════════════════════════════════════════╝

class ReplayBuffer:
    """
    Mémoire de l'agent — stocke les transitions (s, a, r, s', done).

    DIFFÉRENCE avec DQN/DDQN classique :
        DQN/DDQN : apprend sur la transition JUSTE vécue (très corrélé)
        DQN+ER   : stocke les transitions dans un buffer et tire
                   un BATCH aléatoire pour apprendre
                   → casse les corrélations temporelles
                   → apprentissage plus stable

    Paramètres :
        capacity  : taille max du buffer (les plus vieilles transitions sont écrasées)
        batch_size : nombre de transitions tirées à chaque apprentissage
    """
    def __init__(self, capacity=10_000, batch_size=64):
        self.buffer     = deque(maxlen=capacity)  # file circulaire
        self.batch_size = batch_size

    def push(self, state, action, reward, next_state, done):
        """Ajoute une transition dans le buffer."""
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self):
        """Tire un batch aléatoire de transitions."""
        return random.sample(self.buffer, self.batch_size)

    def __len__(self):
        return len(self.buffer)

    def is_ready(self):
        """On n'apprend que quand le buffer est assez rempli."""
        return len(self.buffer) >= self.batch_size


class DoubleDQNWithERAgent:
    """
    Double DQN + Experience Replay.

    Différences avec DQN classique :
    1. DOUBLE DQN : deux réseaux (principal + cible) pour éviter
       la surestimation des Q-valeurs
    2. EXPERIENCE REPLAY : on stocke les transitions et on apprend
       sur un batch aléatoire au lieu de la dernière transition

    Différence avec DDQN sans ER :
        DDQN     : apprend sur 1 transition à chaque step
        DDQN+ER  : stocke toutes les transitions, tire un batch
                   de batch_size transitions aléatoires pour apprendre

    Paramètres :
        buffer_capacity    : taille du replay buffer
        batch_size         : nb de transitions par batch d'apprentissage
        target_update_every: fréquence de mise à jour du réseau cible
    """
    def __init__(self, state_size, num_actions, gamma=0.99, lr=1e-3,
                 epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995,
                 hidden_size=64, target_update_every=100,
                 buffer_capacity=10_000, batch_size=64):  # NOUVEAU : buffer params

        self.num_actions         = num_actions
        self.gamma               = gamma
        self.epsilon             = epsilon_start
        self.epsilon_end         = epsilon_end
        self.epsilon_decay       = epsilon_decay
        self.target_update_every = target_update_every
        self.steps_done          = 0

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Identique au DDQN : deux réseaux
        self.q_network      = QNetwork(state_size, num_actions, hidden_size).to(self.device)
        self.target_network = QNetwork(state_size, num_actions, hidden_size).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        self.loss_fn   = nn.MSELoss()

        # NOUVEAU : le replay buffer
        self.replay_buffer = ReplayBuffer(buffer_capacity, batch_size)
        self.batch_size    = batch_size

    # Identique au DQN/DDQN
    def select_action(self, state, available_actions):
        if np.random.rand() < self.epsilon:
            return int(np.random.choice(available_actions))
        state_t  = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        q_values = self.q_network(state_t).squeeze(0)
        mask     = torch.full((self.num_actions,), float('-inf'), device=self.device)
        mask[available_actions] = 0.0
        return int((q_values + mask).argmax().item())

    def store(self, state, action, reward, next_state, done):
        """
        NOUVEAU : stocke la transition dans le buffer.
        DQN/DDQN apprennent directement ici.
        DDQN+ER stocke d'abord et apprend plus tard sur un batch.
        """
        self.replay_buffer.push(state, action, reward, next_state, done)

    def learn(self, next_available_actions=None):
        """
        NOUVEAU : apprend sur un batch aléatoire du buffer.

        Différence clé avec DQN/DDQN :
            DQN/DDQN : learn(state, action, reward, next_state, done, next_available)
                       → apprend sur 1 seule transition
            DDQN+ER  : learn()
                       → tire batch_size transitions aléatoires et apprend dessus
        """
        if not self.replay_buffer.is_ready():
            return 0.0  # on attend que le buffer soit assez rempli

        # tire un batch aléatoire
        batch = self.replay_buffer.sample()
        states, actions, rewards, next_states, dones = zip(*batch)

        # conversion en tenseurs
        states_t      = torch.FloatTensor(np.array(states)).to(self.device)
        actions_t     = torch.LongTensor(actions).to(self.device)
        rewards_t     = torch.FloatTensor(rewards).to(self.device)
        next_states_t = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones_t       = torch.FloatTensor(dones).to(self.device)

        # Q(s, a) pour chaque transition du batch
        q_values = self.q_network(states_t).gather(1, actions_t.unsqueeze(1)).squeeze(1)

        # Double DQN : principal choisit l'action, cible l'évalue
        with torch.no_grad():
            next_actions  = self.q_network(next_states_t).argmax(dim=1)
            next_q_values = self.target_network(next_states_t).gather(
                1, next_actions.unsqueeze(1)
            ).squeeze(1)
            targets = rewards_t + self.gamma * next_q_values * (1 - dones_t)

        loss = self.loss_fn(q_values, targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # mise à jour périodique du réseau cible
        self.steps_done += 1
        if self.steps_done % self.target_update_every == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

        return loss.item()

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    def save(self, path):
        torch.save(self.q_network.state_dict(), path)

    def load(self, path):
        self.q_network.load_state_dict(torch.load(path, map_location=self.device))
        self.q_network.eval()