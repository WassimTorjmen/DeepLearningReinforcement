import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
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
# ║  PRIORITIZED EXPERIENCE REPLAY BUFFER                      ║
# ╚═════════════════════════════════════════════════════════════╝

class PrioritizedReplayBuffer:
    """
    Buffer avec priorités — les transitions les plus surprenantes
    sont tirées plus souvent.

    DIFFÉRENCE avec Experience Replay classique :
        ER   : toutes les transitions ont la même proba d'être tirées
        PER  : les transitions avec une grande erreur TD (surprise)
               ont une plus haute priorité → tirées plus souvent

    L'erreur TD = |Q(s,a) - cible|
    Plus l'erreur est grande = l'agent est plus surpris = plus important à revoir.

    Paramètres :
        alpha : contrôle l'importance des priorités
                0 = tirage uniforme (comme ER classique)
                1 = tirage entièrement basé sur la priorité
        beta  : correction du biais introduit par le tirage non-uniforme
    """
    def __init__(self, capacity=10_000, batch_size=64, alpha=0.6, beta=0.4):
        self.capacity   = capacity
        self.batch_size = batch_size
        self.alpha      = alpha   # importance des priorités
        self.beta       = beta    # correction du biais

        self.buffer     = []
        self.priorities = []      # NOUVEAU : priorité de chaque transition
        self.pos        = 0       # position d'écriture circulaire

    def push(self, state, action, reward, next_state, done):
        """
        Ajoute une transition avec la priorité maximale actuelle.
        Une nouvelle transition reçoit la priorité max pour être
        vue au moins une fois.
        """
        max_priority = max(self.priorities) if self.priorities else 1.0

        if len(self.buffer) < self.capacity:
            self.buffer.append((state, action, reward, next_state, done))
            self.priorities.append(max_priority)
        else:
            self.buffer[self.pos]     = (state, action, reward, next_state, done)
            self.priorities[self.pos] = max_priority
            self.pos = (self.pos + 1) % self.capacity

    def sample(self):
        """
        NOUVEAU : tire un batch selon les priorités.

        Probabilité de tirer la transition i :
            P(i) = priorité(i)^alpha / somme(priorité^alpha)
        """
        priorities = np.array(self.priorities, dtype=np.float32)
        probs      = priorities ** self.alpha
        probs     /= probs.sum()  # normalise en probabilités

        # tire les indices selon les probabilités
        indices = np.random.choice(len(self.buffer), self.batch_size,
                                   p=probs, replace=False)
        samples = [self.buffer[i] for i in indices]

        # poids d'importance pour corriger le biais du tirage non-uniforme
        weights = (len(self.buffer) * probs[indices]) ** (-self.beta)
        weights /= weights.max()  # normalise

        return samples, indices, torch.FloatTensor(weights)

    def update_priorities(self, indices, td_errors):
        """
        NOUVEAU : met à jour les priorités après l'apprentissage.
        priorité(i) = |erreur TD(i)| + epsilon (pour éviter priorité=0)
        """
        for idx, error in zip(indices, td_errors):
            self.priorities[idx] = abs(error) + 1e-6

    def is_ready(self):
        return len(self.buffer) >= self.batch_size

    def __len__(self):
        return len(self.buffer)


class DoubleDQNWithPERAgent:
    """
    Double DQN + Prioritized Experience Replay.

    Différences par rapport aux algos précédents :

    vs DQN :
        + réseau cible (Double DQN)
        + replay buffer avec priorités (PER)

    vs DDQN :
        + replay buffer avec priorités (PER)

    vs DDQN+ER :
        DDQN+ER : toutes les transitions ont la même proba d'être tirées
        DDQN+PER: les transitions surprenantes sont tirées plus souvent
                  → apprentissage plus efficace sur les cas difficiles

    Paramètres supplémentaires :
        alpha : importance des priorités (0=uniforme, 1=full priorité)
        beta  : correction du biais (augmente de beta_start vers 1.0)
    """
    def __init__(self, state_size, num_actions, gamma=0.99, lr=1e-3,
                 epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995,
                 hidden_size=64, target_update_every=100,
                 buffer_capacity=10_000, batch_size=64,
                 alpha=0.6, beta=0.4):  # NOUVEAU : params PER

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
        self.loss_fn   = nn.MSELoss(reduction='none')  # NOUVEAU : loss par transition

        # NOUVEAU : Prioritized Replay Buffer au lieu du buffer uniforme
        self.replay_buffer = PrioritizedReplayBuffer(
            buffer_capacity, batch_size, alpha, beta
        )
        self.batch_size = batch_size

    # Identique au DQN/DDQN
    def select_action(self, state, available_actions):
        if np.random.rand() < self.epsilon:
            return int(np.random.choice(available_actions))
        state_t  = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        q_values = self.q_network(state_t).squeeze(0)
        mask     = torch.full((self.num_actions,), float('-inf'))
        mask[available_actions] = 0.0
        return int((q_values + mask).argmax().item())

    def store(self, state, action, reward, next_state, done):
        """Identique à DDQN+ER : stocke la transition dans le buffer."""
        self.replay_buffer.push(state, action, reward, next_state, done)

    def learn(self, next_available_actions=None):
        """
        Apprend sur un batch tiré selon les priorités.

        DIFFÉRENCE avec DDQN+ER :
            ER  : batch = random.sample(buffer) → proba uniforme
            PER : batch = sample selon priorités → transitions surprenantes
                  plus souvent + mise à jour des priorités après apprentissage
        """
        if not self.replay_buffer.is_ready():
            return 0.0

        # NOUVEAU : tire selon les priorités (pas uniformément)
        batch, indices, weights = self.replay_buffer.sample()
        states, actions, rewards, next_states, dones = zip(*batch)

        states_t      = torch.FloatTensor(np.array(states)).to(self.device)
        actions_t     = torch.LongTensor(actions).to(self.device)
        rewards_t     = torch.FloatTensor(rewards).to(self.device)
        next_states_t = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones_t       = torch.FloatTensor(dones).to(self.device)
        weights       = weights.to(self.device)

        q_values = self.q_network(states_t).gather(1, actions_t.unsqueeze(1)).squeeze(1)

        # Double DQN : principal choisit, cible évalue
        with torch.no_grad():
            next_actions  = self.q_network(next_states_t).argmax(dim=1)
            next_q_values = self.target_network(next_states_t).gather(
                1, next_actions.unsqueeze(1)
            ).squeeze(1)
            targets = rewards_t + self.gamma * next_q_values * (1 - dones_t)

        # NOUVEAU : loss pondérée par les poids d'importance (correction biais PER)
        td_errors = (q_values - targets).detach().cpu().numpy()
        losses    = self.loss_fn(q_values, targets)
        loss      = (weights * losses).mean()  # pondération par importance

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # NOUVEAU : met à jour les priorités avec les nouvelles erreurs TD
        self.replay_buffer.update_priorities(indices, td_errors)

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