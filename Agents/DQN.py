"""
DQN (Deep Q-Network) : approximation de la Q-fonction par un MLP.
Apprentissage en ligne par minimisation de l'erreur de Bellman, politique ε-greedy.
Version simple sans replay buffer ni target network (voir ddqn.py pour le target).
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class QNetwork(nn.Module):
    """
    Réseau de neurones qui approxime Q(s, a).
    Entrée  : vecteur d'état
    Sortie  : une Q-valeur par action
    """
    def __init__(self, state_size, num_actions, hidden_size=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_actions)  # pas d'activation : Q peut être négatif
        )

    def forward(self, x):
        return self.net(x)


class DQNAgent:
    """
    Agent Deep Q-Network.

    À chaque step il :
      1. choisit une action avec la politique ε-greedy
      2. apprend en minimisant l'erreur de Bellman :
         Q(s,a) doit tendre vers r + γ * max Q(s', a')
    """
    def __init__(self, state_size, num_actions, gamma=0.99, lr=1e-3,
                 epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995,
                 hidden_size=64):

        self.num_actions   = num_actions
        self.gamma         = gamma
        self.epsilon       = epsilon_start
        self.epsilon_end   = epsilon_end
        self.epsilon_decay = epsilon_decay

        self.device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.q_network = QNetwork(state_size, num_actions, hidden_size).to(self.device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        self.loss_fn   = nn.MSELoss()

    def select_action(self, state, available_actions):
        """
        Politique ε-greedy :
        - avec proba ε   → action aléatoire  (exploration)
        - avec proba 1-ε → action avec la plus haute Q-valeur  (exploitation)
        """
        if np.random.rand() < self.epsilon:
            return int(np.random.choice(available_actions))

        state_t  = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        q_values = self.q_network(state_t).squeeze(0)

        # masque les actions illégales avec -inf pour ne jamais les choisir
        mask = torch.full((self.num_actions,), float('-inf'))
        mask[available_actions] = 0.0
        q_values = q_values + mask

        return int(q_values.argmax().item())

    def learn(self, state, action, reward, next_state, done, next_available_actions):
        """
        Met à jour le réseau sur une transition (s, a, r, s', done).

        Cible de Bellman :
            y = r                            si done
            y = r + γ * max_{a'} Q(s', a')  sinon
        On minimise (Q(s,a) - y)²
        """
        s  = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        s_ = torch.FloatTensor(next_state).unsqueeze(0).to(self.device)

        # prédiction
        q_sa = self.q_network(s)[0, action]

        # cible
        with torch.no_grad():
            q_next = self.q_network(s_).squeeze(0)
            mask   = torch.full((self.num_actions,), float('-inf'))
            mask[next_available_actions] = 0.0
            q_next = q_next + mask

        if done:
            target = torch.tensor(reward, dtype=torch.float32).to(self.device)
        else:
            target = torch.tensor(reward + self.gamma * q_next.max().item(),
                                  dtype=torch.float32).to(self.device)

        # descente de gradient
        loss = self.loss_fn(q_sa, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def decay_epsilon(self):
        """Réduit ε après chaque épisode pour passer de l'exploration à l'exploitation."""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    def save(self, path):
        torch.save(self.q_network.state_dict(), path)

    def load(self, path):
        self.q_network.load_state_dict(torch.load(path, map_location=self.device))
        self.q_network.eval()


def evaluate(env, agent, n_games=100):
    """Évalue l'agent sans exploration (ε=0) — c'est ce score qui va dans le rapport."""
    saved_eps   = agent.epsilon
    agent.epsilon = 0.0
    scores = []

    for _ in range(n_games):
        state = env.reset()
        while not env.is_game_over():
            action = agent.select_action(state, env.available_actions())
            state, _, _ = env.step(action)
        scores.append(env.score())

    agent.epsilon = saved_eps
    return np.mean(scores)


def train(env, agent, num_episodes, eval_every=1000):
    """Boucle d'entraînement principale."""
    for episode in range(1, num_episodes + 1):
        state = env.reset()

        while not env.is_game_over():
            available  = env.available_actions()
            action     = agent.select_action(state, available)
            next_state, reward, done = env.step(action)
            next_available = env.available_actions() if not done else [0]

            agent.learn(state, action, reward, next_state, done, next_available)
            state = next_state

        agent.decay_epsilon()

        if episode % eval_every == 0:
            score = evaluate(env, agent)
            print(f"Episode {episode} | Score eval : {score:.3f} | Epsilon : {agent.epsilon:.3f}")
