import numpy as np
import time
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt


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


class ReinforceAgent:
    def __init__(self, state_size, num_actions, gamma=0.99, lr=1e-3, hidden_size=64):
        self.num_actions     = num_actions
        self.gamma           = gamma
        self.device          = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy          = PolicyNetwork(state_size, num_actions, hidden_size).to(self.device)
        self.optimizer       = optim.Adam(self.policy.parameters(), lr=lr)
        self.saved_log_probs = []
        self.rewards         = []

    def _reset_weights(self):
        for layer in self.policy.net:
            if hasattr(layer, 'weight'):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    def select_action(self, state, available_actions):
        # vérifie que l'état ne contient pas de nan
        state_array = np.array(state, dtype=np.float32)
        if np.isnan(state_array).any():
            # état corrompu → action aléatoire
            action = int(np.random.choice(available_actions))
            # log prob fictif pour ne pas casser learn()
            self.saved_log_probs.append(torch.tensor(0.0, requires_grad=True))
            return action

        state_t = torch.FloatTensor(state_array).unsqueeze(0).to(self.device)
        probs   = self.policy(state_t).squeeze(0)

        # si nan → réinitialise et recalcule
        if torch.isnan(probs).any():
            self._reset_weights()
            probs = self.policy(state_t).squeeze(0)

        # si encore nan → action aléatoire
        if torch.isnan(probs).any():
            action = int(np.random.choice(available_actions))
            self.saved_log_probs.append(torch.tensor(0.0, requires_grad=True))
            return action

        mask  = torch.zeros(self.num_actions, device=self.device)
        mask[available_actions] = 1.0
        probs = probs * mask
        probs = probs / (probs.sum() + 1e-8)

        dist   = torch.distributions.Categorical(probs)
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

        G       = 0
        returns = []
        for r in reversed(self.rewards):
            G = r + self.gamma * G
            returns.insert(0, G)

        returns = torch.FloatTensor(returns).to(self.device)

        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)

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


def evaluate(env, agent, n_games=500):
    scores, lengths, times = [], [], []

    for _ in range(n_games):
        env.reset()
        state = env.encode_state()
        steps = 0

        while not env.done:
            available = env.get_actions()
            t0      = time.perf_counter()
            state_t = torch.FloatTensor(state).unsqueeze(0).to(agent.device)
            probs   = agent.policy(state_t).squeeze(0)
            mask    = torch.zeros(agent.num_actions, device=agent.device)
            mask[available] = 1.0
            probs   = probs * mask
            probs   = probs / (probs.sum() + 1e-8)
            if torch.isnan(probs).any():
                action = int(np.random.choice(available))
            else:
                action = int(probs.argmax().item())
            t1      = time.perf_counter()
            times.append((t1 - t0) * 1000)
            _, reward, _ = env.step(action)
            state        = env.encode_state()
            steps       += 1

        scores.append(reward)
        lengths.append(steps)

    return {
        "score_moyen":   round(float(np.mean(scores)),  4),
        "longueur_moy":  round(float(np.mean(lengths)), 2),
        "temps_coup_ms": round(float(np.mean(times)),   4),
    }


def train_and_collect(env, agent, num_episodes, checkpoints, window=100):
    all_losses     = []
    all_rewards    = []
    eval_results   = {}
    checkpoints    = sorted(checkpoints)
    next_check_idx = 0

    for episode in range(1, num_episodes + 1):
        env.reset()
        state          = env.encode_state()
        episode_reward = 0.0

        while not env.done:
            available       = env.get_actions()
            action          = agent.select_action(state, available)
            _, reward, _    = env.step(action)
            agent.store_reward(reward)
            state           = env.encode_state()
            episode_reward += reward

        loss = agent.learn()
        all_losses.append(loss)
        all_rewards.append(episode_reward)

        if next_check_idx < len(checkpoints) and episode == checkpoints[next_check_idx]:
            metrics = evaluate(env, agent, n_games=500)
            eval_results[episode] = metrics
            print(
                f"  {episode:>9,} épisodes | "
                f"Score : {metrics['score_moyen']:.4f} | "
                f"Longueur : {metrics['longueur_moy']:.2f} steps | "
                f"Temps/coup : {metrics['temps_coup_ms']:.4f} ms"
            )
            next_check_idx += 1

    return all_rewards, all_losses, eval_results


def plot_results(all_rewards, all_losses, env_name, agent_name, window=100):
    smoothed = [
        np.mean(all_rewards[max(0, i - window):i + 1])
        for i in range(len(all_rewards))
    ]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    fig.suptitle(f"{agent_name} - {env_name}", fontsize=14)

    ax1.plot(smoothed, color="steelblue")
    ax1.set_ylabel(f"Reward moyen ({window} épisodes)")
    ax1.set_xlabel("Épisode")
    ax1.grid(True, alpha=0.3)

    ax2.plot(all_losses, color="steelblue")
    ax2.set_ylabel("Loss")
    ax2.set_xlabel("Épisode")
    ax2.set_title("Loss par épisode")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    filename = f"{agent_name}_{env_name}.png".replace(" ", "_")
    plt.savefig(filename, dpi=150)
    plt.show()
    print(f"Graphique sauvegardé → {filename}")