"""
Entraîne un TabularQAgent sur GridWorld 5x5 par Q-learning ε-greedy.
Identique à train_tabular_lineworld mais sur la version 2D.
"""

import os
import sys

# Permet d'importer les modules du projet (racine ajoutée au PYTHONPATH)
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from Environnements.grid_world import GridWorld
from Agents.tabular_q_agent import TabularQAgent


# =========================================================
# Helper : gérer step() (3 ou 4 valeurs)
# =========================================================
def unpack_step(step_result):
    if len(step_result) == 3:
        next_state, reward, done = step_result
        return next_state, reward, done, {}

    elif len(step_result) == 4:
        return step_result

    else:
        raise ValueError("Format invalide pour step()")


# =========================================================
# Fonction d'entraînement GridWorld
# =========================================================
def train_gridworld(n_episodes=2000):
    env = GridWorld()

    agent = TabularQAgent(
        alpha=0.1,
        gamma=0.99,
        epsilon=1.0,
        epsilon_decay=0.995,
        epsilon_min=0.05,
    )

    episode_rewards = []
    episode_lengths = []

    # Boucle épisodes
    for episode in range(n_episodes):
        state = env.reset()
        done = False
        total_reward = 0
        steps = 0

        # Boucle épisode
        while not done:
            valid_actions = env.get_actions()

            # Action
            action = agent.choose_action(state, valid_actions)

            # step()
            step_result = env.step(action)
            next_state, reward, done, info = unpack_step(step_result)

            next_valid_actions = [] if done else env.get_actions()

            # apprentissage
            agent.learn(
                state=state,
                action=action,
                reward=reward,
                next_state=next_state,
                done=done,
                next_valid_actions=next_valid_actions,
            )

            state = next_state
            total_reward += reward
            steps += 1

        agent.decay_epsilon()

        episode_rewards.append(total_reward)
        episode_lengths.append(steps)

        # Logs (plus espacés car plus complexe)
        if (episode + 1) % 1000 == 0:
            avg_reward = sum(episode_rewards[-1000:]) / 1000
            avg_steps = sum(episode_lengths[-1000:]) / 1000

            print(
                f"Episode {episode + 1}/{n_episodes} | "
                f"avg_reward={avg_reward:.3f} | "
                f"avg_steps={avg_steps:.2f} | "
                f"epsilon={agent.epsilon:.3f}"
            )

    return agent, episode_rewards, episode_lengths


# =========================================================
# Lancement
# =========================================================
if __name__ == "__main__":
    print(" Training GridWorld...")

    train_gridworld()