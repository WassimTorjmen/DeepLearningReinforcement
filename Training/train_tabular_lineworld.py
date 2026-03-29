import os
import sys

# Permet d'importer les modules du projet
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from Environnements.line_world import LineWorld
from Agents.tabular_q_agent import TabularQAgent


# =========================================================
# Helper : gérer step() qui peut retourner 3 ou 4 valeurs
# =========================================================
def unpack_step(step_result):
    """
    Accepte :
        - (state, reward, done)
        - (state, reward, done, info)

    Retourne toujours :
        state, reward, done, info
    """
    if len(step_result) == 3:
        next_state, reward, done = step_result
        return next_state, reward, done, {}

    elif len(step_result) == 4:
        return step_result

    else:
        raise ValueError("Format invalide pour step()")


# =========================================================
# Fonction d'entraînement principale
# =========================================================
def train_lineworld(n_episodes=5000):
    env = LineWorld()

    # Initialisation de l'agent Q-learning
    agent = TabularQAgent(
        alpha=0.1,
        gamma=0.99,
        epsilon=1.0,
        epsilon_decay=0.995,
        epsilon_min=0.05,
    )

    episode_rewards = []
    episode_lengths = []

    # Boucle principale sur les épisodes
    for episode in range(n_episodes):
        state = env.reset()
        done = False
        total_reward = 0
        steps = 0

        # Boucle d'un épisode
        while not done:
            valid_actions = env.get_actions()

            # Choix de l'action (epsilon-greedy)
            action = agent.choose_action(state, valid_actions)

            # Interaction avec l'environnement
            step_result = env.step(action)

            # On rend le format compatible
            next_state, reward, done, info = unpack_step(step_result)

            # Actions suivantes
            next_valid_actions = [] if done else env.get_actions()

            # Apprentissage Q-learning
            agent.learn(
                state=state,
                action=action,
                reward=reward,
                next_state=next_state,
                done=done,
                next_valid_actions=next_valid_actions,
            )

            # Mise à jour
            state = next_state
            total_reward += reward
            steps += 1

        # Réduction de l'exploration
        agent.decay_epsilon()

        episode_rewards.append(total_reward)
        episode_lengths.append(steps)

        # Logs
        if (episode + 1) % 500 == 0:
            avg_reward = sum(episode_rewards[-500:]) / 500
            avg_steps = sum(episode_lengths[-500:]) / 500

            print(
                f"Episode {episode + 1}/{n_episodes} | "
                f"avg_reward={avg_reward:.3f} | "
                f"avg_steps={avg_steps:.2f} | "
                f"epsilon={agent.epsilon:.3f}"
            )

    return agent, episode_rewards, episode_lengths


# =========================================================
# Lancement du script
# =========================================================
if __name__ == "__main__":
    print(" Training LineWorld...")

    agent, rewards, lengths = train_lineworld()

    print("\nQ-table finale :")
    for state, actions in sorted(agent.q_table.items()):
        print(f"State {state}: {actions}")