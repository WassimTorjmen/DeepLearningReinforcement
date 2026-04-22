import os
import sys

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from Environnements.tictactoe import TicTacToe
from Agents.tabular_q_agent import TabularQAgent
from Agents.random_agent import RandomAgent


def train_tictactoe(n_episodes=20000):
    env = TicTacToe()

    agent = TabularQAgent(
        alpha=0.1,
        gamma=0.99,
        epsilon=1.0,
        epsilon_decay=0.9995,
        epsilon_min=0.05,
    )

    random_opponent = RandomAgent()

    episode_rewards = []
    episode_lengths = []

    wins = 0
    draws = 0
    losses = 0

    for episode in range(n_episodes):
        state = env.reset()
        done = False
        total_reward = 0
        steps = 0

        # L'agent joue toujours X (= joueur 1)
        while not done:
            # -----------------------------
            # 1) Tour de l'agent (X)
            # -----------------------------
            valid_actions = env.get_actions()
            action = agent.choose_action(state, valid_actions)

            next_state, reward, done, info = env.step(action)
            steps += 1

            # Si l'agent gagne ou si match nul juste après son coup
            if done:
                next_valid_actions = []
                agent.learn(
                    state=state,
                    action=action,
                    reward=reward,  # +1 si victoire agent, 0 si nul
                    next_state=next_state,
                    done=done,
                    next_valid_actions=next_valid_actions,
                )

                total_reward += reward

                if info["winner"] == 1:
                    wins += 1
                elif info["winner"] == 0:
                    draws += 1

                break

            # -----------------------------
            # 2) Tour de l'adversaire random (O)
            # -----------------------------
            opponent_action = random_opponent.Choisir_action(env)
            opponent_next_state, opponent_reward, done, opponent_info = env.step(opponent_action)
            steps += 1

            # Cas 1 : l'adversaire gagne
            if done and opponent_info["winner"] == -1:
                agent.learn(
                    state=state,
                    action=action,
                    reward=-1,  # très important
                    next_state=opponent_next_state,
                    done=True,
                    next_valid_actions=[],
                )

                total_reward += -1
                losses += 1
                break

            # Cas 2 : match nul après le coup adverse
            if done and opponent_info["winner"] == 0:
                agent.learn(
                    state=state,
                    action=action,
                    reward=0,
                    next_state=opponent_next_state,
                    done=True,
                    next_valid_actions=[],
                )

                total_reward += 0
                draws += 1
                break

            # Cas 3 : la partie continue
            next_valid_actions = env.get_actions()
            agent.learn(
                state=state,
                action=action,
                reward=0,
                next_state=opponent_next_state,
                done=False,
                next_valid_actions=next_valid_actions,
            )

            state = opponent_next_state
            total_reward += 0

        agent.decay_epsilon()

        episode_rewards.append(total_reward)
        episode_lengths.append(steps)

        if (episode + 1) % 1000 == 0:
            avg_reward = sum(episode_rewards[-1000:]) / 1000
            avg_steps = sum(episode_lengths[-1000:]) / 1000

            print(
                f"Episode {episode + 1}/{n_episodes} | "
                f"avg_reward={avg_reward:.3f} | "
                f"avg_steps={avg_steps:.2f} | "
                f"epsilon={agent.epsilon:.3f}"
            )

    print("\nStatistiques d'entraînement :")
    print(f"Victoires agent : {wins}")
    print(f"Matchs nuls     : {draws}")
    print(f"Défaites agent  : {losses}")

    return agent, episode_rewards, episode_lengths


if __name__ == "__main__":
    print("Training TicTacToe...")
    agent, rewards, lengths = train_tictactoe()
    print("\nTaille Q-table :", len(agent.q_table))