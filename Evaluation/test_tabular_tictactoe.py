import os
import sys

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from Environnements.tictactoe import TicTacToe
from Agents.random_agent import RandomAgent
from Training.train_tabular_tictactoe import train_tictactoe


def evaluate_tictactoe(agent, n_games=1000):
    env = TicTacToe()
    random_opponent = RandomAgent()

    old_epsilon = agent.epsilon
    agent.epsilon = 0.0  # mode exploitation pure

    wins = 0
    draws = 0
    losses = 0

    for _ in range(n_games):
        state = env.reset()
        done = False

        while not done:
            # Tour agent
            valid_actions = env.get_actions()
            action = agent.choose_action(state, valid_actions)
            next_state, reward, done, info = env.step(action)

            if done:
                if info["winner"] == 1:
                    wins += 1
                elif info["winner"] == 0:
                    draws += 1
                break

            # Tour random
            opponent_action = random_opponent.Choisir_action(env)
            next_state, reward, done, info = env.step(opponent_action)

            if done:
                if info["winner"] == -1:
                    losses += 1
                elif info["winner"] == 0:
                    draws += 1
                break

            state = next_state

    agent.epsilon = old_epsilon

    return {
        "n_games": n_games,
        "wins": wins,
        "draws": draws,
        "losses": losses,
        "win_rate": wins / n_games,
        "draw_rate": draws / n_games,
        "loss_rate": losses / n_games,
    }


if __name__ == "__main__":
    agent, rewards, lengths = train_tictactoe(n_episodes=20000)
    metrics = evaluate_tictactoe(agent, n_games=1000)

    print("\nRésultats d'évaluation TicTacToe :")
    for k, v in metrics.items():
        print(f"{k}: {v}")