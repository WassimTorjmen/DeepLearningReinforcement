import os
import sys

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from Environnements.grid_world import GridWorld
from Agents.tabular_q_agent import TabularQAgent
from Evaluation.evaluate_agent import evaluate_agent
from Training.train_tabular_gridworld import train_gridworld


if __name__ == "__main__":
    print("Training GridWorld...")
    agent, rewards, lengths = train_gridworld()

    env = GridWorld()

    metrics = evaluate_agent(env, agent, n_episodes=200)

    print("\nRésultats d'évaluation GridWorld :")
    print(metrics)