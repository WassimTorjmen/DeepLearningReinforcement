import sys
import os
base = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(base)
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, "Agents"))
sys.path.append(os.path.join(project_root, "Environnements"))

from experiment import (
    run_experiment_dqn,
    train_ddqn_er_1player,   evaluate_dqn_1player,
    train_ddqn_er_tictactoe, evaluate_dqn_tictactoe,
    train_ddqn_er_quarto,    evaluate_dqn_quarto,
)

from Environnements.line_world import LineWorld
from Environnements.grid_world  import GridWorld
from Environnements.tictactoe   import TicTacToe
from Environnements.quarto      import QuartoEnv

import ddqn_er

import ddqn_per

if __name__ == "__main__":

    # ════════════════════════════════════════════════════════════
    #  DOUBLE DQN + EXPERIENCE REPLAY
    #  buffer_capacity et batch_size réduits pour éviter blocage
    # ════════════════════════════════════════════════════════════
    run_experiment_dqn(
        env=LineWorld(size=6), env_name="LineWorld",
        train_fn=train_ddqn_er_1player, evaluate_fn=evaluate_dqn_1player,
        agent=ddqn_er.DoubleDQNWithERAgent(8, 2,
              buffer_capacity=5000, batch_size=32),
        agent_name="DDQN_ER",
        num_episodes=10_000, checkpoints=[1_000, 10_000],
    )
    run_experiment_dqn(
        env=GridWorld(rows=5, cols=5), env_name="GridWorld",
        train_fn=train_ddqn_er_1player, evaluate_fn=evaluate_dqn_1player,
        agent=ddqn_er.DoubleDQNWithERAgent(31, 4,
              buffer_capacity=5000, batch_size=32),
        agent_name="DDQN_ER",
        num_episodes=10_000, checkpoints=[1_000, 10_000],
    )
    run_experiment_dqn(
        env=TicTacToe(), env_name="TicTacToe",
        train_fn=train_ddqn_er_tictactoe, evaluate_fn=evaluate_dqn_tictactoe,
        agent=ddqn_er.DoubleDQNWithERAgent(27, 9, hidden_size=128,
              buffer_capacity=5000, batch_size=32),
        agent_name="DDQN_ER",
        num_episodes=100_000, checkpoints=[1_000, 10_000, 100_000],
    )
    run_experiment_dqn(
        env=QuartoEnv(), env_name="Quarto",
        train_fn=train_ddqn_er_quarto, evaluate_fn=evaluate_dqn_quarto,
        agent=ddqn_er.DoubleDQNWithERAgent(105, 32, hidden_size=256,
              buffer_capacity=5000, batch_size=32),
        agent_name="DDQN_ER",
        num_episodes=100_000, checkpoints=[1_000, 10_000, 100_000],
    )
"""
    # ════════════════════════════════════════════════════════════
    #  DOUBLE DQN + PRIORITIZED EXPERIENCE REPLAY
    # ════════════════════════════════════════════════════════════
    run_experiment_dqn(
        env=LineWorld(size=6), env_name="LineWorld",
        train_fn=train_ddqn_er_1player, evaluate_fn=evaluate_dqn_1player,
        agent=ddqn_per.DoubleDQNWithPERAgent(8, 2,
              buffer_capacity=500, batch_size=32),
        agent_name="DDQN_PER",
        num_episodes=10_000, checkpoints=[1_000, 10_000],
    )
    run_experiment_dqn(
        env=GridWorld(rows=5, cols=5), env_name="GridWorld",
        train_fn=train_ddqn_er_1player, evaluate_fn=evaluate_dqn_1player,
        agent=ddqn_per.DoubleDQNWithPERAgent(31, 4,
              buffer_capacity=500, batch_size=32),
        agent_name="DDQN_PER",
        num_episodes=10_000, checkpoints=[1_000, 10_000],
    )
    run_experiment_dqn(
        env=TicTacToe(), env_name="TicTacToe",
        train_fn=train_ddqn_er_tictactoe, evaluate_fn=evaluate_dqn_tictactoe,
        agent=ddqn_per.DoubleDQNWithPERAgent(27, 9, hidden_size=128,
              buffer_capacity=2_000, batch_size=32),
        agent_name="DDQN_PER",
        num_episodes=100_000, checkpoints=[1_000, 10_000, 100_000],
    )
    run_experiment_dqn(
        env=QuartoEnv(), env_name="Quarto",
        train_fn=train_ddqn_er_quarto, evaluate_fn=evaluate_dqn_quarto,
        agent=ddqn_per.DoubleDQNWithPERAgent(105, 32, hidden_size=256,
              buffer_capacity=2_000, batch_size=32),
        agent_name="DDQN_PER",
        num_episodes=100_000, checkpoints=[1_000, 10_000, 100_000],
    )

    """
