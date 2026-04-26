import sys
import os
base = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(base)
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, "Agents"))
sys.path.append(os.path.join(project_root, "Environnements"))

from experiment import (
    run_experiment,
    run_experiment_no_training,
    train_1player, evaluate_1player,
    train_tictactoe, evaluate_tictactoe,
    train_quarto, evaluate_quarto,
    evaluate_no_training_1player,
    evaluate_no_training_tictactoe,
    evaluate_no_training_quarto,
    train_alphazero_quarto,
    train_alphazero_tictactoe,
    train_alphazero_1player,
    run_experiment_dqn,
    train_dqn_1player,   
    evaluate_dqn_1player,
    train_dqn_tictactoe, 
    evaluate_dqn_tictactoe,
    train_dqn_quarto,    
    evaluate_dqn_quarto,
    train_ddqn_er_1player,
    train_ddqn_er_tictactoe,
    train_ddqn_er_quarto
)

from Environnements.line_world import LineWorld
from Environnements.grid_world  import GridWorld
from Environnements.tictactoe   import TicTacToe
from Environnements.quarto      import QuartoEnv

import REINFORCE
import Reinforce_mean_baseline
import Reinforce_critic
import PPO_A2C
import Random_Rollout
import Alpha_zero
import MuZero
import DQN
import ddqn
import ddqn_er
import ddqn_per

if __name__ == "__main__":
    # ════════════════════════════════════════════════════════════
    #  DQN
    #  LineWorld et GridWorld : paramètres de base
    #  TicTacToe : hidden_size=128, 100 000 épisodes
    #  Quarto    : hidden_size=256, 100 000 épisodes
    # ════════════════════════════════════════════════════════════
    #run_experiment_dqn(
    #    env=LineWorld(size=6), env_name="LineWorld",
    #    train_fn=train_dqn_1player, evaluate_fn=evaluate_dqn_1player,
    #    agent=DQN.DQNAgent(8, 2),
    #    agent_name="DQN",
    #    num_episodes=10_000, checkpoints=[1_000, 10_000],
    #)
    #run_experiment_dqn(
    #    env=GridWorld(rows=5, cols=5), env_name="GridWorld",
    #    train_fn=train_dqn_1player, evaluate_fn=evaluate_dqn_1player,
    #    agent=DQN.DQNAgent(31, 4),
    #    agent_name="DQN",
    #    num_episodes=10_000, checkpoints=[1_000, 10_000],
    #)
    #run_experiment_dqn(
    #    env=TicTacToe(), env_name="TicTacToe",
    #    train_fn=train_dqn_tictactoe, evaluate_fn=evaluate_dqn_tictactoe,
    #    agent=DQN.DQNAgent(27, 9, hidden_size=128),
    #    agent_name="DQN",
    #    num_episodes=100_000, checkpoints=[1_000, 10_000, 100_000],
    #)
    #run_experiment_dqn(
    #    env=QuartoEnv(), env_name="Quarto",
    #    train_fn=train_dqn_quarto, evaluate_fn=evaluate_dqn_quarto,
    #    agent=DQN.DQNAgent(105, 32, hidden_size=256),
    #    agent_name="DQN",
    #    num_episodes=100_000, checkpoints=[1_000, 10_000, 100_000],
    #)



    # ════════════════════════════════════════════════════════════
    #  DOUBLE DQN
    # ════════════════════════════════════════════════════════════
    #run_experiment_dqn(
    #    env=LineWorld(size=6), env_name="LineWorld",
    #    train_fn=train_dqn_1player, evaluate_fn=evaluate_dqn_1player,
    #    agent=ddqn.DoubleDQNAgent(8, 2),
    #    agent_name="DoubleDQN",
    #    num_episodes=10_000, checkpoints=[1_000, 10_000],
    #)
    #run_experiment_dqn(
    #    env=GridWorld(rows=5, cols=5), env_name="GridWorld",
    #    train_fn=train_dqn_1player, evaluate_fn=evaluate_dqn_1player,
    #    agent=ddqn.DoubleDQNAgent(31, 4),
    #    agent_name="DoubleDQN",
    #    num_episodes=10_000, checkpoints=[1_000, 10_000],
    #)
    #run_experiment_dqn(
    #    env=TicTacToe(), env_name="TicTacToe",
    #    train_fn=train_dqn_tictactoe, evaluate_fn=evaluate_dqn_tictactoe,
    #    agent=ddqn.DoubleDQNAgent(27, 9, hidden_size=128),
    #    agent_name="DoubleDQN",
    #    num_episodes=100_000, checkpoints=[1_000, 10_000, 100_000],
    #)
    #run_experiment_dqn(
    #    env=QuartoEnv(), env_name="Quarto",
    #    train_fn=train_dqn_quarto, evaluate_fn=evaluate_dqn_quarto,
    #    agent=ddqn.DoubleDQNAgent(105, 32, hidden_size=256),
    #    agent_name="DoubleDQN",
    #    num_episodes=100_000, checkpoints=[1_000, 10_000, 100_000],
    #)
    # ════════════════════════════════════════════════════════════
    #  DOUBLE DQN + EXPERIENCE REPLAY
    #  buffer_capacity et batch_size réduits pour éviter blocage
    # ════════════════════════════════════════════════════════════
    #run_experiment_dqn(
    #    env=LineWorld(size=6), env_name="LineWorld",
    #    train_fn=train_ddqn_er_1player, evaluate_fn=evaluate_dqn_1player,
    #    agent=ddqn_er.DoubleDQNWithERAgent(8, 2,
    #          buffer_capacity=5000, batch_size=32),
    #    agent_name="DDQN_ER",
    #   num_episodes=10_000, checkpoints=[1_000, 10_000],
    #)
    #run_experiment_dqn(
    #    env=GridWorld(rows=5, cols=5), env_name="GridWorld",
    #    train_fn=train_ddqn_er_1player, evaluate_fn=evaluate_dqn_1player,
    #    agent=ddqn_er.DoubleDQNWithERAgent(31, 4,
    #          buffer_capacity=5000, batch_size=32),
    #    agent_name="DDQN_ER",
    #    num_episodes=10_000, checkpoints=[1_000, 10_000],
    #)
    #run_experiment_dqn(
    #    env=TicTacToe(), env_name="TicTacToe",
    #    train_fn=train_ddqn_er_tictactoe, evaluate_fn=evaluate_dqn_tictactoe,
    #    agent=ddqn_er.DoubleDQNWithERAgent(27, 9, hidden_size=128,
    #          buffer_capacity=5000, batch_size=32, ),
    #    agent_name="DDQN_ER_1M",
    #    num_episodes=1_000_000, checkpoints=[1_000, 10_000, 100_000, 1_000_000],
    #)
    """run_experiment_dqn(
        env=QuartoEnv(), env_name="Quarto",
        train_fn=train_ddqn_er_quarto, evaluate_fn=evaluate_dqn_quarto,
        agent=ddqn_er.DoubleDQNWithERAgent(105, 32, hidden_size=256,
              buffer_capacity=5000, batch_size=32),
        agent_name="DDQN_ER",
        num_episodes=100_000, checkpoints=[1_000, 10_000, 100_000],
    )"""

    # ════════════════════════════════════════════════════════════
    #  DOUBLE DQN + PRIORITIZED EXPERIENCE REPLAY
    # ════════════════════════════════════════════════════════════
    #run_experiment_dqn(
    #    env=LineWorld(size=6), env_name="LineWorld",
    #    train_fn=train_ddqn_er_1player, evaluate_fn=evaluate_dqn_1player,
    #    agent=ddqn_per.DoubleDQNWithPERAgent(8, 2,
    #          buffer_capacity=500, batch_size=32),
    #    agent_name="DDQN_PER",
    #    num_episodes=10_000, checkpoints=[1_000, 10_000],
    #)
    #run_experiment_dqn(
    #    env=GridWorld(rows=5, cols=5), env_name="GridWorld",
    #    train_fn=train_ddqn_er_1player, evaluate_fn=evaluate_dqn_1player,
    #    agent=ddqn_per.DoubleDQNWithPERAgent(31, 4,
    #          buffer_capacity=500, batch_size=32),
    #    agent_name="DDQN_PER",
    #    num_episodes=10_000, checkpoints=[1_000, 10_000],
    #)
    run_experiment_dqn(
        env=TicTacToe(), env_name="TicTacToe",
        train_fn=train_ddqn_er_tictactoe, evaluate_fn=evaluate_dqn_tictactoe,
        agent=ddqn_per.DoubleDQNWithPERAgent(27, 9, hidden_size=128,
              buffer_capacity=2_000, batch_size=32),
        agent_name="DDQN_PER",
        num_episodes=100_000, checkpoints=[1_000, 10_000, 100_000],
    )
    #run_experiment_dqn(
    #    env=QuartoEnv(), env_name="Quarto",
    #    train_fn=train_ddqn_er_quarto, evaluate_fn=evaluate_dqn_quarto,
    #    agent=ddqn_per.DoubleDQNWithPERAgent(105, 32, hidden_size=256,
    #          buffer_capacity=2_000, batch_size=32),
    #    agent_name="DDQN_PER",
    #    num_episodes=100_000, checkpoints=[1_000, 10_000, 100_000],
    #)



    # ════════════════════════════════════════════════════════════
    #  REINFORCE
    # ════════════════════════════════════════════════════════════
    # run_experiment(
    #     env=LineWorld(size=6), env_name="LineWorld",
    #     train_fn=train_1player, evaluate_fn=evaluate_1player,
    #     agent=REINFORCE.ReinforceAgent(8, 2),
    #     agent_name="REINFORCE",
    #     num_episodes=10_000, checkpoints=[1_000, 10_000],
    # )
    # run_experiment(
    #     env=GridWorld(rows=5, cols=5), env_name="GridWorld",
    #     train_fn=train_1player, evaluate_fn=evaluate_1player,
    #     agent=REINFORCE.ReinforceAgent(31, 4),
    #     agent_name="REINFORCE",
    #     num_episodes=10_000, checkpoints=[1_000, 10_000],
    # )
    # run_experiment(
    #     env=TicTacToe(), env_name="TicTacToe",
    #     train_fn=train_tictactoe, evaluate_fn=evaluate_tictactoe,
    #     agent=REINFORCE.ReinforceAgent(27, 9),
    #     agent_name="REINFORCE",
    #     num_episodes=10_000, checkpoints=[1_000, 10_000],
    # )
    # run_experiment(
    #     env=QuartoEnv(), env_name="Quarto",
    #     train_fn=train_quarto, evaluate_fn=evaluate_quarto,
    #     agent=REINFORCE.ReinforceAgent(105, 32, hidden_size=128),
    #     agent_name="REINFORCE",
    #     num_episodes=10_000, checkpoints=[1_000, 10_000],
    # )

    # ════════════════════════════════════════════════════════════
    #  REINFORCE MEAN BASELINE
    # ════════════════════════════════════════════════════════════
    # run_experiment(
    #     env=LineWorld(size=6), env_name="LineWorld",
    #     train_fn=train_1player, evaluate_fn=evaluate_1player,
    #     agent=Reinforce_mean_baseline.ReinforceAgentMeanBaseline(8, 2),
    #     agent_name="REINFORCE_MeanBaseline",
    #     num_episodes=10_000, checkpoints=[1_000, 10_000],
    # )
    # run_experiment(
    #     env=GridWorld(rows=5, cols=5), env_name="GridWorld",
    #     train_fn=train_1player, evaluate_fn=evaluate_1player,
    #     agent=Reinforce_mean_baseline.ReinforceAgentMeanBaseline(31, 4),
    #     agent_name="REINFORCE_MeanBaseline",
    #     num_episodes=10_000, checkpoints=[1_000, 10_000],
    # )
    # run_experiment(
    #     env=TicTacToe(), env_name="TicTacToe",
    #     train_fn=train_tictactoe, evaluate_fn=evaluate_tictactoe,
    #     agent=Reinforce_mean_baseline.ReinforceAgentMeanBaseline(27, 9),
    #     agent_name="REINFORCE_MeanBaseline",
    #     num_episodes=10_000, checkpoints=[1_000, 10_000],
    # )
    # run_experiment(
    #     env=QuartoEnv(), env_name="Quarto",
    #     train_fn=train_quarto, evaluate_fn=evaluate_quarto,
    #     agent=Reinforce_mean_baseline.ReinforceAgentMeanBaseline(105, 32, hidden_size=128),
    #     agent_name="REINFORCE_MeanBaseline",
    #     num_episodes=10_000, checkpoints=[1_000, 10_000],
    # )

    # ════════════════════════════════════════════════════════════
    #  REINFORCE WITH CRITIC
    # ════════════════════════════════════════════════════════════
    # run_experiment(
    #     env=LineWorld(size=6), env_name="LineWorld",
    #     train_fn=train_1player, evaluate_fn=evaluate_1player,
    #     agent=REINFORCE_critic.ReinforceAgentCritic(8, 2),
    #     agent_name="REINFORCE_Critic",
    #     num_episodes=10_000, checkpoints=[1_000, 10_000],
    # )
    # run_experiment(
    #     env=GridWorld(rows=5, cols=5), env_name="GridWorld",
    #     train_fn=train_1player, evaluate_fn=evaluate_1player,
    #     agent=REINFORCE_critic.ReinforceAgentCritic(31, 4),
    #     agent_name="REINFORCE_Critic",
    #     num_episodes=10_000, checkpoints=[1_000, 10_000],
    # )
    # run_experiment(
    #     env=TicTacToe(), env_name="TicTacToe",
    #     train_fn=train_tictactoe, evaluate_fn=evaluate_tictactoe,
    #     agent=REINFORCE_critic.ReinforceAgentCritic(27, 9),
    #     agent_name="REINFORCE_Critic",
    #     num_episodes=10_000, checkpoints=[1_000, 10_000],
    # )
    # run_experiment(
    #     env=QuartoEnv(), env_name="Quarto",
    #     train_fn=train_quarto, evaluate_fn=evaluate_quarto,
    #     agent=REINFORCE_critic.ReinforceAgentCritic(105, 32, hidden_size=128),
    #     agent_name="REINFORCE_Critic",
    #     num_episodes=10_000, checkpoints=[1_000, 10_000],
    # )

    # ════════════════════════════════════════════════════════════
    #  PPO A2C
    # ════════════════════════════════════════════════════════════
    #run_experiment(
    #    train_fn=train_1player, evaluate_fn=evaluate_1player,
    #    agent=PPO_A2C.PPOAgent(8, 2),
    #    agent_name="PPO_A2C",
    #    num_episodes=10_000, checkpoints=[1_000, 10_000],
    #)
    #run_experiment(
    #    env=GridWorld(rows=5, cols=5), env_name="GridWorld",
    #    train_fn=train_1player, evaluate_fn=evaluate_1player,
    #    agent=PPO_A2C.PPOAgent(31, 4),
    #    agent_name="PPO_A2C",
    #    num_episodes=10_000, checkpoints=[1_000, 10_000],
    #)
    #run_experiment(
    #    env=TicTacToe(), env_name="TicTacToe",
    #    train_fn=train_tictactoe, evaluate_fn=evaluate_tictactoe,
    #    agent=PPO_A2C.PPOAgent(27, 9),
    #    agent_name="PPO_A2C",
    #    num_episodes=10_000, checkpoints=[1_000, 10_000],
    #)
    #run_experiment(
    #    env=QuartoEnv(), env_name="Quarto",
    #    train_fn=train_quarto, evaluate_fn=evaluate_quarto,
    #    agent=PPO_A2C.PPOAgent(105, 32, hidden_size=128),
    #    agent_name="PPO_A2C",
    #    num_episodes=10_000, checkpoints=[1_000, 10_000],
    #)

    
