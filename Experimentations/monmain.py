import sys
import os

base = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(base, "Agents"))
sys.path.append(os.path.join(base, "Environnements"))

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
    train_alphazero_1player
)

from line_world import LineWorld
from grid_world  import GridWorld
from tictactoe   import TicTacToe
from quarto      import QuartoEnv

import REINFORCE
import Reinforce_mean_baseline
import Reinforce_critic
import PPO_A2C
import Random_Rollout
import Alpha_zero
import MuZero

if __name__ == "__main__":

    #════════════════════════════════════════════════════════════
    # REINFORCE
    # ════════════════════════════════════════════════════════════
    """run_experiment(
        env=LineWorld(size=6), env_name="LineWorld",
        train_fn=train_1player, evaluate_fn=evaluate_1player,
        agent=REINFORCE.ReinforceAgent(8, 2),
        agent_name="REINFORCE",
        num_episodes=10_000, checkpoints=[1_000, 10_000],
    )
    run_experiment(
        env=GridWorld(rows=5, cols=5), env_name="GridWorld",
        train_fn=train_1player, evaluate_fn=evaluate_1player,
        agent=REINFORCE.ReinforceAgent(31, 4),
        agent_name="REINFORCE",
        num_episodes=10_000, checkpoints=[1_000, 10_000],
     )
    run_experiment(
        env=TicTacToe(), env_name="TicTacToe",
        train_fn=train_tictactoe, evaluate_fn=evaluate_tictactoe,
        agent=REINFORCE.ReinforceAgent(27, 9),
        agent_name="REINFORCE",
        num_episodes=10_000, checkpoints=[1_000, 10_000],
     )
    run_experiment(
        env=QuartoEnv(), env_name="Quarto",
        train_fn=train_quarto, evaluate_fn=evaluate_quarto,
        agent=REINFORCE.ReinforceAgent(105, 32, hidden_size=128),
        agent_name="REINFORCE",
        num_episodes=10_000, checkpoints=[1_000, 10_000],
     )
    
    # TicTacToe — test avec lr plus petit et réseau plus grand
    #run_experiment(
    #    env=TicTacToe(), env_name="TicTacToe",
    #    train_fn=train_tictactoe, evaluate_fn=evaluate_tictactoe,
    #    agent=REINFORCE.ReinforceAgent(27, 9, lr=5e-4, hidden_size=128),  # ← changé
    #    agent_name="REINFORCE_lr5e4_h128",
    #    num_episodes=10_000, checkpoints=[1_000, 10_000],
    #)

    # Quarto — test avec lr très petit et réseau plus grand
    #run_experiment(
    #    env=QuartoEnv(), env_name="Quarto",
    #    train_fn=train_quarto, evaluate_fn=evaluate_quarto,
    #    agent=REINFORCE.ReinforceAgent(105, 32, lr=1e-4, hidden_size=256),  # ← changé
    #    agent_name="REINFORCE_lr1e4_h256",
    #    num_episodes=10_000, checkpoints=[1_000, 10_000],
    #)

    # TicTacToe — test avec lr plus petit et réseau plus grand
    run_experiment(
        env=TicTacToe(), env_name="TicTacToe",
        train_fn=train_tictactoe, evaluate_fn=evaluate_tictactoe,
        agent=REINFORCE.ReinforceAgent(27, 9, lr=5e-4, hidden_size=128),  # ← changé
        agent_name="REINFORCE_lr5e4_h128_100000",
        num_episodes=100_000, checkpoints=[1_000, 10_000, 100_000]          
    )
"""
    # Quarto — test avec lr très petit et réseau plus grand
    run_experiment(
        env=QuartoEnv(), env_name="Quarto",
        train_fn=train_quarto, evaluate_fn=evaluate_quarto,
        agent=REINFORCE.ReinforceAgent(105, 32, lr=1e-4, hidden_size=256),  # ← changé
        agent_name="REINFORCE_lr1e4_h256_100000",
        num_episodes=100_000, checkpoints=[1_000, 10_000, 100_000]
    )
