
import sys,os


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))



from Environnements.line_world import LineWorld
from Environnements.grid_world import GridWorld
from Environnements.tictactoe import TicTacToe
from Environnements.quarto import QuartoEnv
from Agents.Expert_Apprentice import ExpertApprenticeAgent

from experiment import (
    train_1player,
    train_tictactoe,
    train_quarto,
    evaluate_1player,
    evaluate_tictactoe,
    evaluate_quarto
)

def train(env_name="lineworld", episodes=10000):

    if env_name == "lineworld":
        env = LineWorld()
        agent = ExpertApprenticeAgent(8, 2)
        train_1player(env, agent, episodes, [1000, 5000, 10000], evaluate_1player)

    elif env_name == "gridworld":
        env = GridWorld()
        agent = ExpertApprenticeAgent(31, 4)
        train_1player(env, agent, episodes, [1000, 5000, 10000], evaluate_1player)

    elif env_name == "tictactoe":
        env = TicTacToe()
        agent = ExpertApprenticeAgent(27, 9)
        train_tictactoe(env, agent, episodes, [1000, 5000, 10000], evaluate_tictactoe)

    elif env_name == "quarto":
        env = QuartoEnv()
        agent = ExpertApprenticeAgent(105, 32)
        train_quarto(env, agent, episodes, [1000, 5000, 10000], evaluate_quarto)


if __name__ == "__main__":
    train("quarto", 10000)