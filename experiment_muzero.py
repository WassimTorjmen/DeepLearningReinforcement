# =========================================================
# MAIN MUZERO
# =========================================================

import argparse

from Training.train_agents import main as train_main
from Evaluation.evaluate_agents import main as eval_main
from experiment_agents import main as exp_main


def run_train(env, episodes):
    print("\n=== TRAIN MUZERO ===")
    train_main([
        "--agent", "muzero",
        "--env", env,
        "--episodes", str(episodes)
    ])


def run_evaluate(env, n_games):
    print("\n=== EVALUATE MUZERO ===")
    eval_main([
        "--agent", "muzero",
        "--env", env,
        "--n_games", str(n_games)
    ])


def run_experiment(env, episodes, n_eval):
    print("\n=== EXPERIMENT MUZERO ===")
    exp_main([
        "--agent", "muzero",
        "--env", env,
        "--episodes", str(episodes),
        "--n_eval", str(n_eval)
    ])


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--mode", choices=["train", "eval", "experiment"], default="experiment")
    parser.add_argument("--env", default="all")
    parser.add_argument("--episodes", type=int, default=5000)
    parser.add_argument("--n_eval", type=int, default=200)

    args = parser.parse_args()

    if args.mode == "train":
        run_train(args.env, args.episodes)

    elif args.mode == "eval":
        run_evaluate(args.env, args.n_eval)

    elif args.mode == "experiment":
        run_experiment(args.env, args.episodes, args.n_eval)