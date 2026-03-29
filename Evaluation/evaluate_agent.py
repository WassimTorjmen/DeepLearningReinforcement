import time


def evaluate_agent(env, agent, n_episodes=100):
    old_epsilon = agent.epsilon
    agent.epsilon = 0.0

    total_rewards = 0
    total_steps = 0
    total_action_time = 0.0

    for _ in range(n_episodes):
        state = env.reset()
        done = False
        steps = 0

        while not done:
            valid_actions = env.get_actions()

            start = time.perf_counter()
            action = agent.choose_action(state, valid_actions)
            total_action_time += time.perf_counter() - start

            next_state, reward, done, info = env.step(action)
            state = next_state

            total_rewards += reward
            steps += 1

        total_steps += steps

    agent.epsilon = old_epsilon

    return {
        "avg_reward": total_rewards / n_episodes,
        "avg_steps": total_steps / n_episodes,
        "avg_action_time": total_action_time / max(1, total_steps),
    }