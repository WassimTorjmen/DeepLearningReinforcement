import time


def unpack_step(step_result):
    """
    Permet de gérer tous les environnements :
    - (state, reward, done)
    - (state, reward, done, info)
    """
    if len(step_result) == 3:
        next_state, reward, done = step_result
        return next_state, reward, done, {}
    elif len(step_result) == 4:
        return step_result
    else:
        raise ValueError("Format de step invalide")


def evaluate_agent(env, agent, n_episodes=100):
    # On sauvegarde epsilon
    old_epsilon = agent.epsilon

    # Mode exploitation uniquement
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

            # Temps de décision
            start = time.perf_counter()
            action = agent.choose_action(state, valid_actions)
            total_action_time += time.perf_counter() - start

            # Step robuste
            next_state, reward, done, info = unpack_step(env.step(action))

            state = next_state
            total_rewards += reward
            steps += 1

        total_steps += steps

    # On restaure epsilon
    agent.epsilon = old_epsilon

    return {
        "avg_reward": total_rewards / n_episodes,
        "avg_steps": total_steps / n_episodes,
        "avg_action_time": total_action_time / max(1, total_steps),
    }