import numpy as np
import time
import torch
import matplotlib.pyplot as plt


def evaluate_1player(env, agent, n_games=500, max_steps=200):
    scores, lengths, times = [], [], []
    for _ in range(n_games):
        env.reset()
        state = env.encode_state()
        steps = 0
        while not env.done and steps < max_steps:
            available = env.get_actions()
            t0        = time.perf_counter()
            state_t   = torch.FloatTensor(state).unsqueeze(0).to(agent.device)
            probs     = agent.policy(state_t).squeeze(0)
            mask      = torch.zeros(agent.num_actions)
            mask[available] = 1.0
            probs     = probs * mask
            probs     = probs / (probs.sum() + 1e-8)
            action    = int(probs.argmax().item())
            t1        = time.perf_counter()
            times.append((t1 - t0) * 1000)
            _, reward, _ = env.step(action)
            state        = env.encode_state()
            steps       += 1
        scores.append(reward)
        lengths.append(steps)
    return {
        "score_moyen":   round(float(np.mean(scores)),  4),
        "longueur_moy":  round(float(np.mean(lengths)), 2),
        "temps_coup_ms": round(float(np.mean(times)),   4),
    }


def evaluate_tictactoe(env, agent, n_games=500):
    scores, lengths, times = [], [], []
    for _ in range(n_games):
        env.reset()
        steps = 0
        while not env.done:
            state     = env.encode_state()
            available = env.get_actions()
            t0      = time.perf_counter()
            state_t = torch.FloatTensor(state).unsqueeze(0).to(agent.device)
            probs   = agent.policy(state_t).squeeze(0)
            mask    = torch.zeros(agent.num_actions)
            mask[available] = 1.0
            probs   = probs * mask
            probs   = probs / (probs.sum() + 1e-8)
            action  = int(probs.argmax().item())
            t1      = time.perf_counter()
            times.append((t1 - t0) * 1000)
            _, _, done, info = env.step(action)
            steps += 1
            if done:
                scores.append(1 if info["winner"] == 1 else 0)
                lengths.append(steps)
                break
            _, _, done, info = env.step(np.random.choice(env.get_actions()))
            steps += 1
            if done:
                scores.append(-1 if info["winner"] == -1 else 0)
                lengths.append(steps)
                break
    return {
        "score_moyen":   round(float(np.mean(scores)),  4),
        "longueur_moy":  round(float(np.mean(lengths)), 2),
        "temps_coup_ms": round(float(np.mean(times)),   4),
    }


def evaluate_quarto(env, agent, n_games=500):
    scores, lengths, times = [], [], []
    for _ in range(n_games):
        env.reset()
        steps = 0
        while not env.done:
            available = env.get_actions()
            if env.current_player == 1:
                state   = env.encode_state()
                t0      = time.perf_counter()
                state_t = torch.FloatTensor(state).unsqueeze(0).to(agent.device)
                probs   = agent.policy(state_t).squeeze(0)
                mask    = torch.zeros(agent.num_actions)
                mask[available] = 1.0
                probs   = probs * mask
                probs   = probs / (probs.sum() + 1e-8)
                action  = int(probs.argmax().item())
                t1      = time.perf_counter()
                times.append((t1 - t0) * 1000)
            else:
                action = int(np.random.choice(available))
            _, _, done, _ = env.step(action)
            steps += 1
            if done:
                scores.append(1 if env.winner == 1 else (-1 if env.winner == 2 else 0))
                lengths.append(steps)
                break
    return {
        "score_moyen":   round(float(np.mean(scores)),  4),
        "longueur_moy":  round(float(np.mean(lengths)), 2),
        "temps_coup_ms": round(float(np.mean(times)),   4),
    }


def _parse_loss(loss_result):
    if isinstance(loss_result, tuple):
        return loss_result[0], loss_result[1]
    return loss_result, None


def train_1player(env, agent, num_episodes, checkpoints, evaluate_fn, window=100, max_steps=200):
    all_policy_losses, all_critic_losses, all_rewards, eval_results = [], [], [], {}
    checkpoints    = sorted(checkpoints)
    next_check_idx = 0
    for episode in range(1, num_episodes + 1):
        env.reset()
        state          = env.encode_state()
        episode_reward = 0.0
        steps          = 0
        while not env.done and steps < max_steps:
            available       = env.get_actions()
            action          = agent.select_action(state, available)
            _, reward, _    = env.step(action)
            agent.store_reward(reward)
            state           = env.encode_state()
            episode_reward += reward
            steps          += 1
        policy_loss, critic_loss = _parse_loss(agent.learn())
        all_policy_losses.append(policy_loss)
        if critic_loss is not None:
            all_critic_losses.append(critic_loss)
        all_rewards.append(episode_reward)
        if next_check_idx < len(checkpoints) and episode == checkpoints[next_check_idx]:
            metrics = evaluate_fn(env, agent)
            eval_results[episode] = metrics
            print(f"  {episode:>9,} épisodes | Score : {metrics['score_moyen']:.4f} | "
                  f"Longueur : {metrics['longueur_moy']:.2f} | Temps/coup : {metrics['temps_coup_ms']:.4f} ms")
            next_check_idx += 1
    return all_rewards, all_policy_losses, all_critic_losses, eval_results


def train_tictactoe(env, agent, num_episodes, checkpoints, evaluate_fn, window=100):
    all_policy_losses, all_critic_losses, all_rewards, eval_results = [], [], [], {}
    checkpoints    = sorted(checkpoints)
    next_check_idx = 0
    for episode in range(1, num_episodes + 1):
        env.reset()
        episode_reward = 0.0
        while not env.done:
            state     = env.encode_state()
            available = env.get_actions()
            action    = agent.select_action(state, available)
            _, _, done, info = env.step(action)
            if done:
                reward = 1 if info["winner"] == 1 else 0
                agent.store_reward(reward)
                episode_reward += reward
                break
            _, _, done, info = env.step(np.random.choice(env.get_actions()))
            if done:
                reward = -1 if info["winner"] == -1 else 0
                agent.store_reward(reward)
                episode_reward += reward
                break
            agent.store_reward(0)
        policy_loss, critic_loss = _parse_loss(agent.learn())
        all_policy_losses.append(policy_loss)
        if critic_loss is not None:
            all_critic_losses.append(critic_loss)
        all_rewards.append(episode_reward)
        if next_check_idx < len(checkpoints) and episode == checkpoints[next_check_idx]:
            metrics = evaluate_fn(env, agent)
            eval_results[episode] = metrics
            print(f"  {episode:>9,} épisodes | Score : {metrics['score_moyen']:.4f} | "
                  f"Longueur : {metrics['longueur_moy']:.2f} | Temps/coup : {metrics['temps_coup_ms']:.4f} ms")
            next_check_idx += 1
    return all_rewards, all_policy_losses, all_critic_losses, eval_results


def train_quarto(env, agent, num_episodes, checkpoints, evaluate_fn, window=100):
    all_policy_losses, all_critic_losses, all_rewards, eval_results = [], [], [], {}
    checkpoints    = sorted(checkpoints)
    next_check_idx = 0
    for episode in range(1, num_episodes + 1):
        env.reset()
        episode_reward = 0.0
        while not env.done:
            available = env.get_actions()
            if env.current_player == 1:
                state  = env.encode_state()
                action = agent.select_action(state, available)
                _, _, done, _ = env.step(action)
                if done:
                    reward = 1 if env.winner == 1 else (-1 if env.winner == 2 else 0)
                    agent.store_reward(reward)
                    episode_reward += reward
                else:
                    agent.store_reward(0)
            else:
                action = int(np.random.choice(available))
                _, _, done, _ = env.step(action)
                if done and hasattr(agent, 'saved_log_probs') and len(agent.saved_log_probs) > 0:
                    reward = -1 if env.winner == 2 else 0
                    agent.store_reward(reward)
                    episode_reward += reward
        policy_loss, critic_loss = _parse_loss(agent.learn())
        all_policy_losses.append(policy_loss)
        if critic_loss is not None:
            all_critic_losses.append(critic_loss)
        all_rewards.append(episode_reward)
        if next_check_idx < len(checkpoints) and episode == checkpoints[next_check_idx]:
            metrics = evaluate_fn(env, agent)
            eval_results[episode] = metrics
            print(f"  {episode:>9,} épisodes | Score : {metrics['score_moyen']:.4f} | "
                  f"Longueur : {metrics['longueur_moy']:.2f} | Temps/coup : {metrics['temps_coup_ms']:.4f} ms")
            next_check_idx += 1
    return all_rewards, all_policy_losses, all_critic_losses, eval_results


def plot_results(all_rewards, all_policy_losses, all_critic_losses, env_name, agent_name, window=100):
    smoothed = [np.mean(all_rewards[max(0, i-window):i+1]) for i in range(len(all_rewards))]
    n_plots = 3 if all_critic_losses else 2
    fig, axes = plt.subplots(n_plots, 1, figsize=(10, 4*n_plots))
    fig.suptitle(f"{agent_name} - {env_name}", fontsize=14)
    axes[0].plot(smoothed, color="steelblue")
    axes[0].set_ylabel(f"Reward moyen ({window} épisodes)")
    axes[0].set_xlabel("Épisode")
    axes[0].grid(True, alpha=0.3)
    axes[1].plot(all_policy_losses, color="steelblue")
    axes[1].set_ylabel("Policy Loss")
    axes[1].set_xlabel("Épisode")
    axes[1].set_title("Policy Loss par épisode")
    axes[1].grid(True, alpha=0.3)
    if all_critic_losses:
        axes[2].plot(all_critic_losses, color="darkorange")
        axes[2].set_ylabel("Critic Loss")
        axes[2].set_xlabel("Épisode")
        axes[2].set_title("Critic Loss par épisode")
        axes[2].grid(True, alpha=0.3)
    plt.tight_layout()
    filename = f"{agent_name}_{env_name}.png".replace(" ", "_")
    plt.savefig(filename, dpi=150)
    plt.show()
    print(f"Graphique sauvegardé → {filename}")


def run_experiment(env, env_name, train_fn, evaluate_fn, agent, agent_name, num_episodes, checkpoints):
    print("=" * 65)
    print(f"  {agent_name}  —  {env_name}")
    print("=" * 65)
    all_rewards, all_policy_losses, all_critic_losses, eval_results = train_fn(
        env, agent, num_episodes, checkpoints, evaluate_fn
    )
    print()
    print("=" * 65)
    print("  RÉSULTATS  (policy évaluée sur 500 parties, sans exploration)")
    print("=" * 65)
    print(f"{'Épisodes':>12} | {'Score moyen':>12} | {'Longueur moy':>13} | {'Temps/coup':>12}")
    print("-" * 65)
    for ep, m in eval_results.items():
        print(f"{ep:>12,} | {m['score_moyen']:>12.4f} | {m['longueur_moy']:>13.2f} | {m['temps_coup_ms']:>11.4f}ms")
    plot_results(all_rewards, all_policy_losses, all_critic_losses, env_name, agent_name)
    agent.save(f"{agent_name}_{env_name}.pt")
    print(f"Modèle sauvegardé → {agent_name}_{env_name}.pt\n")


# ══ AGENTS SANS ENTRAÎNEMENT ══════════════════════════════════

def evaluate_no_training_1player(env, agent, n_games=500, max_steps=200):
    scores, lengths, times = [], [], []
    for _ in range(n_games):
        env.reset()
        steps = 0
        while not env.done and steps < max_steps:
            available = env.get_actions()
            t0        = time.perf_counter()
            action    = agent.select_action(env, available)
            t1        = time.perf_counter()
            times.append((t1 - t0) * 1000)
            _, reward, _ = env.step(action)
            steps       += 1
        scores.append(reward)
        lengths.append(steps)
    return {
        "score_moyen":   round(float(np.mean(scores)),  4),
        "longueur_moy":  round(float(np.mean(lengths)), 2),
        "temps_coup_ms": round(float(np.mean(times)),   4),
    }


def evaluate_no_training_tictactoe(env, agent, n_games=500):
    scores, lengths, times = [], [], []
    for _ in range(n_games):
        env.reset()
        steps = 0
        while not env.done:
            available = env.get_actions()
            t0        = time.perf_counter()
            action    = agent.select_action(env, available)
            t1        = time.perf_counter()
            times.append((t1 - t0) * 1000)
            _, _, done, info = env.step(action)
            steps += 1
            if done:
                scores.append(1 if info["winner"] == 1 else 0)
                lengths.append(steps)
                break
            _, _, done, info = env.step(np.random.choice(env.get_actions()))
            steps += 1
            if done:
                scores.append(-1 if info["winner"] == -1 else 0)
                lengths.append(steps)
                break
    return {
        "score_moyen":   round(float(np.mean(scores)),  4),
        "longueur_moy":  round(float(np.mean(lengths)), 2),
        "temps_coup_ms": round(float(np.mean(times)),   4),
    }


def evaluate_no_training_quarto(env, agent, n_games=500):
    scores, lengths, times = [], [], []
    for _ in range(n_games):
        env.reset()
        steps = 0
        while not env.done:
            available = env.get_actions()
            if env.current_player == 1:
                t0     = time.perf_counter()
                action = agent.select_action(env, available)
                t1     = time.perf_counter()
                times.append((t1 - t0) * 1000)
            else:
                action = int(np.random.choice(available))
            _, _, done, _ = env.step(action)
            steps += 1
            if done:
                scores.append(1 if env.winner == 1 else (-1 if env.winner == 2 else 0))
                lengths.append(steps)
                break
    return {
        "score_moyen":   round(float(np.mean(scores)),  4),
        "longueur_moy":  round(float(np.mean(lengths)), 2),
        "temps_coup_ms": round(float(np.mean(times)),   4),
    }


def run_experiment_no_training(env, env_name, evaluate_fn, agent, agent_name, n_games=500):
    print("=" * 65)
    print(f"  {agent_name}  —  {env_name}")
    print("=" * 65)
    metrics = evaluate_fn(env, agent, n_games=n_games)
    print()
    print("=" * 65)
    print(f"  RÉSULTATS  (évaluation sur {n_games} parties)")
    print("=" * 65)
    print(f"{'Score moyen':>12} | {'Longueur moy':>13} | {'Temps/coup':>12}")
    print("-" * 45)
    print(f"{metrics['score_moyen']:>12.4f} | {metrics['longueur_moy']:>13.2f} | {metrics['temps_coup_ms']:>11.4f}ms")
    print()


# ══ ALPHAZERO / MUZERO ════════════════════════════════════════

def train_alphazero_1player(env, agent, num_episodes, checkpoints, evaluate_fn, window=100, max_steps=200):
    all_policy_losses, all_critic_losses, all_rewards, eval_results = [], [], [], {}
    checkpoints    = sorted(checkpoints)
    next_check_idx = 0
    for episode in range(1, num_episodes + 1):
        env.reset()
        episode_reward = 0.0
        steps          = 0
        while not env.done and steps < max_steps:
            available        = env.get_actions()
            action           = agent.select_action_mcts(env, available)
            _, reward, _     = env.step(action)
            episode_reward  += reward
            steps           += 1
        agent.store_reward(episode_reward)
        policy_loss, critic_loss = _parse_loss(agent.learn())
        all_policy_losses.append(policy_loss)
        if critic_loss is not None:
            all_critic_losses.append(critic_loss)
        all_rewards.append(episode_reward)
        if next_check_idx < len(checkpoints) and episode == checkpoints[next_check_idx]:
            metrics = evaluate_fn(env, agent)
            eval_results[episode] = metrics
            print(f"  {episode:>9,} épisodes | Score : {metrics['score_moyen']:.4f} | "
                  f"Longueur : {metrics['longueur_moy']:.2f} | Temps/coup : {metrics['temps_coup_ms']:.4f} ms")
            next_check_idx += 1
    return all_rewards, all_policy_losses, all_critic_losses, eval_results


def train_alphazero_tictactoe(env, agent, num_episodes, checkpoints, evaluate_fn, window=100):
    all_policy_losses, all_critic_losses, all_rewards, eval_results = [], [], [], {}
    checkpoints    = sorted(checkpoints)
    next_check_idx = 0
    for episode in range(1, num_episodes + 1):
        env.reset()
        episode_reward = 0.0
        while not env.done:
            available = env.get_actions()
            action    = agent.select_action_mcts(env, available)
            _, _, done, info = env.step(action)
            if done:
                episode_reward += 1 if info["winner"] == 1 else 0
                break
            _, _, done, info = env.step(np.random.choice(env.get_actions()))
            if done:
                episode_reward += -1 if info["winner"] == -1 else 0
                break
        agent.store_reward(episode_reward)
        policy_loss, critic_loss = _parse_loss(agent.learn())
        all_policy_losses.append(policy_loss)
        if critic_loss is not None:
            all_critic_losses.append(critic_loss)
        all_rewards.append(episode_reward)
        if next_check_idx < len(checkpoints) and episode == checkpoints[next_check_idx]:
            metrics = evaluate_fn(env, agent)
            eval_results[episode] = metrics
            print(f"  {episode:>9,} épisodes | Score : {metrics['score_moyen']:.4f} | "
                  f"Longueur : {metrics['longueur_moy']:.2f} | Temps/coup : {metrics['temps_coup_ms']:.4f} ms")
            next_check_idx += 1
    return all_rewards, all_policy_losses, all_critic_losses, eval_results


def train_alphazero_quarto(env, agent, num_episodes, checkpoints, evaluate_fn, window=100):
    all_policy_losses, all_critic_losses, all_rewards, eval_results = [], [], [], {}
    checkpoints    = sorted(checkpoints)
    next_check_idx = 0
    for episode in range(1, num_episodes + 1):
        env.reset()
        episode_reward = 0.0
        while not env.done:
            available = env.get_actions()
            if env.current_player == 1:
                action = agent.select_action_mcts(env, available)
            else:
                action = int(np.random.choice(available))
            _, _, done, _ = env.step(action)
            if done:
                episode_reward = 1 if env.winner == 1 else (-1 if env.winner == 2 else 0)
                break
        agent.store_reward(episode_reward)
        policy_loss, critic_loss = _parse_loss(agent.learn())
        all_policy_losses.append(policy_loss)
        if critic_loss is not None:
            all_critic_losses.append(critic_loss)
        all_rewards.append(episode_reward)
        if next_check_idx < len(checkpoints) and episode == checkpoints[next_check_idx]:
            metrics = evaluate_fn(env, agent)
            eval_results[episode] = metrics
            print(f"  {episode:>9,} épisodes | Score : {metrics['score_moyen']:.4f} | "
                  f"Longueur : {metrics['longueur_moy']:.2f} | Temps/coup : {metrics['temps_coup_ms']:.4f} ms")
            next_check_idx += 1
    return all_rewards, all_policy_losses, all_critic_losses, eval_results


# ══ DQN / DOUBLE DQN ══════════════════════════════════════════

def evaluate_dqn_1player(env, agent, n_games=500, max_steps=200):
    """CORRIGÉ : max_steps pour éviter boucle infinie sur GridWorld."""
    saved_eps     = agent.epsilon
    agent.epsilon = 0.0
    scores, lengths, times = [], [], []
    for _ in range(n_games):
        env.reset()
        state = env.encode_state()
        steps = 0
        while not env.done and steps < max_steps:  # ← CORRIGÉ
            available = env.get_actions()
            t0        = time.perf_counter()
            state_t   = torch.FloatTensor(state).unsqueeze(0).to(agent.device)
            q_values  = agent.q_network(state_t).squeeze(0)
            mask      = torch.full((agent.num_actions,), float('-inf'))
            mask[available] = 0.0
            action    = int((q_values + mask).argmax().item())
            t1        = time.perf_counter()
            times.append((t1 - t0) * 1000)
            _, reward, _ = env.step(action)
            state        = env.encode_state()
            steps       += 1
        scores.append(reward)
        lengths.append(steps)
    agent.epsilon = saved_eps
    return {
        "score_moyen":   round(float(np.mean(scores)),  4),
        "longueur_moy":  round(float(np.mean(lengths)), 2),
        "temps_coup_ms": round(float(np.mean(times)),   4),
    }


def evaluate_dqn_tictactoe(env, agent, n_games=500):
    saved_eps     = agent.epsilon
    agent.epsilon = 0.0
    scores, lengths, times = [], [], []
    for _ in range(n_games):
        env.reset()
        steps = 0
        while not env.done:
            state     = env.encode_state()
            available = env.get_actions()
            t0        = time.perf_counter()
            state_t   = torch.FloatTensor(state).unsqueeze(0).to(agent.device)
            q_values  = agent.q_network(state_t).squeeze(0)
            mask      = torch.full((agent.num_actions,), float('-inf'))
            mask[available] = 0.0
            action    = int((q_values + mask).argmax().item())
            t1        = time.perf_counter()
            times.append((t1 - t0) * 1000)
            _, _, done, info = env.step(action)
            steps += 1
            if done:
                scores.append(1 if info["winner"] == 1 else 0)
                lengths.append(steps)
                break
            _, _, done, info = env.step(np.random.choice(env.get_actions()))
            steps += 1
            if done:
                scores.append(-1 if info["winner"] == -1 else 0)
                lengths.append(steps)
                break
    agent.epsilon = saved_eps
    return {
        "score_moyen":   round(float(np.mean(scores)),  4),
        "longueur_moy":  round(float(np.mean(lengths)), 2),
        "temps_coup_ms": round(float(np.mean(times)),   4),
    }


def evaluate_dqn_quarto(env, agent, n_games=500):
    saved_eps     = agent.epsilon
    agent.epsilon = 0.0
    scores, lengths, times = [], [], []
    for _ in range(n_games):
        env.reset()
        steps = 0
        while not env.done:
            available = env.get_actions()
            if env.current_player == 1:
                state   = env.encode_state()
                t0      = time.perf_counter()
                state_t = torch.FloatTensor(state).unsqueeze(0).to(agent.device)
                q_values = agent.q_network(state_t).squeeze(0)
                mask    = torch.full((agent.num_actions,), float('-inf'))
                mask[available] = 0.0
                action  = int((q_values + mask).argmax().item())
                t1      = time.perf_counter()
                times.append((t1 - t0) * 1000)
            else:
                action = int(np.random.choice(available))
            _, _, done, _ = env.step(action)
            steps += 1
            if done:
                scores.append(1 if env.winner == 1 else (-1 if env.winner == 2 else 0))
                lengths.append(steps)
                break
    agent.epsilon = saved_eps
    return {
        "score_moyen":   round(float(np.mean(scores)),  4),
        "longueur_moy":  round(float(np.mean(lengths)), 2),
        "temps_coup_ms": round(float(np.mean(times)),   4),
    }


def train_dqn_1player(env, agent, num_episodes, checkpoints, evaluate_fn, max_steps=200):
    all_losses, all_rewards, eval_results = [], [], {}
    checkpoints    = sorted(checkpoints)
    next_check_idx = 0
    for episode in range(1, num_episodes + 1):
        env.reset()
        state          = env.encode_state()
        episode_reward = 0.0
        episode_loss   = 0.0
        steps          = 0
        while not env.done and steps < max_steps:
            available        = env.get_actions()
            action           = agent.select_action(state, available)
            _, reward, done  = env.step(action)
            next_state       = env.encode_state()
            next_available   = env.get_actions() if not done else [0]
            s  = torch.FloatTensor(state).unsqueeze(0).to(agent.device)
            s_ = torch.FloatTensor(next_state).unsqueeze(0).to(agent.device)
            q_sa = agent.q_network(s)[0, action]
            with torch.no_grad():
                q_next = agent.q_network(s_).squeeze(0)
                mask   = torch.full((agent.num_actions,), float('-inf'))
                mask[next_available] = 0.0
                q_next = q_next + mask
            target = torch.tensor(
                reward if done else reward + agent.gamma * q_next.max().item(),
                dtype=torch.float32
            ).to(agent.device)
            loss = agent.loss_fn(q_sa, target)
            agent.optimizer.zero_grad()
            loss.backward()
            agent.optimizer.step()
            if hasattr(agent, 'steps_done'):
                agent.steps_done += 1
                if agent.steps_done % agent.target_update_every == 0:
                    agent.target_network.load_state_dict(agent.q_network.state_dict())
            episode_loss   += loss.item()
            episode_reward += reward
            state           = next_state
            steps          += 1
        agent.decay_epsilon()
        all_losses.append(episode_loss / max(steps, 1))
        all_rewards.append(episode_reward)
        if next_check_idx < len(checkpoints) and episode == checkpoints[next_check_idx]:
            metrics = evaluate_fn(env, agent)
            eval_results[episode] = metrics
            print(f"  {episode:>9,} épisodes | Score : {metrics['score_moyen']:.4f} | "
                  f"Longueur : {metrics['longueur_moy']:.2f} | ε : {agent.epsilon:.3f}")
            next_check_idx += 1
    return all_rewards, all_losses, [], eval_results


def train_dqn_tictactoe(env, agent, num_episodes, checkpoints, evaluate_fn):
    all_losses, all_rewards, eval_results = [], [], {}
    checkpoints    = sorted(checkpoints)
    next_check_idx = 0
    for episode in range(1, num_episodes + 1):
        env.reset()
        episode_reward = 0.0
        episode_loss   = 0.0
        steps          = 0
        while not env.done:
            state     = env.encode_state()
            available = env.get_actions()
            action    = agent.select_action(state, available)
            _, _, done, info = env.step(action)
            next_state = env.encode_state()
            if done:
                reward = 1 if info["winner"] == 1 else 0
                next_available = [0]
            else:
                reward = 0
                _, _, done2, info2 = env.step(np.random.choice(env.get_actions()))
                if done2:
                    reward = -1 if info2["winner"] == -1 else 0
                    done   = True
                    next_available = [0]
                else:
                    next_available = env.get_actions()
                next_state = env.encode_state()
            s  = torch.FloatTensor(state).unsqueeze(0).to(agent.device)
            s_ = torch.FloatTensor(next_state).unsqueeze(0).to(agent.device)
            q_sa = agent.q_network(s)[0, action]
            with torch.no_grad():
                q_next = agent.q_network(s_).squeeze(0)
                mask   = torch.full((agent.num_actions,), float('-inf'))
                mask[next_available] = 0.0
                q_next = q_next + mask
            target = torch.tensor(
                reward if done else reward + agent.gamma * q_next.max().item(),
                dtype=torch.float32
            ).to(agent.device)
            loss = agent.loss_fn(q_sa, target)
            agent.optimizer.zero_grad()
            loss.backward()
            agent.optimizer.step()
            if hasattr(agent, 'steps_done'):
                agent.steps_done += 1
                if agent.steps_done % agent.target_update_every == 0:
                    agent.target_network.load_state_dict(agent.q_network.state_dict())
            episode_loss   += loss.item()
            episode_reward += reward
            steps          += 1
        agent.decay_epsilon()
        all_losses.append(episode_loss / max(steps, 1))
        all_rewards.append(episode_reward)
        if next_check_idx < len(checkpoints) and episode == checkpoints[next_check_idx]:
            metrics = evaluate_fn(env, agent)
            eval_results[episode] = metrics
            print(f"  {episode:>9,} épisodes | Score : {metrics['score_moyen']:.4f} | "
                  f"Longueur : {metrics['longueur_moy']:.2f} | ε : {agent.epsilon:.3f}")
            next_check_idx += 1
    return all_rewards, all_losses, [], eval_results


def train_dqn_quarto(env, agent, num_episodes, checkpoints, evaluate_fn):
    all_losses, all_rewards, eval_results = [], [], {}
    checkpoints    = sorted(checkpoints)
    next_check_idx = 0
    for episode in range(1, num_episodes + 1):
        env.reset()
        episode_reward = 0.0
        episode_loss   = 0.0
        steps          = 0
        while not env.done:
            available = env.get_actions()
            if env.current_player == 1:
                state  = env.encode_state()
                action = agent.select_action(state, available)
                _, _, done, _ = env.step(action)
                next_state    = env.encode_state()
                next_available = env.get_actions() if not done else [0]
                reward = 1 if (done and env.winner == 1) else (-1 if (done and env.winner == 2) else 0)
                s  = torch.FloatTensor(state).unsqueeze(0).to(agent.device)
                s_ = torch.FloatTensor(next_state).unsqueeze(0).to(agent.device)
                q_sa = agent.q_network(s)[0, action]
                with torch.no_grad():
                    q_next = agent.q_network(s_).squeeze(0)
                    mask   = torch.full((agent.num_actions,), float('-inf'))
                    mask[next_available] = 0.0
                    q_next = q_next + mask
                target = torch.tensor(
                    reward if done else reward + agent.gamma * q_next.max().item(),
                    dtype=torch.float32
                ).to(agent.device)
                loss = agent.loss_fn(q_sa, target)
                agent.optimizer.zero_grad()
                loss.backward()
                agent.optimizer.step()
                if hasattr(agent, 'steps_done'):
                    agent.steps_done += 1
                    if agent.steps_done % agent.target_update_every == 0:
                        agent.target_network.load_state_dict(agent.q_network.state_dict())
                episode_loss   += loss.item()
                episode_reward += reward
                steps          += 1
            else:
                action = int(np.random.choice(available))
                env.step(action)
        agent.decay_epsilon()
        all_losses.append(episode_loss / max(steps, 1))
        all_rewards.append(episode_reward)
        if next_check_idx < len(checkpoints) and episode == checkpoints[next_check_idx]:
            metrics = evaluate_fn(env, agent)
            eval_results[episode] = metrics
            print(f"  {episode:>9,} épisodes | Score : {metrics['score_moyen']:.4f} | "
                  f"Longueur : {metrics['longueur_moy']:.2f} | ε : {agent.epsilon:.3f}")
            next_check_idx += 1
    return all_rewards, all_losses, [], eval_results


def run_experiment_dqn(env, env_name, train_fn, evaluate_fn, agent, agent_name, num_episodes, checkpoints):
    print("=" * 65)
    print(f"  {agent_name}  —  {env_name}")
    print("=" * 65)
    all_rewards, all_losses, _, eval_results = train_fn(
        env, agent, num_episodes, checkpoints, evaluate_fn
    )
    print()
    print("=" * 65)
    print("  RÉSULTATS  (ε=0, évaluation sur 500 parties)")
    print("=" * 65)
    print(f"{'Épisodes':>12} | {'Score moyen':>12} | {'Longueur moy':>13} | {'Temps/coup':>12}")
    print("-" * 65)
    for ep, m in eval_results.items():
        print(f"{ep:>12,} | {m['score_moyen']:>12.4f} | {m['longueur_moy']:>13.2f} | {m['temps_coup_ms']:>11.4f}ms")
    smoothed = [np.mean(all_rewards[max(0,i-100):i+1]) for i in range(len(all_rewards))]
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))
    fig.suptitle(f"{agent_name} - {env_name}", fontsize=14)
    axes[0].plot(smoothed, color="steelblue")
    axes[0].set_ylabel("Reward moyen (100 épisodes)")
    axes[0].set_xlabel("Épisode")
    axes[0].grid(True, alpha=0.3)
    axes[1].plot(all_losses, color="steelblue")
    axes[1].set_ylabel("Loss (moyenne par épisode)")
    axes[1].set_xlabel("Épisode")
    axes[1].set_title("Loss par épisode")
    axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    filename = f"{agent_name}_{env_name}.png".replace(" ", "_")
    plt.savefig(filename, dpi=150)
    plt.show()
    print(f"Graphique sauvegardé → {filename}")
    agent.save(f"{agent_name}_{env_name}.pt")
    print(f"Modèle sauvegardé → {agent_name}_{env_name}.pt\n")


# ══ DDQN + ER  et  DDQN + PER ════════════════════════════════

def train_ddqn_er_1player(env, agent, num_episodes, checkpoints, evaluate_fn, max_steps=200):
    all_losses, all_rewards, eval_results = [], [], {}
    checkpoints    = sorted(checkpoints)
    next_check_idx = 0
    for episode in range(1, num_episodes + 1):
        env.reset()
        state          = env.encode_state()
        episode_reward = 0.0
        episode_loss   = 0.0
        steps          = 0
        while not env.done and steps < max_steps:
            available       = env.get_actions()
            action          = agent.select_action(state, available)
            _, reward, done = env.step(action)
            next_state      = env.encode_state()
            agent.store(state, action, reward, next_state, done)
            loss = agent.learn()
            episode_loss   += loss
            episode_reward += reward
            state           = next_state
            steps          += 1
        agent.decay_epsilon()
        all_losses.append(episode_loss / max(steps, 1))
        all_rewards.append(episode_reward)
        if next_check_idx < len(checkpoints) and episode == checkpoints[next_check_idx]:
            metrics = evaluate_fn(env, agent)
            eval_results[episode] = metrics
            print(f"  {episode:>9,} épisodes | Score : {metrics['score_moyen']:.4f} | "
                  f"Longueur : {metrics['longueur_moy']:.2f} | ε : {agent.epsilon:.3f}")
            next_check_idx += 1
    return all_rewards, all_losses, [], eval_results


def train_ddqn_er_tictactoe(env, agent, num_episodes, checkpoints, evaluate_fn):
    all_losses, all_rewards, eval_results = [], [], {}
    checkpoints    = sorted(checkpoints)
    next_check_idx = 0
    for episode in range(1, num_episodes + 1):
        env.reset()
        episode_reward = 0.0
        episode_loss   = 0.0
        steps          = 0
        while not env.done:
            state     = env.encode_state()
            available = env.get_actions()
            action    = agent.select_action(state, available)
            _, _, done, info = env.step(action)
            next_state = env.encode_state()
            if done:
                reward = 1 if info["winner"] == 1 else 0
            else:
                reward = 0
                _, _, done2, info2 = env.step(np.random.choice(env.get_actions()))
                if done2:
                    reward = -1 if info2["winner"] == -1 else 0
                    done   = True
                next_state = env.encode_state()
            agent.store(state, action, reward, next_state, done)
            loss = agent.learn()
            episode_loss   += loss
            episode_reward += reward
            steps          += 1
        agent.decay_epsilon()
        all_losses.append(episode_loss / max(steps, 1))
        all_rewards.append(episode_reward)
        if next_check_idx < len(checkpoints) and episode == checkpoints[next_check_idx]:
            metrics = evaluate_fn(env, agent)
            eval_results[episode] = metrics
            print(f"  {episode:>9,} épisodes | Score : {metrics['score_moyen']:.4f} | "
                  f"Longueur : {metrics['longueur_moy']:.2f} | ε : {agent.epsilon:.3f}")
            next_check_idx += 1
    return all_rewards, all_losses, [], eval_results


def train_ddqn_er_quarto(env, agent, num_episodes, checkpoints, evaluate_fn):
    all_losses, all_rewards, eval_results = [], [], {}
    checkpoints    = sorted(checkpoints)
    next_check_idx = 0
    for episode in range(1, num_episodes + 1):
        env.reset()
        episode_reward = 0.0
        episode_loss   = 0.0
        steps          = 0
        while not env.done:
            available = env.get_actions()
            if env.current_player == 1:
                state  = env.encode_state()
                action = agent.select_action(state, available)
                _, _, done, _ = env.step(action)
                next_state    = env.encode_state()
                reward = 1 if (done and env.winner == 1) else (-1 if (done and env.winner == 2) else 0)
                agent.store(state, action, reward, next_state, done)
                loss = agent.learn()
                episode_loss   += loss
                episode_reward += reward
                steps          += 1
            else:
                env.step(int(np.random.choice(available)))
        agent.decay_epsilon()
        all_losses.append(episode_loss / max(steps, 1))
        all_rewards.append(episode_reward)
        if next_check_idx < len(checkpoints) and episode == checkpoints[next_check_idx]:
            metrics = evaluate_fn(env, agent)
            eval_results[episode] = metrics
            print(f"  {episode:>9,} épisodes | Score : {metrics['score_moyen']:.4f} | "
                  f"Longueur : {metrics['longueur_moy']:.2f} | ε : {agent.epsilon:.3f}")
            next_check_idx += 1
    return all_rewards, all_losses, [], eval_results