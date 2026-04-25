"""
Pipeline générique d'expérimentation pour tous les agents Policy/Value.

Convention :
    - evaluate_*           : éval avec accès direct à agent.policy (REINFORCE, PPO, ...)
    - evaluate_no_training : éval via agent.select_action(env, ...) (Random Rollout, AlphaZero, MuZero)
    - train_*              : entraînement classique (le réseau choisit l'action)
    - train_alphazero_*    : entraînement où l'agent appelle agent.select_action_mcts(...)

run_experiment() orchestre train + eval + plot + save.
"""

import numpy as np
import time
import torch
import matplotlib.pyplot as plt


# ════════════════════════════════════════════════════════════════════
#  ÉVALUATIONS — agents avec attribut .policy (REINFORCE, PPO, ...)
# ════════════════════════════════════════════════════════════════════

def evaluate_1player(env, agent, n_games=500, max_steps=200):
    """Évalue un agent sur un env 1 joueur (LineWorld/GridWorld) avec limite de steps."""
    scores, lengths, times = [], [], []

    for _ in range(n_games):
        env.reset()
        state = env.encode_state()
        steps = 0

        while not env.done and steps < max_steps:  # AJOUT max_steps
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
    """TicTacToe : l'agent (X) joue, puis l'adversaire random (O), jusqu'à la fin."""
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
    """Quarto : l'agent joue J1, un random joue J2."""
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
    """Normalise le retour de agent.learn() : peut être un float ou un tuple (policy, critic)."""
    if isinstance(loss_result, tuple):
        return loss_result[0], loss_result[1]
    return loss_result, None


# ════════════════════════════════════════════════════════════════════
#  ENTRAÎNEMENTS — boucles classiques (1 player / TicTacToe / Quarto)
# ════════════════════════════════════════════════════════════════════

def train_1player(env, agent, num_episodes, checkpoints, evaluate_fn, window=100, max_steps=200):
    """Boucle d'entraînement pour LineWorld / GridWorld."""
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
    """TicTacToe : l'agent joue X, alterne avec un random qui joue O."""
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
    """Quarto : l'agent joue J1, un random joue J2 ; on stocke les rewards seulement aux tours de l'agent."""
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
    """Trace 2 ou 3 sous-graphes : reward lissée, policy loss et (optionnel) critic loss."""
    smoothed = [
        np.mean(all_rewards[max(0, i - window):i + 1])
        for i in range(len(all_rewards))
    ]
    n_plots = 3 if all_critic_losses else 2
    fig, axes = plt.subplots(n_plots, 1, figsize=(10, 4 * n_plots))
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
    """Pipeline complet : train → eval aux checkpoints → graphique → sauvegarde du modèle."""
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


# ════════════════════════════════════════════════════════════════════
#  ÉVALUATIONS — agents sans .policy (RandomRollout, AlphaZero, MuZero)
#  → on appelle agent.select_action(env, available)
# ════════════════════════════════════════════════════════════════════

def evaluate_no_training_1player(env, agent, n_games=500):
    """Éval pour LineWorld/GridWorld via select_action(env, available)."""
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
    """Variante de run_experiment pour les agents sans phase d'entraînement (RandomRollout)."""
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


# ════════════════════════════════════════════════════════════════════
#  ENTRAÎNEMENTS AlphaZero / MuZero (utilisent select_action_mcts)
# ════════════════════════════════════════════════════════════════════

def train_alphazero_1player(env, agent, num_episodes, checkpoints, evaluate_fn, window=100, max_steps=200):
    """Boucle AlphaZero/MuZero pour LineWorld / GridWorld."""
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
    """Variante TicTacToe : agent (X) joue par MCTS, random joue O."""
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
    """Variante Quarto : agent (J1) joue par MCTS, random joue J2."""
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