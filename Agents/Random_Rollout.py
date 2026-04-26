import numpy as np
import copy
import time


class RandomRolloutAgent:
    """
    Random Rollout Agent.

    Pas de réseau, pas d'entraînement.
    Pour chaque action légale, on simule `n_rollouts` parties aléatoires
    et on choisit l'action qui donne le meilleur score moyen.

    Paramètres :
        n_rollouts : nombre de parties simulées par action (plus c'est grand, mieux c'est)
        depth      : profondeur max de simulation (None = jusqu'à la fin)
    """

    def __init__(self, n_rollouts=50, depth=None):
        self.n_rollouts = n_rollouts
        self.depth      = depth

    def select_action(self, env, available_actions):
        """
        Pour chaque action disponible :
            1. Simule n_rollouts parties aléatoires depuis l'état courant
            2. Calcule le score moyen
        Retourne l'action avec le meilleur score moyen.

        On utilise copy.deepcopy(env) pour ne pas modifier l'env réel.
        """
        best_action = available_actions[0]
        best_score  = float('-inf')

        for action in available_actions:
            scores = []

            for _ in range(self.n_rollouts):
                # copie de l'env pour simuler sans toucher à l'env réel
                sim_env = copy.deepcopy(env)

                # on joue l'action candidate
                sim_env.step(action)

                # puis on joue aléatoirement jusqu'à la fin
                steps = 0
                while not sim_env.done:
                    if self.depth is not None and steps >= self.depth:
                        break
                    available = sim_env.get_actions()
                    if not available:
                        break
                    random_action = int(np.random.choice(available))
                    sim_env.step(random_action)
                    steps += 1

                scores.append(self._get_score(sim_env))

            mean_score = np.mean(scores)

            if mean_score > best_score:
                best_score  = mean_score
                best_action = action

        return best_action

    def _get_score(self, env):
        """
        Retourne le score final de la simulation.
        Gère les différents types d'environnements.
        """
        # TicTacToe et Quarto ont un attribut winner
        if hasattr(env, 'winner'):
            if env.winner == 1:  return 1.0
            if env.winner == 2 or env.winner == -1: return -1.0
            return 0.0

        # LineWorld et GridWorld : on regarde la dernière récompense
        # on fait un step fictif pour récupérer le score
        if hasattr(env, 'agent_position') and env.done:
            if env.agent_position == env.goal_position:
                return 1.0
            return 0.0

        return 0.0

    def store_reward(self, reward):
        """Pas d'apprentissage — méthode vide pour compatibilité."""
        pass

    def learn(self):
        """Pas d'apprentissage — retourne 0 pour compatibilité."""
        return 0.0

    def save(self, path):
        """Pas de modèle à sauvegarder."""
        pass

    def load(self, path):
        """Pas de modèle à charger."""
        pass