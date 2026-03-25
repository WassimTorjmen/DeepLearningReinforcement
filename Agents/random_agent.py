import random


class RandomAgent:
    """Agent aléatoire — random.choice sur une liste Python est optimal ici."""

    def Choisir_action(self, env):
        actions = env.get_actions()
        return random.choice(actions)