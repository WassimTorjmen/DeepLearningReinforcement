"""
Agent baseline : choisit une action uniformément au hasard parmi les actions légales.
Sert de référence ('niveau zéro') pour comparer les autres agents.
"""

import random


class RandomAgent:
    """Agent aléatoire — random.choice sur une liste Python est optimal ici."""

    def Choisir_action(self, env):
        # Récupère les actions légales et en tire une au hasard
        actions = env.get_actions()
        return random.choice(actions)
