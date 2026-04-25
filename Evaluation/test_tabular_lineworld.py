"""
Script de test bout-en-bout : entraîne un agent tabulaire sur LineWorld
puis l'évalue, et affiche les métriques finales.
"""

import os
import sys

# Ajoute la racine du projet au PYTHONPATH
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from Environnements.line_world import LineWorld
from Training.train_tabular_lineworld import train_lineworld
from Evaluation.evaluate_agent import evaluate_agent


if __name__ == "__main__":
    # 1) Entraînement sur 5000 épisodes
    agent, rewards, lengths = train_lineworld(n_episodes=5000)

    # 2) Évaluation sur 200 parties (sans exploration)
    env = LineWorld()
    metrics = evaluate_agent(env, agent, n_episodes=200)

    print("\nRésultats d'évaluation :")
    print(metrics)
