"""
Benchmark de performance pour l'environnement Quarto.

Fait jouer un grand nombre de parties (par défaut 100 000) entre deux
RandomAgents pour mesurer :
  - Le débit de l'environnement (parties par seconde)
  - La distribution statistique des résultats (victoires J1, J2, nuls)

Cela sert à :
  1. Vérifier que l'environnement Quarto fonctionne correctement sur de
     longues séries de parties sans erreur ni blocage.
  2. Établir une baseline statistique : contre un adversaire aléatoire,
     les taux de victoire doivent être à peu près symétriques.
  3. Mesurer la performance brute de l'env pour s'assurer que le training
     RL ne sera pas limité par la vitesse de simulation.

Usage :
    python Benchmarks/benchmark_quarto.py
"""

import os
import sys
import time
from Environnements.quarto import QuartoEnv
from Agents.random_agent import RandomAgent


racine_projet = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(racine_projet)


def jouer_une_partie(env, agent_1, agent_2):
    """
    Joue une partie complète de Quarto entre deux agents.

    L'environnement est réinitialisé puis les agents jouent à tour de rôle
    (joueur 1 puis joueur 2) jusqu'à ce que la partie se termine
    (victoire ou match nul quand le plateau est plein).

    Note : l'objet env est réutilisé entre les parties (reset à chaque appel)
    pour éviter de réallouer de la mémoire à chaque partie.

    Paramètres :
        env     : instance de QuartoEnv (sera reset en interne)
        agent_1 : agent qui joue en tant que joueur 1
        agent_2 : agent qui joue en tant que joueur 2

    Retourne :
        int — le gagnant : 1 (joueur 1), 2 (joueur 2), ou 0 (match nul)
    """
    env.reset()

    # Boucle de jeu : alterne entre les deux agents selon current_player
    while not env.done:
        if env.current_player == 1:
            action = agent_1.Choisir_action(env)
        else:
            action = agent_2.Choisir_action(env)

        # step() gère automatiquement les phases CHOOSE/PLACE et le
        # changement de joueur — l'agent n'a qu'à fournir l'action
        env.step(action)

    return int(env.winner)


def lancer_benchmark(nombre_parties=100000):
    """
    Lance le benchmark : joue `nombre_parties` parties et affiche les stats.

    Utilise time.perf_counter() (horloge haute résolution) pour un chronométrage
    précis, indépendant du wall-clock et des interruptions système.

    Paramètres :
        nombre_parties : nombre de parties à simuler (défaut 100 000)
    """
    # Un seul environnement et deux agents instanciés une fois,
    # réutilisés pour toutes les parties (l'env est reset dans jouer_une_partie)
    env = QuartoEnv()
    agent_1 = RandomAgent()
    agent_2 = RandomAgent()

    # Chronomètre haute résolution
    temps_debut = time.perf_counter()

    # Compteurs de résultats
    victoires_j1 = 0
    victoires_j2 = 0
    matchs_nuls = 0

    for _ in range(nombre_parties):
        gagnant = jouer_une_partie(env, agent_1, agent_2)

        if gagnant == 1:
            victoires_j1 += 1
        elif gagnant == 2:
            victoires_j2 += 1
        else:
            matchs_nuls += 1      # gagnant == 0

    temps_fin = time.perf_counter()

    # Calcul des métriques de performance
    temps_total = temps_fin - temps_debut
    parties_par_seconde = nombre_parties / temps_total

    # Affichage du rapport
    print("===== BENCHMARK QUARTO =====")
    print(f"Nombre de parties : {nombre_parties}")
    print(f"Temps total : {temps_total:.4f} secondes")
    print(f"Parties par seconde : {parties_par_seconde:.2f}")
    print(f"Victoires joueur 1 : {victoires_j1}")
    print(f"Victoires joueur 2 : {victoires_j2}")
    print(f"Matchs nuls : {matchs_nuls}")


# Point d'entrée : lance le benchmark avec 100 000 parties
if __name__ == "__main__":
    lancer_benchmark(100000)