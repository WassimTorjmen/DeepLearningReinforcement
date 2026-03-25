import os
import sys
import time

racine_projet = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(racine_projet)

from Environnements.quarto import QuartoEnv
from Agents.random_agent import RandomAgent


def jouer_une_partie(env, agent_1, agent_2):
    env.reset()

    while not env.done:
        if env.current_player == 1:
            action = agent_1.Choisir_action(env)
        else:
            action = agent_2.Choisir_action(env)

        env.step(action)

    return int(env.winner)


def lancer_benchmark(nombre_parties=100000):
    env = QuartoEnv()
    agent_1 = RandomAgent()
    agent_2 = RandomAgent()

    temps_debut = time.perf_counter()

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
            matchs_nuls += 1

    temps_fin = time.perf_counter()

    temps_total = temps_fin - temps_debut
    parties_par_seconde = nombre_parties / temps_total

    print("===== BENCHMARK QUARTO =====")
    print(f"Nombre de parties : {nombre_parties}")
    print(f"Temps total : {temps_total:.4f} secondes")
    print(f"Parties par seconde : {parties_par_seconde:.2f}")
    print(f"Victoires joueur 1 : {victoires_j1}")
    print(f"Victoires joueur 2 : {victoires_j2}")
    print(f"Matchs nuls : {matchs_nuls}")


if __name__ == "__main__":
    lancer_benchmark(100000)