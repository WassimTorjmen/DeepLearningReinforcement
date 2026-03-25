import os
import sys

# On remonte au dossier racine du projet
racine_projet = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(racine_projet)

from Environnements.quarto import QuartoEnv
from Encodings.quarto_encoding import encoder_etat_quarto, encoder_action_quarto


def test_encodage_quarto():
    # Création de l'environnement
    env = QuartoEnv()

    # Réinitialisation du jeu
    env.reset()

    print("=== TEST ENCODAGE QUARTO ===")
    print()

    # -----------------------------
    # 1. Test de l'encodage de l'état initial
    # -----------------------------
    vecteur_etat = encoder_etat_quarto(env)

    print("Etat initial du jeu :")
    print(env.get_state())
    print()

    print("Vecteur d'état :")
    print(vecteur_etat)
    print()

    print("Taille du vecteur d'état :", len(vecteur_etat))
    print("Taille attendue : 116")
    print()

    # -----------------------------
    # 2. Test de l'encodage d'une action possible
    # -----------------------------
    actions_possibles = env.get_actions()

    print("Phase actuelle :", env.phase)
    print("Actions possibles :", actions_possibles)
    print()

    action = actions_possibles[0]

    vecteur_action = encoder_action_quarto(action, env.phase)

    print("Action choisie pour le test :", action)
    print("Vecteur d'action :")
    print(vecteur_action)
    print()

    print("Taille du vecteur d'action :", len(vecteur_action))
    print("Taille attendue : 32")
    print()

    # -----------------------------
    # 3. Appliquer l'action puis retester
    # -----------------------------
    env.step(action)

    print("Etat après une action :")
    print(env.get_state())
    print()

    nouveau_vecteur_etat = encoder_etat_quarto(env)

    print("Nouveau vecteur d'état :")
    print(nouveau_vecteur_etat)
    print()

    print("Nouvelle phase :", env.phase)
    print("Nouveau joueur courant :", env.current_player)
    print()

    nouvelles_actions = env.get_actions()
    print("Nouvelles actions possibles :", nouvelles_actions)
    print()

    nouvelle_action = nouvelles_actions[0]
    nouveau_vecteur_action = encoder_action_quarto(nouvelle_action, env.phase)

    print("Nouvelle action choisie :", nouvelle_action)
    print("Nouveau vecteur d'action :")
    print(nouveau_vecteur_action)
    print()

    print("=== FIN DU TEST ===")


if __name__ == "__main__":
    test_encodage_quarto()