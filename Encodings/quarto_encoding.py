def encoder_piece(piece):
    """
    Transforme une pièce entière (de 0 à 15) en 4 bits.
    Chaque bit représente une caractéristique de la pièce.
    une pièce est un nombre entre 0 et 15
    """

    bit_0 = piece & 1
    bit_1 = (piece >> 1) & 1
    bit_2 = (piece >> 2) & 1
    bit_3 = (piece >> 3) & 1

    return [bit_0, bit_1, bit_2, bit_3]


def encoder_etat_quarto(env):
    """
    Encode l'état complet du jeu Quarto en un vecteur de taille 116.

    Composition du vecteur :
    - 80 valeurs pour le plateau
    - 16 valeurs pour les pièces disponibles
    - 16 valeurs pour la pièce à jouer
    - 2 valeurs pour la phase
    - 2 valeurs pour le joueur courant
    """

    vecteur_etat = []

    
    # 1. Encodage du plateau
    # Chaque case est codée par 5 valeurs :
    # - 1 valeur pour dire si la case est occupée
    # - 4 valeurs pour encoder la pièce
    # 16 cases * 5 = 80 valeurs
    

    # on parcourt le plateau 
    for ligne in range(4):
        for colonne in range(4):
            # on récupère le contenu de la case 
            piece = env.board[ligne][colonne]

            # Si la case est vide, on ajoute les 5 zéros 
            if piece is None:
                vecteur_etat.extend([0, 0, 0, 0, 0])

            # Si la case contient une pièce
            else:
                # on encode cette pièce on récupère les 4 bits 
                bits_piece = encoder_piece(piece)
                vecteur_etat.extend([1] + bits_piece)

    
    # 2. Encodage des pièces encore disponibles
    # Vecteur de taille 16 :
    # 1 si la pièce est disponible, 0 sinon
    
    for piece in range(16):
        if piece in env.available_pieces:
            vecteur_etat.append(1)
        else:
            vecteur_etat.append(0)

  
    # 3. Encodage de la pièce à jouer
    # One-hot vector de taille 16
    
    vecteur_piece_a_jouer = [0] * 16

    if env.piece_to_play is not None:
        vecteur_piece_a_jouer[env.piece_to_play] = 1

    vecteur_etat.extend(vecteur_piece_a_jouer)

    # -------------------------------------------------
    # 4. Encodage de la phase
    # choose -> [1, 0]
    # place  -> [0, 1]
    # -------------------------------------------------
    if env.phase == "choose":
        vecteur_etat.extend([1, 0])
    else:
        vecteur_etat.extend([0, 1])

    # -------------------------------------------------
    # 5. Encodage du joueur courant
    # joueur 1 -> [1, 0]
    # joueur 2 -> [0, 1]
    # -------------------------------------------------
    if env.current_player == 1:
        vecteur_etat.extend([1, 0])
    else:
        vecteur_etat.extend([0, 1])

    return vecteur_etat


def encoder_action_quarto(action, phase):
    """
    Encode une action de Quarto en un vecteur one-hot de taille 32.

    - indices 0 à 15   : positions du plateau (phase 'place')
    - indices 16 à 31  : choix d'une pièce (phase 'choose')
    """

    vecteur_action = [0] * 32

    # Phase 'place' : l'action est une position (ligne, colonne)
    
    if phase == "place":
        ligne, colonne = action
        index = ligne * 4 + colonne
        vecteur_action[index] = 1

    
    # Phase 'choose' : l'action est un numéro de pièce
    elif phase == "choose":
        piece = action
        vecteur_action[16 + piece] = 1

    return vecteur_action