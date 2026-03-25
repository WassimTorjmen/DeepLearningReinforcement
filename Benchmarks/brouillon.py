class QuartoEnv:
    def __init__(self):
        # on crée un plateua de 4lignes * 4 colonnes 
        self.board = [[None for _ in range(4)] for _ in range(4)]
        
        # Liste des pièces disponibles : 0 à 15
        self.available_pieces = list(range(16))
        
        # Pièce que le joueur courant doit poser
        self.piece_to_play = None
        
        # Joueur courant : 1 ou 2
        self.current_player = 1
        
        # Etat de la partie
        self.done = False # la partie n'es pas encore terminée 
        self.winner = None # il n y a pas encore de gagnant 
        
        # Phase du tour :
        # la partie commence par choisir une pièce pour l'adversaire 
        self.phase = "choose"


    # reset permet de recommencer une nouvelle partie 
    def reset(self):
        # réinitialisation
        self.board = [[None for _ in range(4)] for _ in range(4)]
        self.available_pieces = list(range(16))
        self.piece_to_play = None
        self.current_player = 1
        self.done = False
        self.winner = None
        self.phase = "choose"
        # à la fin 
        return self.get_state()
    # cette fonciton retourne une représentation de l'état de jeu 
    def get_state(self):
        return {
            "board": self.board,
            "available_pieces": self.available_pieces,
            "piece_to_play": self.piece_to_play,
            "current_player": self.current_player,
            "phase": self.phase,
            "done": self.done,
            "winner": self.winner
        }

    # les actions autorisées dans le quarto 
    # on a deux phases : soit on choisit une pièce pour l'adversaire, soit on pose la pièce 
    def get_actions(self):
        # si la partie est fini , il n ya plus d'action possibles 
        if self.done:
            return []
        
        # Si on est dans la phase "place",
        # les actions possibles = toutes les cases vides
        if self.phase == "place":
            actions = []
            for i in range(4):
                for j in range(4):
                    if self.board[i][j] is None:
                        actions.append((i, j))
            return actions
        
        # Si on est dans la phase "choose",
        # les actions possibles = toutes les pièces restantes
        elif self.phase == "choose":
            return self.available_pieces.copy()


    # on applique une action au jeu 
    def step(self, action):
        # si la partie est finie on retourne l'état actuel 
        if self.done:
            return self.get_state(), 0, True
        
        # Phase 1 : poser la pièce reçue
        # si je suis dans la phase place donc l'action c'est une case 
        if self.phase == "place":
            row, col = action
            
            # on doit ensuite vérifier que cette case est vide 
            if self.board[row][col] is not None:
                raise ValueError("Cette case est déjà occupée.")
            
            # Ensuite on pose la pièce 
            self.board[row][col] = self.piece_to_play
            
            # On vérifie si ça crée un Quarto
            if self.check_quarto():
                self.done = True
                self.winner = self.current_player
                return self.get_state(), 1, True
            
            # Si le plateau est plein et pas de gagnant = match nul
            if self.is_board_full():
                self.done = True
                self.winner = None
                return self.get_state(), 0, True
            
            # Sinon on passe à la phase "choose"
            self.phase = "choose"
            return self.get_state(), 0, False
        
        # Phase 2 : choisir une pièce pour l'adversaire
        elif self.phase == "choose":
            # l'action est une pièce 
            piece = action
            # on vérifie que la pièce ets dispo 
            if piece not in self.available_pieces:
                raise ValueError("Cette pièce n'est pas disponible.")
            
            # On retire  la pièce choisie des pièces disponibles
            self.available_pieces.remove(piece)
            
            # Cette pièce sera jouée par l'adversaire
            self.piece_to_play = piece
            
            # On change de joueur
            if self.current_player == 1:
                self.current_player = 2
            else:
                self.current_player = 1
            
            # Prochaine phase : l'adversaire doit placer la pièce
            self.phase = "place"
            return self.get_state(), 0, False

    # on vérifie si le plateau est plein 
    def is_board_full(self):
        for i in range(4):
            for j in range(4):
                if self.board[i][j] is None:
                    return False
        return True

    def check_quarto(self):
        # on met dan sla liste les 4 lignes , les 4 colonnes et les 2 diagonales 
        lines = []

        # Ajouter les 4 lignes
        for i in range(4):
            lines.append(self.board[i])

        # Ajouter les 4 colonnes
        for j in range(4):
            column = []
            for i in range(4):
                column.append(self.board[i][j])
            lines.append(column)

        # Ajouter les 2 diagonales
        diag1 = []
        diag2 = []
        for i in range(4):
            diag1.append(self.board[i][i])
            diag2.append(self.board[i][3 - i])
        lines.append(diag1)
        lines.append(diag2)

        # Vérifier chaque ligne/colonne/diagonale
        for line in lines:
            if self.line_has_quarto(line):
                return True
        
        return False

    def line_has_quarto(self, line):
        # Si une case est vide, ce n'est pas un Quarto
        if None in line:
            return False
        
        # bit 1 : taille → 0 = petite, 1 = grande
        # bit 2 : couleur → 0 = claire, 1 = foncée
        # bit 3 : forme → 0 = ronde, 1 = carrée
        # bit 4 : remplissage → 0 = creuse, 1 = pleine

        #0  = 0000
        #1  = 0001  0 c'est l première caractéristique  1 c'est la dernière caractéristique 
        #2  = 0010
        #3  = 0011
        #4  = 0100
        #5  = 0101
        #6  = 0110
        #7  = 0111
        #8  = 1000
        #9  = 1001
        #10 = 1010
        #11 = 1011
        #12 = 1100
        #13 = 1101
        #14 = 1110
        #15 = 1111

        # On teste les 4 caractéristiques (4 bits)
        # chaque bit correspond à une caractéristqiue 
        for bit in range(4):
            values = []
            # line soir une ligne soit une colonne soit une diagonale , quand on appele cette fonction en haut c'est ce qu'on lui donne comme argument 
            for piece in line:
                # Extraire le bit numéro "bit"
                value = (piece >> bit) & 1
                values.append(value)
            
            # Si les 4 pièces ont la même valeur sur ce bit
            if values[0] == values[1] == values[2] == values[3]:
                return True
        
        return False

    def render(self):
        print(f"Joueur courant : {self.current_player}")
        print(f"Phase : {self.phase}")
        print(f"Pièce à jouer : {self.piece_to_play}")
        print("Pièces disponibles :", self.available_pieces)
        print()

        for i in range(4):
            row_display = []
            for j in range(4):
                if self.board[i][j] is None:
                    row_display.append("[    ]")
                else:
                    # Affichage binaire sur 4 bits
                    row_display.append(f"[{self.board[i][j]:04b}]")
            print(" ".join(row_display))
        
        print()
        if self.done:
            if self.winner is not None:
                print(f"Partie terminée. Gagnant : joueur {self.winner}")
            else:
                print("Partie terminée. Match nul.")






"""
Par ex pour quarto :
en entré
le plateau
les pièce restante
la pièce sélectionné

Et en sortie:
la position de la pièce sélectionné pour le joueur si c’est la phase sélection
la position à déposer si c’est la phase dépôt de pièce
"""




from time import perf_counter
from deeprl_5iabd.config import settings
from deeprl_5iabd.envs.line_world import LineWorld
from deeprl_5iabd.envs.grid_world import GridWorld
from deeprl_5iabd.envs.tictactoe import TicTacToe
from deeprl_5iabd.envs.quarto import QuartoEnv
from deeprl_5iabd.envs.base_env import BaseEnv
from deeprl_5iabd.agents.reinforce import reinforce
from deeprl_5iabd.agents.policy_net import PolicyNetwork
from deeprl_5iabd.agents.random_agent import RandomPlayer
from deeprl_5iabd.tracking.tb_logger import TensorBoardLogger
from torch.distributions import Categorical

def count_n_match_time(env: BaseEnv, num_episode):
    player = RandomPlayer(action_dim=len(env.get_action_space()))
    s = perf_counter()
    
    i = 0
    while (i <= num_episode):
        while (not env.is_game_over()):
            action_spaces = env.get_action_space()
            probs = player.forward(x=None, mask=action_spaces)
            probs_dist = Categorical(probs)
            action_pos = probs_dist.sample()
            env.step(action_pos.item())
        i += 1
    e = perf_counter()

    total_time = e - s
    match_per_s = num_episode / total_time
    return total_time, match_per_s

if __name__ == "__main__":

    envs = [LineWorld(), GridWorld(), TicTacToe(), QuartoEnv()]
    total_match = 1_000_000

    print(f"\n\nTEST SUR {total_match} PARTIES ...")
    print("-" * 44)
    print(f"{'Environnement':<20} {'Durée (s)':>10} {'Matchs/sec':>12}")
    print("-" * 44)
    for env in envs:
        total_duration, match_per_s = count_n_match_time(env, total_match)
        print(f"{env.env_name:<20} {round(total_duration, 2):>10} {match_per_s:>12.0f}")






import torch
from deeprl_5iabd.agents.base_agent import BaseAgent
from deeprl_5iabd.helper import softmax_with_mask

class RandomPlayer(BaseAgent):
    def init(self, action_dim):
        self.name = "RandomPlayer"
        self.action_dim = action_dim

    def forward(self, x, mask):
        logits = torch.randn(self.action_dim)
        return softmax_with_mask(logits, mask)

    def save(self, filename):
        print("Random player is not a model, it is a random policy")

    def clone(self, name=None):
        return RandomPlayer(action_dim=self.action_dim)

    @classmethod
    def load(cls, filename):
        print("Random player is not a model, it is a random policy")

if name == "main":
    rp = RandomPlayer(action_dim=5)
    print(rp.forward(x=None, mask=[0,1,1,0,0]))
import torch
import pygame
import torch.nn.functional as F
import numpy as np

class ImageButton:
    def init(self, x, y, width, height):
        self.rect = pygame.Rect(x, y, width, height)
        self.image = None

    def draw(self, screen):
        if self.image:
            screen.blit(self.image, self.rect)
        else:
            pygame.draw.rect(screen, (200, 200, 200), self.rect)

    def is_clicked(self, event):
        return event.type == pygame.MOUSEBUTTONDOWN and self.rect.collidepoint(event.pos)

def get_default_device() -> str:
    if hasattr(torch, "accelerator") and torch.accelerator.is_available():
        return torch.accelerator.current_accelerator().type
    return "cpu"

def softmax_with_mask(S, M):
    M = torch.tensor(M, dtype=S.dtype, device=S.device)
    positive_or_null_s = S - S.min()
    masked_positive_or_null_s = positive_or_null_s * M
    negative_or_null_s = masked_positive_or_null_s - masked_positive_or_null_s.max()
    exp_s = torch.exp(negative_or_null_s)
    masked_exp_s = exp_s * M
    return masked_exp_s / masked_exp_s.sum()






rom time import perf_counter
from deeprl_5iabd.config import settings
from deeprl_5iabd.envs.line_world import LineWorld
from deeprl_5iabd.envs.grid_world import GridWorld
from deeprl_5iabd.envs.tictactoe import TicTacToe
from deeprl_5iabd.envs.quarto import QuartoEnv
from deeprl_5iabd.envs.base_env import BaseEnv
from deeprl_5iabd.agents.reinforce import reinforce
from deeprl_5iabd.agents.policy_net import PolicyNetwork
from deeprl_5iabd.agents.random_agent import RandomPlayer
from deeprl_5iabd.tracking.tb_logger import TensorBoardLogger
from torch.distributions import Categorical

def count_n_match_time(env: BaseEnv, num_episode):
    player = RandomPlayer(action_dim=len(env.get_action_space()))
    s = perf_counter()

    i = 0
    while (i <= num_episode):
        while (not env.is_game_over()):
            action_spaces = env.get_action_space()
            probs = player.forward(x=None, mask=action_spaces)
            probs_dist = Categorical(probs)
            action_pos = probs_dist.sample()
            env.step(action_pos.item())
        i += 1
    e = perf_counter()

    total_time = e - s
    match_per_s = num_episode / total_time
    return total_time, match_per_s

if name == "main":

    envs = [LineWorld(), GridWorld(), TicTacToe(), QuartoEnv()]
    total_match = 1_000_000

    print(f"\n\nTEST SUR {total_match} PARTIES ...")
    print("-" * 44)
    print(f"{'Environnement':<20} {'Durée (s)':>10} {'Matchs/sec':>12}")
    print("-" * 44)
    for env in envs:
        total_duration, match_per_s = count_n_match_time(env, total_match)
        print(f"{env.env_name:<20} {round(total_duration, 2):>10} {match_per_s:>12.0f}")





import torch
from deeprl_5iabd.agents.base_agent import BaseAgent
from deeprl_5iabd.helper import softmax_with_mask

class RandomPlayer(BaseAgent):
    def init(self, action_dim):
        self.name = "RandomPlayer"
        self.action_dim = action_dim

    def forward(self, x, mask):
        logits = torch.randn(self.action_dim)
        return softmax_with_mask(logits, mask)

    def save(self, filename):
        print("Random player is not a model, it is a random policy")

    def clone(self, name=None):
        return RandomPlayer(action_dim=self.action_dim)

    @classmethod
    def load(cls, filename):
        print("Random player is not a model, it is a random policy")

if name == "main":
    rp = RandomPlayer(action_dim=5)
    print(rp.forward(x=None, mask=[0,1,1,0,0]))
import torch
import pygame
import torch.nn.functional as F
import numpy as np

class ImageButton:
    def init(self, x, y, width, height):
        self.rect = pygame.Rect(x, y, width, height)
        self.image = None

    def draw(self, screen):
        if self.image:
            screen.blit(self.image, self.rect)
        else:
            pygame.draw.rect(screen, (200, 200, 200), self.rect)

    def is_clicked(self, event):
        return event.type == pygame.MOUSEBUTTONDOWN and self.rect.collidepoint(event.pos)

def get_default_device() -> str:
    if hasattr(torch, "accelerator") and torch.accelerator.is_available():
        return torch.accelerator.current_accelerator().type
    return "cpu"

def softmax_with_mask(S, M):
    M = torch.tensor(M, dtype=S.dtype, device=S.device)
    positive_or_null_s = S - S.min()
    masked_positive_or_null_s = positive_or_null_s * M
    negative_or_null_s = masked_positive_or_null_s - masked_positive_or_null_s.max()
    exp_s = torch.exp(negative_or_null_s)
    masked_exp_s = exp_s * M
    return masked_exp_s / masked_exp_s.sum()