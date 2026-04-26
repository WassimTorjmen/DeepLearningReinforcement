"""
MuZeroStochastic.py
====================
MuZero Stochastique — extension de MuZero pour les environnements stochastiques.

Différences clés avec MuZero déterministe :
  1. Le réseau Dynamics intègre un encodeur de transition stochastique (afterstate).
     Il prédit une distribution de probabilité sur les "chances" (aléa) puis
     échantillonne un vecteur latent représentant l'issue aléatoire.
  2. Un réseau AfterstatePrediction prédit policy + value depuis l'afterstate.
  3. La loss ajoute un terme KL pour régulariser la distribution stochastique
     (comme dans le papier MuZero Stochastique de Antonoglou et al., 2021).

Architecture :
  h_t  = RepresentationNetwork(obs_t)
  h̃_t  = AfterstateDynamics(h_t, a_t)         ← afterstate (après action, avant aléa)
  z_t  = ChanceEncoder(h̃_t)                    ← vecteur chance (stochastique)
  h_t1 = TransitionNetwork(h̃_t, z_t)           ← état latent suivant
  r_t, π_t, v_t = PredictionNetwork(h_t, h_t1)

Pour simplifier l'intégration dans le projet, on conserve la même interface
que MuZeroAgent (select_action, select_action_mcts, store_reward, learn, save, load).
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# ══════════════════════════════════════════════════════════════════════════════
#  Réseaux de base
# ══════════════════════════════════════════════════════════════════════════════

class RepresentationNetwork(nn.Module):
    """Encode l'observation réelle en état latent h."""
    def __init__(self, state_size, hidden_size=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_size, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, hidden_size), nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)


class AfterstateDynamicsNetwork(nn.Module):
    """
    Calcule l'afterstate h̃ à partir de (h, a).
    L'afterstate représente l'état APRÈS l'action mais AVANT l'aléa.
    """
    def __init__(self, hidden_size, num_actions):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_size + num_actions, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, hidden_size), nn.ReLU(),
        )

    def forward(self, h, a_onehot):
        return self.net(torch.cat([h, a_onehot], dim=-1))


class ChanceEncoder(nn.Module):
    """
    Encode la distribution stochastique (chance) depuis l'afterstate.
    Produit mu et log_var pour le reparameterization trick (VAE-style).
    La dimension chance_size représente l'espace des événements aléatoires.
    """
    def __init__(self, hidden_size, chance_size=8):
        super().__init__()
        self.mu_head     = nn.Linear(hidden_size, chance_size)
        self.logvar_head = nn.Linear(hidden_size, chance_size)

    def forward(self, h_after):
        mu     = self.mu_head(h_after)
        logvar = self.logvar_head(h_after).clamp(-4, 4)
        return mu, logvar

    def sample(self, mu, logvar):
        """Reparameterization trick : z = mu + eps * std."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std


class TransitionNetwork(nn.Module):
    """
    Calcule l'état latent suivant h_{t+1} à partir de (afterstate h̃, chance z).
    Prédit aussi la récompense.
    """
    def __init__(self, hidden_size, chance_size=8):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_size + chance_size, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, hidden_size), nn.ReLU(),
        )
        self.next_state_head = nn.Linear(hidden_size, hidden_size)
        self.reward_head     = nn.Linear(hidden_size, 1)

    def forward(self, h_after, z):
        x = self.net(torch.cat([h_after, z], dim=-1))
        return self.next_state_head(x), self.reward_head(x).squeeze(-1)


class PredictionNetwork(nn.Module):
    """Prédit policy + value depuis l'état latent."""
    def __init__(self, hidden_size, num_actions):
        super().__init__()
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_size, num_actions), nn.Softmax(dim=-1)
        )
        self.value_head = nn.Sequential(
            nn.Linear(hidden_size, 1), nn.Tanh()
        )

    def forward(self, h):
        return self.policy_head(h), self.value_head(h).squeeze(-1)


# ══════════════════════════════════════════════════════════════════════════════
#  Agent MuZero Stochastique
# ══════════════════════════════════════════════════════════════════════════════

class MuZeroStochasticAgent:
    """
    MuZero Stochastique.

    Interface identique à MuZeroAgent et AlphaZeroAgent pour s'intégrer
    dans les fonctions train_alphazero_* d'experiment.py.

    Paramètres :
        state_size    : taille du vecteur encode_state()
        num_actions   : nombre d'actions légales max
        hidden_size   : taille des états latents
        chance_size   : dimension de l'espace stochastique (vecteur z)
        n_simulations : profondeur de planification (lookahead)
        lr            : learning rate
        kl_weight     : poids du terme KL dans la loss totale
    """

    def __init__(self, state_size, num_actions, hidden_size=64,
                 chance_size=8, n_simulations=10, lr=1e-3, kl_weight=0.1):
        self.num_actions   = num_actions
        self.hidden_size   = hidden_size
        self.chance_size   = chance_size
        self.n_simulations = n_simulations
        self.kl_weight     = kl_weight
        self.device        = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Réseaux
        self.representation     = RepresentationNetwork(state_size, hidden_size).to(self.device)
        self.afterstate_dyn     = AfterstateDynamicsNetwork(hidden_size, num_actions).to(self.device)
        self.chance_encoder     = ChanceEncoder(hidden_size, chance_size).to(self.device)
        self.transition         = TransitionNetwork(hidden_size, chance_size).to(self.device)
        self.prediction         = PredictionNetwork(hidden_size, num_actions).to(self.device)

        # Optimiseur unique sur tous les paramètres
        self.optimizer = optim.Adam(
            list(self.representation.parameters()) +
            list(self.afterstate_dyn.parameters()) +
            list(self.chance_encoder.parameters()) +
            list(self.transition.parameters()) +
            list(self.prediction.parameters()),
            lr=lr
        )

        # Buffers d'expérience
        self.states_buffer  = []
        self.actions_buffer = []
        self.rewards_buffer = []
        self.pi_buffer      = []

    # ── Utilitaires ──────────────────────────────────────────────────────────

    def _encode_action(self, action):
        a = torch.zeros(1, self.num_actions).to(self.device)
        a[0, int(action)] = 1.0
        return a

    def _forward_one_step(self, h, action):
        """
        Passe complète pour une action :
          h → afterstate → chance → next_h + reward
        Retourne (next_h, reward, mu, logvar).
        """
        a_onehot = self._encode_action(action)
        h_after  = self.afterstate_dyn(h, a_onehot)
        mu, logvar = self.chance_encoder(h_after)
        z          = self.chance_encoder.sample(mu, logvar)
        next_h, reward = self.transition(h_after, z)
        return next_h, reward, mu, logvar

    def _plan(self, h, available_actions):
        """
        Planification à un pas avec stochasticité.
        Pour chaque action, on évalue n_simulations fois (en ré-échantillonnant z)
        puis on fait la moyenne des valeurs estimées.
        Retourne la distribution π sur les actions disponibles.
        """
        scores = np.zeros(self.num_actions)

        with torch.no_grad():
            for action in available_actions:
                vals = []
                for _ in range(self.n_simulations):
                    next_h, reward, _, _ = self._forward_one_step(h, action)
                    _, value = self.prediction(next_h)
                    vals.append(reward.item() + value.item())
                scores[action] = float(np.mean(vals))

        # Distribution proportionnelle aux valeurs (softmax pour éviter valeurs négatives)
        pi = np.zeros(self.num_actions)
        sub = scores[available_actions]
        sub = sub - sub.max()           # stabilité numérique
        exp_s = np.exp(sub)
        exp_s = exp_s / (exp_s.sum() + 1e-8)
        for i, a in enumerate(available_actions):
            pi[a] = exp_s[i]

        return pi

    # ── Interface publique ───────────────────────────────────────────────────

    def select_action(self, state_or_env, available_actions):
        """Évaluation — le réseau joue sans ré-échantillonnage."""
        if hasattr(state_or_env, 'encode_state'):
            state = state_or_env.encode_state()
        else:
            state = state_or_env

        state_t = torch.FloatTensor(np.array(state)).unsqueeze(0).to(self.device)
        with torch.no_grad():
            h         = self.representation(state_t)
            policy, _ = self.prediction(h)
        policy = policy.squeeze(0)
        mask   = torch.zeros(self.num_actions).to(self.device)
        mask[available_actions] = 1.0
        policy = policy * mask
        policy = policy / (policy.sum() + 1e-8)
        return int(policy.argmax().item())

    def select_action_mcts(self, env, available_actions):
        """Entraînement — planification stochastique."""
        state   = env.encode_state()
        state_t = torch.FloatTensor(np.array(state)).unsqueeze(0).to(self.device)

        with torch.no_grad():
            h = self.representation(state_t)

        pi = self._plan(h, available_actions)

        self.states_buffer.append(state)
        self.pi_buffer.append(pi)

        action = int(np.random.choice(self.num_actions, p=pi))
        self.actions_buffer.append(action)
        return action

    def store_reward(self, reward):
        n = len(self.states_buffer) - len(self.rewards_buffer)
        self.rewards_buffer.extend([reward] * n)

    def learn(self):
        if not self.states_buffer or len(self.states_buffer) != len(self.rewards_buffer):
            self._clear_buffers()
            return 0.0, 0.0

        states  = torch.FloatTensor(np.array(self.states_buffer)).to(self.device)
        pi_mcts = torch.FloatTensor(np.array(self.pi_buffer)).to(self.device)
        rewards = torch.FloatTensor(np.array(self.rewards_buffer)).to(self.device)
        actions = torch.LongTensor(self.actions_buffer).to(self.device)

        # ── Forward ──────────────────────────────────────────────────────────
        h                       = self.representation(states)
        policy_pred, value_pred = self.prediction(h)

        # ── Policy loss (cross-entropy vs distribution MCTS) ─────────────────
        policy_loss = -(pi_mcts * torch.log(policy_pred + 1e-8)).sum(dim=1).mean()

        # ── Value loss ────────────────────────────────────────────────────────
        value_loss = nn.MSELoss()(value_pred, rewards)

        # ── Reward loss + KL (via stochastic forward) ─────────────────────────
        a_onehot  = torch.zeros(len(actions), self.num_actions).to(self.device)
        a_onehot.scatter_(1, actions.unsqueeze(1), 1.0)

        h_after         = self.afterstate_dyn(h.detach(), a_onehot)
        mu, logvar      = self.chance_encoder(h_after)
        z               = self.chance_encoder.sample(mu, logvar)
        _, reward_pred  = self.transition(h_after, z)

        reward_loss = nn.MSELoss()(reward_pred, rewards)

        # KL divergence : KL(N(mu,sigma) || N(0,1))
        kl_loss = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(dim=-1).mean()

        total_loss = policy_loss + value_loss + reward_loss + self.kl_weight * kl_loss

        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.representation.parameters()) +
            list(self.afterstate_dyn.parameters()) +
            list(self.chance_encoder.parameters()) +
            list(self.transition.parameters()) +
            list(self.prediction.parameters()),
            max_norm=1.0
        )
        self.optimizer.step()

        self._clear_buffers()
        return policy_loss.item(), value_loss.item()

    def _clear_buffers(self):
        self.states_buffer  = []
        self.actions_buffer = []
        self.rewards_buffer = []
        self.pi_buffer      = []

    def save(self, path):
        torch.save({
            "representation":  self.representation.state_dict(),
            "afterstate_dyn":  self.afterstate_dyn.state_dict(),
            "chance_encoder":  self.chance_encoder.state_dict(),
            "transition":      self.transition.state_dict(),
            "prediction":      self.prediction.state_dict(),
        }, path)

    def load(self, path):
        data = torch.load(path, map_location=self.device)
        self.representation.load_state_dict(data["representation"])
        self.afterstate_dyn.load_state_dict(data["afterstate_dyn"])
        self.chance_encoder.load_state_dict(data["chance_encoder"])
        self.transition.load_state_dict(data["transition"])
        self.prediction.load_state_dict(data["prediction"])
        for net in [self.representation, self.afterstate_dyn,
                    self.chance_encoder, self.transition, self.prediction]:
            net.eval()