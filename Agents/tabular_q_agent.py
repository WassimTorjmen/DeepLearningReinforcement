import random


class TabularQAgent:
    def __init__(
        self,
        alpha=0.1,
        gamma=0.99,
        epsilon=1.0,
        epsilon_decay=0.995,
        epsilon_min=0.05,
    ):
        # Hyperparamètres du Q-learning
        self.alpha = alpha
        self.gamma = gamma

        # Exploration / exploitation
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        # Q-table : dictionnaire de dictionnaires
        # q_table[state_key][action] = valeur Q
        self.q_table = {}

    def state_to_key(self, state):
        """
        Convertit un état en clé compatible dictionnaire.
        - int -> int
        - tuple -> tuple
        - list -> tuple(list)
        """
        if isinstance(state, tuple):
            return state
        if isinstance(state, list):
            return tuple(state)
        return state

    def ensure_state_exists(self, state, valid_actions):
        """
        Si l'état n'existe pas encore dans la Q-table,
        on crée ses entrées d'actions à 0.0
        """
        state_key = self.state_to_key(state)

        if state_key not in self.q_table:
            self.q_table[state_key] = {}

        for action in valid_actions:
            if action not in self.q_table[state_key]:
                self.q_table[state_key][action] = 0.0

    def choose_action(self, state, valid_actions):
        """
        Politique epsilon-greedy :
        - avec probabilité epsilon : action aléatoire
        - sinon : meilleure action connue
        """
        self.ensure_state_exists(state, valid_actions)
        state_key = self.state_to_key(state)

        if random.random() < self.epsilon:
            return random.choice(valid_actions)

        q_values = self.q_table[state_key]
        return max(valid_actions, key=lambda a: q_values[a])

    def learn(self, state, action, reward, next_state, done, next_valid_actions):
        """
        Mise à jour Tabular Q-learning :
        Q(s,a) <- Q(s,a) + alpha * [target - Q(s,a)]

        target = reward                       si done
        target = reward + gamma * max Q(s',a') sinon
        """
        self.ensure_state_exists(state, [action])
        state_key = self.state_to_key(state)

        current_q = self.q_table[state_key][action]

        if done or len(next_valid_actions) == 0:
            target = reward
        else:
            self.ensure_state_exists(next_state, next_valid_actions)
            next_state_key = self.state_to_key(next_state)
            max_next_q = max(self.q_table[next_state_key][a] for a in next_valid_actions)
            target = reward + self.gamma * max_next_q

        self.q_table[state_key][action] = current_q + self.alpha * (target - current_q)

    def decay_epsilon(self):
        """
        Réduit progressivement epsilon après chaque épisode
        """
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)