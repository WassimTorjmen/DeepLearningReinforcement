# Rapport de preparation - Rendu final Deep Reinforcement Learning

Projet : 2026-5A-IABD-DRL  
Matiere : Deep Reinforcement Learning  
Objectif du syllabus : evaluer plusieurs techniques d'apprentissage par renforcement profond sur des environnements de complexite croissante, comprendre les atouts de chaque algorithme, expliquer le code, les choix d'hyperparametres et interpreter les resultats.

Ce document sert de support de comprehension du projet, de base de rapport et de fiche de preparation pour la soutenance.

## 1. Resume executif

Le projet implemente une plateforme experimentale de reinforcement learning autour de plusieurs environnements :

- `LineWorld` : environnement 1D simple.
- `GridWorld` : environnement 2D avec objectif et piege.
- `TicTacToe` : jeu adversarial contre un agent aleatoire.
- `Quarto` : jeu de plateau plus complexe, choisi comme gameplay principal avance.

Les agents implementes couvrent plusieurs familles :

- Baselines : `RandomAgent`, `RandomRolloutAgent`.
- Methodes tabulaires : `TabularQAgent`.
- Methodes value-based profondes : `DQNAgent`, `DoubleDQNAgent`.
- Policy gradients : `REINFORCE`, `REINFORCE with mean baseline`, `REINFORCE with critic`.
- Actor-critic style : `PPO_A2C`.
- Planification : `MCTS`, `Expert Apprentice`, `AlphaZero`, `MuZero`.

L'idee forte du projet est la progression pedagogique :

1. Valider l'interface des environnements sur des cas simples.
2. Ajouter des encodages vectoriels compatibles avec les reseaux de neurones.
3. Comparer des algorithmes de plus en plus complexes.
4. Montrer que les methodes simples marchent bien sur LineWorld/GridWorld mais deviennent instables sur Quarto.
5. Expliquer pourquoi les methodes avec critic, MCTS ou planification sont mieux adaptees aux jeux adversariaux complexes.

## 2. Alignement avec le syllabus

### 2.1 Exigences du syllabus

Le syllabus demande :

- Environnements de test :
  - Line World.
  - Grid World.
  - TicTacToe versus Random.
  - Un jeu au choix parmi Quarto, Bobail, Pond, ColorPop.

- Types d'agents a etudier :
  - Random.
  - Tabular Q-learning.
  - Deep Q-learning.
  - Double Deep Q-learning.
  - Double DQN avec Experience Replay.
  - Double DQN avec Prioritized Experience Replay.
  - REINFORCE.
  - REINFORCE with mean baseline.
  - REINFORCE with baseline learned by a critic.
  - PPO A2C style.
  - Random Rollout.
  - Monte Carlo Tree Search.
  - Expert Apprentice.
  - AlphaZero.
  - MuZero.
  - MuZero stochastique.

- Metriques :
  - Score moyen de la policy obtenue apres 1 000, 10 000, 100 000, 1 000 000 episodes si possible.
  - Temps moyen pour executer un coup.
  - Longueur moyenne d'une partie si la duree est variable.

- Livrables :
  - Code source.
  - Demonstration rapide.
  - Specifications d'encodage.
  - Readme de reproduction.
  - Rapport consequent sur les experimentations, resultats et observations critiques.
  - Documents de resultats.
  - Slides de presentation.

### 2.2 Etat du projet par rapport au syllabus

| Element attendu | Etat dans le projet | Commentaire |
|---|---:|---|
| LineWorld | Implemente | Environnement simple et stable. |
| GridWorld | Implemente | Version 5x5 avec objectif et piege. |
| TicTacToe vs Random | Implemente | L'agent joue X contre un adversaire random. |
| Jeu choisi | Quarto implemente | Bon choix car plus complexe et strategique. |
| Random | Implemente | `Agents/random_agent.py`. |
| Tabular Q-learning | Implemente | `Agents/tabular_q_agent.py`. |
| Deep Q-learning | Implemente | `Agents/DQN.py`. |
| Double DQN | Implemente | `Agents/ddqn.py`. |
| Experience Replay | Non implemente dans l'etat actuel | A mentionner comme limite ou piste. |
| Prioritized Experience Replay | Non implemente dans l'etat actuel | A mentionner comme limite ou piste. |
| REINFORCE | Implemente | `Agents/REINFORCE.py`. |
| REINFORCE mean baseline | Implemente | `Agents/Reinforce_mean_baseline.py`. |
| REINFORCE critic | Implemente | `Agents/Reinforce_critic.py`. |
| PPO A2C style | Implemente partiellement | Inspiration PPO avec critic et clipping, pas PPO complet multi-epoch. |
| Random Rollout | Implemente | Planification Monte Carlo simple. |
| MCTS UCT | Implemente | `Agents/MCTS.py`. |
| Expert Apprentice | Implemente | MCTS comme expert, reseau comme apprenti. |
| AlphaZero | Implemente simplifie | Policy-value network + MCTS PUCT. |
| MuZero | Implemente simplifie | Representation, dynamics, prediction. |
| MuZero stochastique | Non implemente dans l'etat actuel | A mentionner comme extension. |
| GUI / humain | Implemente | Interfaces pygame pour plusieurs environnements. |
| Encodages | Presents | Dossier `Encodings/`, quelques specs a mettre a jour. |
| Graphiques de resultats | Presents | PNG dans `Experimentations/`. |
| Modeles sauvegardes | A verifier/regenerer | Aucun `.pt` trouve au moment de l'analyse locale. |

## 3. Architecture du projet

### 3.1 Dossiers principaux

```text
Agents/
  Algorithmes d'apprentissage et agents de planification.

Environnements/
  Environnements RL et interfaces graphiques pygame.

Encodings/
  Specifications Markdown des vecteurs d'etat et d'action.

Experimentations/
  Pipeline experimental, scripts principaux, graphes de resultats.

Training/
  Scripts d'entrainement tabulaire.

Evaluation/
  Evaluation generique d'agents.

Benchmarks/
  Benchmark de performance, notamment pour Quarto.
```

### 3.2 Interface commune des environnements

Les environnements suivent une interface commune :

| Methode | Role |
|---|---|
| `reset()` | Reinitialise l'environnement. |
| `step(action)` | Applique une action et retourne l'etat suivant, la reward et le flag `done`. |
| `get_actions()` | Retourne les actions legales dans l'etat courant. |
| `available_actions()` | Alias de compatibilite. |
| `get_action_mask()` | Masque binaire pour filtrer les actions invalides. |
| `encode_state()` | Convertit l'etat en vecteur numerique pour les reseaux. |
| `encode_action_vector()` | Convertit une action en one-hot vector. |
| `render()` | Affichage console. |

Cette uniformisation est importante car elle permet d'utiliser le meme pipeline experimental avec plusieurs environnements.

## 4. Environnements

## 4.1 LineWorld

Fichier : `Environnements/line_world.py`

### Principe

L'agent se deplace sur une ligne de taille `size=6`. Il part de la case `0` et doit atteindre la case `size - 1`.

Actions :

- `0` : aller a gauche.
- `1` : aller a droite.

Rewards :

- `+1` si l'objectif est atteint.
- `0` sinon.

### Pourquoi cet environnement ?

LineWorld sert a valider les bases :

- boucle `reset -> action -> step -> reward`;
- apprentissage sur reward sparse;
- actions legales;
- convergence attendue simple.

Si un algorithme ne marche pas sur LineWorld, il ne marchera probablement pas sur les jeux plus complexes.

### Encodage reel

Pour `size=6`, l'etat a `8` dimensions :

```text
[position_agent_normalisee, position_goal_normalisee, one_hot_position_agent]
```

Exemple :

```text
agent en position 2, goal en position 5
[0.4, 1.0, 0, 0, 1, 0, 0, 0]
```

Interet :

- la position normalisee donne une information continue de distance;
- le one-hot donne la position exacte sans ambiguite.

## 4.2 GridWorld

Fichier : `Environnements/grid_world.py`

### Principe

L'agent se deplace dans une grille 5x5.

Positions :

- depart : `(0, 0)`;
- objectif : `(0, cols - 1)`, donc en haut a droite;
- piege : `(rows - 1, cols - 1)`, donc en bas a droite.

Actions :

- `0` : haut.
- `1` : bas.
- `2` : gauche.
- `3` : droite.

Rewards :

- `+1` si l'agent atteint l'objectif.
- `-1` si l'agent tombe dans le piege.
- `0` sinon.

### Encodage reel

Le code actuel produit un etat de taille `31` :

```text
2 dimensions position agent
2 dimensions position goal
2 dimensions position trap
25 dimensions one-hot position agent
= 31
```

Attention : le fichier `Encodings/gridworld_encoding_spec.md` indique une taille `29`, mais ce n'est plus aligne avec le code actuel, car le code encode aussi le piege.

### Interet pedagogique

GridWorld ajoute :

- plusieurs directions;
- un mauvais terminal avec reward negative;
- le besoin d'apprendre une trajectoire et d'eviter un risque.

## 4.3 TicTacToe

Fichier : `Environnements/tictactoe.py`

### Principe

Jeu de morpion 3x3.

- Joueur 1 : `X`, code `1`.
- Joueur 2 : `O`, code `-1`.
- L'agent joue `X`.
- L'adversaire joue aleatoirement.

Actions :

- `0` a `8`, une action par case.

Rewards :

- victoire : reward positive;
- match nul : reward neutre;
- defaite : reward negative dans les fonctions d'entrainement/evaluation.

### Encodage

L'etat a `27` dimensions :

```text
9 cases x 3 valeurs par case
```

Chaque case est encodee en one-hot :

```text
[vide, X, O]
```

Pourquoi ne pas utiliser directement `0`, `1`, `-1` ?

Parce qu'un reseau pourrait interpreter ces valeurs comme ordonnees numeriquement. Le one-hot dit seulement quelle categorie est presente.

## 4.4 Quarto

Fichier : `Environnements/quarto.py`

### Principe

Quarto est le jeu choisi pour aller au-dela des environnements de test.

Regles principales :

- plateau 4x4;
- 16 pieces;
- chaque piece possede 4 attributs binaires;
- un joueur gagne s'il aligne 4 pieces partageant au moins un attribut commun;
- particularite forte : un joueur choisit la piece que l'adversaire devra poser.

Phases :

- `PHASE_CHOOSE` : choisir une piece pour l'adversaire.
- `PHASE_PLACE` : placer la piece recue.

### Espace d'actions

Le projet encode les actions sur 32 possibilites :

```text
0..15   : choisir une piece
16..31  : placer une piece sur une case du plateau
```

Le masque d'action est essentiel :

- en phase `choose`, seules les pieces disponibles sont legales;
- en phase `place`, seules les cases vides sont legales.

### Encodage de l'etat

L'etat a `105` dimensions :

| Bloc | Taille | Contenu |
|---|---:|---|
| Piece a jouer | 5 | flag aucune piece + 4 attributs |
| Plateau | 80 | 16 cases x 5 valeurs |
| Pieces disponibles | 16 | une valeur par piece |
| Phase du jeu | 2 | one-hot choose/place |
| Joueur courant | 2 | one-hot joueur 1/joueur 2 |
| Total | 105 |  |

### Pourquoi cet encodage est pertinent ?

Quarto repose sur les attributs des pieces. Encoder une piece uniquement par son ID brut serait peu informatif. Encoder ses 4 attributs binaires donne au reseau les informations necessaires pour apprendre les alignements.

## 5. Agents implementes

## 5.1 RandomAgent

Fichier : `Agents/random_agent.py`

Agent baseline. Il choisit uniformement une action parmi les actions legales.

Role :

- verifier que l'environnement fonctionne;
- fournir une reference minimale;
- comparer les agents entraines a un comportement sans apprentissage.

## 5.2 Tabular Q-learning

Fichier : `Agents/tabular_q_agent.py`

### Idee

Le Q-learning apprend une table :

```text
Q(s, a)
```

qui estime la valeur de faire l'action `a` dans l'etat `s`.

Mise a jour :

```text
Q(s,a) <- Q(s,a) + alpha * [target - Q(s,a)]
```

Avec :

```text
target = reward + gamma * max_a' Q(s', a')
```

si l'episode n'est pas termine.

### Hyperparametres

| Hyperparametre | Valeur | Role | Interpretation |
|---|---:|---|---|
| `alpha` | 0.1 | taux d'apprentissage | mise a jour progressive de la Q-table |
| `gamma` | 0.99 | discount factor | valorise fortement les rewards futures |
| `epsilon` | 1.0 | exploration initiale | au debut, l'agent explore beaucoup |
| `epsilon_decay` | 0.995 | decroissance exploration | transition progressive vers l'exploitation |
| `epsilon_min` | 0.05 | exploration minimale | garde un peu d'exploration |

### Quand l'utiliser ?

Adapted aux petits espaces discrets comme LineWorld et GridWorld. Pas adapte a Quarto car l'espace d'etats devient trop grand.

## 5.3 DQN

Fichier : `Agents/DQN.py`

### Idee

DQN remplace la Q-table par un reseau de neurones :

```text
etat -> Q-value pour chaque action
```

Le reseau apprend a approximer la cible de Bellman.

### Architecture

```text
Linear(state_size, hidden_size)
ReLU
Linear(hidden_size, hidden_size)
ReLU
Linear(hidden_size, num_actions)
```

Pas d'activation en sortie, car une Q-value peut etre negative, positive, grande ou petite.

### Limite actuelle

Le fichier contient une base DQN, mais les fonctions generiques `evaluate` et `train` appellent `env.score()`, qui n'existe pas dans les environnements actuels. Pour les resultats finaux, le pipeline principal utilise surtout les agents policy-gradient et planning.

## 5.4 Double DQN

Fichier : `Agents/ddqn.py`

### Idee

DQN peut surestimer les valeurs car il utilise le meme reseau pour choisir et evaluer l'action suivante.

Double DQN separe :

- reseau principal : choisit la meilleure action;
- reseau cible : evalue cette action.

Cela stabilise l'apprentissage.

### Hyperparametre important

`target_update_every=100` : le reseau cible est resynchronise toutes les 100 steps.

### Limite actuelle

Comme DQN, l'implementation existe, mais il manque les variantes Experience Replay et Prioritized Experience Replay demandees dans le syllabus.

## 5.5 REINFORCE

Fichier : `Agents/REINFORCE.py`

### Idee

REINFORCE est une methode policy-gradient Monte Carlo.

Au lieu d'apprendre `Q(s,a)`, l'agent apprend directement une politique :

```text
pi(a | s)
```

L'agent echantillonne une action selon la distribution de probabilites produite par le reseau. A la fin de l'episode, il augmente la probabilite des actions qui ont mene a de bons retours.

### Architecture

```text
Linear(state_size, hidden_size)
ReLU
Linear(hidden_size, hidden_size)
ReLU
Linear(hidden_size, num_actions)
Softmax
```

### Pourquoi ReLU ?

ReLU est simple, rapide et efficace pour les couches cachees. Elle evite une partie des problemes de saturation des sigmoides et permet au reseau d'apprendre des relations non lineaires.

### Pourquoi Softmax en sortie ?

Softmax transforme les scores du reseau en distribution de probabilite :

```text
somme des probabilites = 1
```

C'est necessaire pour echantillonner une action avec `Categorical`.

### Masquage des actions invalides

Le reseau produit une probabilite pour toutes les actions. Ensuite :

1. on met a zero les actions invalides;
2. on renormalise les probabilites restantes.

C'est indispensable pour TicTacToe et Quarto, ou toutes les actions ne sont pas legales a chaque etat.

### Hyperparametres

| Hyperparametre | Valeur courante | Role |
|---|---:|---|
| `gamma` | 0.99 | donne beaucoup d'importance aux rewards futures |
| `lr` | 1e-3 par defaut | vitesse d'apprentissage |
| `hidden_size` | 64 par defaut | capacite du reseau |
| `hidden_size` Quarto | 128 ou 256 selon essais | Quarto demande plus de capacite |
| `lr` Quarto | 1e-4 dans certains essais | stabilise l'apprentissage sur environnement complexe |

### Interpretation

REINFORCE marche bien sur LineWorld/GridWorld, mais devient bruite sur TicTacToe et instable sur Quarto. C'est normal : REINFORCE attend la fin de l'episode pour apprendre, donc les gradients ont une forte variance.

## 5.6 REINFORCE with mean baseline

Fichier : `Agents/Reinforce_mean_baseline.py`

### Idee

On soustrait la moyenne des retours :

```text
advantage = G_t - mean(G)
```

Cela reduit la variance du gradient.

Difference avec REINFORCE classique :

- REINFORCE classique normalise avec moyenne et ecart-type.
- Mean baseline soustrait seulement la moyenne.

### Pourquoi c'est utile ?

Une baseline ne change pas l'objectif theorique, mais stabilise la mise a jour. Elle permet de renforcer les actions meilleures que la moyenne et de penaliser celles moins bonnes.

## 5.7 REINFORCE with critic

Fichier : `Agents/Reinforce_critic.py`

### Idee

On ajoute un second reseau, le critic, qui apprend :

```text
V(s)
```

La politique utilise ensuite :

```text
advantage = G_t - V(s_t)
```

Le critic estime ce qui etait attendu depuis un etat. L'actor est donc mis a jour selon ce qui a ete meilleur ou pire que prevu.

### Architecture

Policy :

```text
etat -> probabilites d'actions
```

Critic :

```text
etat -> valeur scalaire V(s)
```

La sortie critic n'a pas de Softmax, car ce n'est pas une probabilite mais une valeur.

### Interet

Sur les resultats observes, le critic ameliore fortement TicTacToe par rapport a REINFORCE classique. C'est un point important a defendre en soutenance : plus l'environnement est stochastique ou adversarial, plus la reduction de variance est utile.

## 5.8 PPO A2C style

Fichier : `Agents/PPO_A2C.py`

### Idee

L'implementation reprend :

- une policy;
- un critic;
- un advantage;
- un ratio entre nouvelle et ancienne probabilite;
- un clipping du ratio.

Le clipping limite les changements trop grands de politique :

```text
ratio clipped dans [1 - clip_eps, 1 + clip_eps]
```

Avec `clip_eps=0.2`, la politique ne devrait pas trop s'eloigner d'un coup.

### Precision critique

Le fichier est nomme `PPO_A2C`, et il est correct de le presenter comme une version simplifiee inspiree de PPO/A2C. Ce n'est pas un PPO complet standard, car un PPO complet utilise generalement :

- collecte de batchs;
- plusieurs epochs d'optimisation;
- ancienne policy figee explicitement;
- parfois entropy bonus.

## 5.9 Random Rollout

Fichier : `Agents/Random_Rollout.py`

### Idee

Pour chaque action candidate :

1. on copie l'environnement;
2. on joue cette action;
3. on simule plusieurs parties aleatoires;
4. on choisit l'action au meilleur score moyen.

### Interet

Pas besoin d'entrainement. C'est une methode de planification simple.

### Limite

Tres couteux sur Quarto, car chaque decision demande beaucoup de simulations.

## 5.10 MCTS UCT

Fichier : `Agents/MCTS.py`

### Idee

Monte Carlo Tree Search construit un arbre de recherche avec quatre phases :

1. Selection.
2. Expansion.
3. Simulation.
4. Backpropagation.

La selection utilise UCT :

```text
UCT = exploitation + exploration
```

Le terme exploration favorise les actions peu visitees.

### Atout

MCTS reutilise les simulations precedentes dans un arbre. Il est donc plus structure que Random Rollout.

### Limite

Il reste couteux en temps de decision.

## 5.11 Expert Apprentice

Fichier : `Agents/Expert_Apprentice.py`

### Idee

Deux phases :

1. Un expert MCTS choisit les actions.
2. Un reseau apprenti apprend a imiter l'expert par classification supervisee.

Avantage :

- MCTS est fort mais lent;
- l'apprenti est moins fort mais beaucoup plus rapide a l'inference.

## 5.12 AlphaZero simplifie

Fichier : `Agents/Alpha_zero.py`

### Idee

AlphaZero combine :

- un reseau policy-value;
- MCTS guide par le reseau;
- apprentissage a partir des distributions de visite MCTS.

Le reseau a deux tetes :

```text
policy head : probabilite sur actions
value head  : estimation du resultat final
```

### Activations

- `Softmax` pour la policy : distribution d'actions.
- `Tanh` pour la value : valeur bornee entre `-1` et `1`.

### Hyperparametres

| Hyperparametre | Valeur | Role |
|---|---:|---|
| `hidden_size` | 128 par defaut | capacite plus grande que REINFORCE |
| `n_simulations` | 5 dans `main.py` | compromis temps/qualite |
| `c_puct` | 1.0 | equilibre exploration/exploitation dans MCTS |
| `lr` | 1e-3 | apprentissage reseau |

### Interpretation

AlphaZero obtient de bons resultats sur TicTacToe en peu d'episodes, car MCTS fournit une cible de politique plus informative qu'une simple reward finale.

## 5.13 MuZero simplifie

Fichier : `Agents/MuZero.py`

### Idee

MuZero n'apprend pas seulement une policy et une value. Il apprend aussi un modele latent de dynamique.

Trois reseaux :

- representation : transforme l'etat reel en etat latent;
- dynamics : predit l'etat latent suivant et la reward;
- prediction : predit policy et value depuis le latent.

### Difference avec AlphaZero

AlphaZero utilise l'environnement pour simuler.  
MuZero apprend a simuler dans un espace latent.

### Limite de cette implementation

C'est une version simplifiee :

- planification a un niveau;
- pas de vrai arbre MuZero complet;
- pas de version stochastique.

## 6. Pipeline experimental

Fichier principal : `Experimentations/experiment.py`

### 6.1 Fonctions d'entrainement

| Fonction | Usage |
|---|---|
| `train_1player` | LineWorld et GridWorld |
| `train_tictactoe` | TicTacToe contre random |
| `train_quarto` | Quarto contre random |
| `train_alphazero_1player` | AlphaZero/MuZero sur LineWorld/GridWorld |
| `train_alphazero_tictactoe` | AlphaZero/MuZero sur TicTacToe |
| `train_alphazero_quarto` | AlphaZero/MuZero sur Quarto |

### 6.2 Fonctions d'evaluation

Le syllabus insiste sur un point important :

```text
Les metriques doivent evaluer la policy obtenue, pas la policy en mode entrainement.
```

Le projet respecte cette idee en evaluation :

- l'action est choisie par `argmax`, pas par echantillonnage;
- l'adversaire est random pour TicTacToe et Quarto;
- les scores sont calcules sur 500 parties par defaut.

### 6.3 Graphiques

`plot_results` produit :

- reward moyen lissee sur une fenetre de 100 episodes;
- policy loss;
- critic loss si l'agent possede un critic.

Ces graphes sont sauvegardes dans `Experimentations/`.

## 7. Metriques et interpretation

## 7.1 Score moyen

Le score moyen mesure la performance de la politique apres entrainement.

Interpretation generale :

- `1.0` : victoire ou succes quasi systematique.
- `0.0` : performance neutre, nul, ou equilibre victoires/defaites.
- `-1.0` : defaite ou mauvais terminal systematique.

Selon l'environnement :

- LineWorld : `1` signifie atteindre l'objectif.
- GridWorld : `1` signifie atteindre l'objectif, `-1` tomber dans le piege.
- TicTacToe : score positif si l'agent gagne contre random.
- Quarto : score positif si joueur 1 gagne, negatif si joueur 2 gagne.

## 7.2 Longueur moyenne

Nombre moyen de steps par partie.

Interpretation :

- courte et score eleve : l'agent resout vite.
- courte et score negatif : l'agent perd vite.
- longue et score proche de zero : parties incertaines, nuls ou strategie peu decisive.

## 7.3 Temps moyen par coup

Temps de decision de l'agent.

Interpretation :

- reseau simple : decision tres rapide;
- Random Rollout/MCTS : decision plus lente car simulations;
- AlphaZero/MuZero : compromis entre reseau et planification.

Cette metrique est essentielle pour discuter le compromis performance/temps.

## 7.4 Policy loss

La policy loss n'est pas une metrique de performance directe.

En policy gradient, elle peut etre negative, positive ou bruitee. Ce qui compte surtout est la tendance de la reward.

Une loss tres instable peut signaler :

- gradients bruites;
- learning rate trop eleve;
- rewards rares;
- environnement trop complexe pour la methode.

## 7.5 Critic loss

La critic loss mesure l'erreur entre :

```text
V(s_t)
```

et le retour reel observe.

Si elle baisse, le critic estime mieux les retours, ce qui stabilise l'actor.

## 8. Resultats observes dans les graphes

Les observations ci-dessous proviennent des PNG presents dans `Experimentations/`.

## 8.1 LineWorld

REINFORCE atteint un reward moyen de `1.0` tres rapidement et reste stable.

Interpretation :

- l'environnement est simple;
- reward terminale facile a attribuer;
- peu d'actions;
- convergence attendue.

Message soutenance :

> LineWorld valide que notre pipeline d'apprentissage, d'encodage et d'evaluation fonctionne.

## 8.2 GridWorld

REINFORCE converge vers un score proche de `1.0`, apres une phase initiale instable.

Interpretation :

- l'agent apprend a atteindre le goal;
- il evite progressivement le piege;
- la complexite est superieure a LineWorld mais reste raisonnable.

Message soutenance :

> GridWorld montre que la methode sait apprendre une trajectoire dans un espace 2D avec une reward negative.

## 8.3 TicTacToe

REINFORCE classique progresse mais reste instable, autour de `0.6 - 0.7` sur les graphes.

REINFORCE Critic et PPO/A2C montent plutot vers `0.9+`.

AlphaZero atteint aussi un niveau eleve rapidement, environ `0.85 - 0.95`.

Interpretation :

- TicTacToe est adversarial;
- une simple reward finale donne un signal bruite;
- le critic reduit la variance;
- MCTS/AlphaZero fournit des cibles d'action plus informatives.

Message soutenance :

> L'ajout d'une baseline apprise ou d'une planification ameliore nettement la stabilite sur un jeu adversarial.

## 8.4 Quarto

Quarto est le cas le plus difficile.

Observations :

- REINFORCE peut monter au debut mais reste instable.
- Sur le run recent, on a observe :
  - 1 000 episodes : score `0.1060`;
  - 10 000 episodes : score `0.2740`;
  - crash avant 100 000 episodes a cause de probabilites `NaN`.
- Le graphe `REINFORCE_lr1e4_h256_100000_Quarto.png` montre une progression jusqu'a environ `0.4 - 0.5`, puis un retour vers une zone proche de `0`.

Interpretation :

- l'espace d'action est grand : 32 actions;
- l'etat est plus grand : 105 dimensions;
- les rewards sont rares;
- le jeu alterne choix de piece et placement;
- une mauvaise action peut aider l'adversaire plusieurs tours plus tard;
- REINFORCE a une variance elevee;
- l'entrainement peut devenir numeriquement instable.

Message soutenance :

> Quarto met en evidence les limites des methodes policy-gradient simples et justifie l'usage de baselines, critic, MCTS, AlphaZero ou d'une meilleure stabilisation numerique.

## 9. Choix d'hyperparametres

## 9.1 Discount factor `gamma = 0.99`

`gamma` determine l'importance des recompenses futures.

Pourquoi `0.99` ?

- Dans ces environnements, la reward importante arrive souvent en fin d'episode.
- Un gamma faible ferait oublier le futur.
- `0.99` permet de propager la reward finale vers les actions precedentes.

Limite :

- Peut augmenter la variance sur des episodes longs.

## 9.2 Learning rate

Valeurs observees :

- `1e-3` par defaut.
- `5e-4` pour certains essais TicTacToe.
- `1e-4` pour Quarto.

Interpretation :

- `1e-3` est une valeur classique avec Adam pour petits MLP.
- Sur Quarto, le probleme est plus instable, donc baisser a `1e-4` est logique.
- Un learning rate trop grand peut provoquer des poids extremes et des `NaN`.

## 9.3 Hidden size

Valeurs observees :

- `64` par defaut;
- `128` pour Quarto dans plusieurs agents;
- `256` pour certains essais REINFORCE Quarto.

Interpretation :

- LineWorld/GridWorld n'ont pas besoin d'un grand reseau.
- TicTacToe demande un peu plus de representation.
- Quarto demande plus de capacite car l'etat encode beaucoup plus d'information.

Limite :

- Un reseau plus grand peut aussi sur-apprendre ou rendre l'optimisation plus instable.

## 9.4 Nombre d'episodes

Valeurs observees :

- 1 000 episodes pour AlphaZero/MuZero dans `main.py`.
- 10 000 episodes pour REINFORCE/PPO/Critic.
- 100 000 episodes pour essais avances TicTacToe/Quarto.

Interpretation :

- Les methodes avec MCTS coutent cher, donc moins d'episodes.
- Les methodes policy-gradient simples sont rapides par coup, donc peuvent etre entrainees plus longtemps.

## 9.5 Nombre de simulations MCTS

Valeurs observees :

- MCTS/AlphaZero : `n_simulations=5` dans certains runs experimentaux.
- Random Rollout : `n_rollouts=10` dans les exemples commentes.

Interpretation :

- Plus de simulations donne de meilleures decisions.
- Mais le temps par coup augmente fortement.
- `5` ou `10` est un compromis pour pouvoir executer les experiences dans un temps raisonnable.

## 9.6 Gradient clipping

Plusieurs agents utilisent :

```python
torch.nn.utils.clip_grad_norm_(..., max_norm=1.0)
```

Role :

- eviter l'explosion des gradients;
- stabiliser l'entrainement;
- utile dans les environnements avec rewards rares ou instables.

## 10. Fonctions d'activation

## 10.1 ReLU

Utilisee dans les couches cachees.

Avantages :

- simple;
- rapide;
- bonne propagation du gradient pour valeurs positives;
- standard dans les MLP.

Pourquoi pas sigmoid ?

- sigmoid peut saturer;
- gradients faibles;
- apprentissage plus lent.

## 10.2 Softmax

Utilisee en sortie des policies.

Role :

- transforme les logits en probabilites;
- permet d'echantillonner avec une distribution categorielle;
- necessaire pour les policy gradients.

Limite :

- si les poids deviennent numeriquement instables, Softmax peut produire des `NaN`;
- sur Quarto, c'est une source possible du crash observe.

## 10.3 Tanh

Utilisee pour les tetes value dans AlphaZero/MuZero.

Role :

- borne la valeur entre `-1` et `1`;
- coherent avec des resultats de partie du type defaite/nul/victoire.

## 10.4 Pas d'activation en sortie Q-value

DQN et DDQN n'utilisent pas d'activation finale.

Raison :

- une Q-value n'est pas une probabilite;
- elle peut etre negative ou positive;
- elle doit rester libre en amplitude.

## 11. Observations critiques

## 11.1 Points forts

- Plusieurs environnements de difficulte croissante.
- Interface commune bien pensee.
- Encodages explicites.
- Masques d'actions illegales.
- Plusieurs familles d'algorithmes.
- Graphiques de reward/loss deja generes.
- GUI disponible pour les demonstrations.
- Quarto est un bon choix de gameplay avance.

## 11.2 Limites techniques

### Specs d'encodage a mettre a jour

`Encodings/gridworld_encoding_spec.md` indique une taille `29`, mais le code actuel encode aussi le piege et produit `31` dimensions.

### DQN/DDQN pas completement relies au pipeline final

Les fichiers DQN/DDQN existent, mais certaines fonctions appellent `env.score()`, absent des environnements actuels.

### Experience Replay absent

Le syllabus demande :

- DoubleDQNWithExperienceReplay;
- DoubleDQNWithPrioritizedExperienceReplay.

Ces variantes ne sont pas presentes dans l'etat actuel.

### PPO simplifie

`PPO_A2C.py` est une version inspiree PPO/A2C, pas un PPO complet strict.

### Instabilite sur Quarto

Le crash `NaN` de REINFORCE sur Quarto est une limite importante.

Cause probable :

- Softmax produit des probabilites non finies apres mise a jour instable;
- Adam peut conserver des etats internes instables;
- reward sparse et variance elevee;
- environnement plus long et adversarial.

Correction possible, a valider avant toute modification :

- remplacer les tests `torch.isnan` par `torch.isfinite`;
- recreer l'optimiseur si la policy est reinitialisee;
- utiliser des logits masques au lieu de probabilites Softmax renormalisees;
- reduire encore le learning rate;
- ajouter entropy regularization.

## 12. Comment presenter le projet en soutenance

La soutenance dure 20 minutes. Proposition de deroule :

### 0 - 2 min : contexte

- Objectif : comparer plusieurs algorithmes DRL.
- Environnements : LineWorld, GridWorld, TicTacToe, Quarto.
- Quarto choisi comme jeu principal complexe.

### 2 - 5 min : environnements et encodages

- Interface commune.
- Encodage one-hot.
- Masques d'actions.
- Zoom sur Quarto : 105 dimensions, 32 actions, phases choose/place.

### 5 - 10 min : agents

Presenter par familles :

- Random et Q-learning : baselines.
- DQN/DDQN : value-based profond.
- REINFORCE et variantes : policy-gradient.
- PPO/Critic : stabilisation par value function.
- MCTS/AlphaZero/MuZero : planification.

### 10 - 15 min : resultats

- LineWorld : resolu.
- GridWorld : resolu, evite le piege.
- TicTacToe : critic/PPO/AlphaZero plus stables que REINFORCE.
- Quarto : progression difficile, instabilite, limite des methodes simples.

### 15 - 18 min : analyse critique

- Pourquoi Quarto est difficile.
- Compromis performance/temps de decision.
- Pourquoi les hyperparametres ont ete ajustes.
- Ce qui manque par rapport au syllabus.

### 18 - 20 min : conclusion et demo

- Montrer une GUI.
- Expliquer comment lancer une experience.
- Conclure sur les apprentissages.

## 13. Reponses pretes pour questions de soutenance

### Pourquoi avoir choisi Quarto ?

Quarto est plus riche que LineWorld, GridWorld et TicTacToe. Il combine un plateau 4x4, des pieces avec attributs, deux phases de jeu et une interaction strategique originale : on donne a l'adversaire la piece qu'il doit poser. Cela force l'agent a raisonner non seulement sur son coup actuel, mais aussi sur ce qu'il offre a l'adversaire.

### Pourquoi utiliser un masque d'actions ?

Parce que le reseau produit une sortie pour toutes les actions possibles, mais toutes ne sont pas legales a chaque etat. Sans masque, l'agent pourrait choisir une case deja occupee ou une piece deja utilisee. Le masque garantit que la policy finale ne choisit que des actions valides.

### Pourquoi REINFORCE est instable ?

REINFORCE apprend a partir du retour final de l'episode. Sur des jeux longs et adversariaux, le signal est rare et bruite. Une action prise tot peut avoir un impact longtemps apres, donc l'attribution de credit est difficile. Cela produit des gradients de forte variance.

### Pourquoi ajouter un critic ?

Le critic estime la valeur attendue d'un etat. Au lieu de dire seulement "ce coup a mene a une victoire", on regarde si le resultat est meilleur ou pire que ce qui etait attendu. Cela reduit la variance et stabilise l'actor.

### Pourquoi AlphaZero marche bien sur TicTacToe ?

Parce que MCTS explore plusieurs futurs possibles et fournit une distribution d'actions plus informative que la reward finale seule. Le reseau apprend a imiter cette recherche, ce qui accelere l'apprentissage.

### Pourquoi les graphes de loss sont bruites ?

En reinforcement learning, surtout en policy gradient, la loss n'est pas comparable a une loss supervisee classique. Les donnees changent car elles dependent de la policy actuelle. La reward moyenne est plus importante pour evaluer la performance.

### Pourquoi mesurer le temps moyen par coup ?

Deux agents peuvent avoir le meme score mais des couts tres differents. Un reseau simple joue tres vite. MCTS peut mieux jouer mais prend plus de temps car il simule beaucoup de parties. Le temps par coup mesure ce compromis.

## 14. Commandes utiles

Depuis la racine du projet :

```powershell
python Training/train_tabular_lineworld.py
python Training/train_tabular_gridworld.py
python Evaluation/test_tabular_lineworld.py
python Benchmarks/benchmark_quarto.py
python Experimentations/main.py
python Experimentations/monmain.py
```

Interfaces graphiques :

```powershell
python Environnements/lineworld_gui.py
python Environnements/gridworld_gui.py
python Environnements/tictactoe_gui.py
python Environnements/quarto_gui.py
```

Note : les GUI utilisent `pygame`.

## 15. A faire avant rendu final si possible

Priorite haute :

- Mettre a jour `Encodings/gridworld_encoding_spec.md` pour indiquer `31` dimensions.
- Stabiliser REINFORCE sur Quarto ou presenter clairement le crash comme limite.
- Regenerer les modeles `.pt` si le rendu exige des modeles sauvegardes.
- Ajouter un README de reproduction court.

Priorite moyenne :

- Ajouter Experience Replay et Prioritized Experience Replay si le temps le permet.
- Harmoniser les imports et scripts de lancement.
- Ajouter un tableau final de resultats numeriques par agent/environnement.

Priorite soutenance :

- Preparer 1 slide par environnement.
- Preparer 1 slide par famille d'algorithmes.
- Preparer 1 slide de resultats.
- Preparer 1 slide de limites et perspectives.

## 16. Conclusion

Le projet est solide pour une soutenance car il montre une vraie demarche experimentale :

- partir d'environnements simples;
- complexifier progressivement;
- tester plusieurs familles d'algorithmes;
- mesurer score, longueur et temps de decision;
- analyser les limites.

Le message principal a faire passer :

> Les algorithmes simples suffisent pour LineWorld et GridWorld, mais les jeux adversariaux et surtout Quarto necessitent des methodes plus stables, avec baseline, critic ou planification. Les resultats montrent donc non seulement ce qui marche, mais aussi pourquoi certains algorithmes atteignent leurs limites.

