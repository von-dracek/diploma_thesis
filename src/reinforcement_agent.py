#it seems that we will need to enumerate the states that the agent can use (i.e. all
# branching and #stages possibilities that result in < 10000 leaf nodes).
#the algorithm will be as follows:

#start from a random state and let the agent explore the state space to minimise cvar where the reward will
#be minimum cvar and also potentially a penalization for choosing a large tree

#an improvement could be achieved if the agent also gets characterisation of the dataset as input
#(i.e. number of stocks, minimum return of the stocks, maximum and maybe some quantiles)

#algorithms
#q learning, rainbow?
#https://www.toptal.com/machine-learning/deep-dive-into-reinforcement-learning

#idea for experiment
#train the algorithm on some dataset and then evaluate it out of sample in the following way:
#calculate the loss function for each possible tree (maybe randomly sampling from the environment)
#calculate the loss function for the prediction from the model
#look at the distribution of loss function over each possible tree (or randomly sampled) and see if it performs better than 50th percentile

#idea for experiment
#give the agent state consisting of last N actions
#idea - If you wrote a rock-paper-scissors agent to play against a human opponent, you might actually formulate it as a reinforcement learning problem, taking the last N plays as the state, because that could learn to take advantage of human players' poor judgement of randomness.

#idea - structure this as a grid game problem - let the agent choose first the number of stages and then iteratively a branching based on the previous state (previously chosen branchings) and then
#receive reward after large enough number of branchings is reached and get a reward - we can use q learning for this
#invalid actions must be ignored - i.e. allow to choose only actions that lead to a sensible tree at least of length n
#we can do this by subclassing the gridworld environment

from itertools import combinations_with_replacement
from typing import List
import tensorflow as tf
import numpy as np
import random
import logging

from main import get_cvar_value
from src.configuration import EPSILON_FOR_EPSILON_GREEDY_ACTION, MAX_NUMBER_LEAVES_IN_SCENARIO_TREE, NUMBER_STEPS_TO_TRAIN_REINFORCEMENT_AGENT


def non_increasing(L):
    return all(x>=y for x, y in zip(L, L[1:]))


class Environment:
          environment: list

          #initialize simple environment - fixed branching, only choose number of stages
          def __init__(self, maximum_stages: int, branching: int):
                    L = maximum_stages
                    self.environment = []
                    branchings = list(reversed(list(range(1, L)))) #we require at least 3 child nodes in each stage to achieve good result in moment matching
                    # only generating possible branchings that are non increasing - we need a good discretization of the near future while the far future need not be as precise
                    for length in range(1, L):
                            self._check_and_add([branching] * length)


          def __init__complex_env(self):
                    L = 10
                    self.environment = []
                    branchings = list(reversed(list(range(3, L)))) #we require at least 3 child nodes in each stage to achieve good result in moment matching
                    # only generating possible branchings that are non increasing - we need a good discretization of the near future while the far future need not be as precise
                    for length in range(1, L):
                              for elem in list(combinations_with_replacement(branchings, length)):
                                        self._check_and_add(elem)

          def _check_and_add(self, i: List):
                    if np.prod(i) < MAX_NUMBER_LEAVES_IN_SCENARIO_TREE and non_increasing(i):
                              self.environment.append(list(i,))
          @property
          def shape(self):
                   return len(self.environment)

class Agent:
          def _greedy_action(self, option_probabilities: np.ndarray) -> int:
                    return np.amax(option_probabilities)

          def _epsilon_greedy_action(self, option_probabilities: np.ndarray) -> int:
                    rnd = random.random()
                    logging.info(f"Option probabilities: {option_probabilities}")
                    logging.info(f"Performed pulls: {self.performed_pulls}")
                    logging.info(f"Cumulative rewards: {self.cumulative_rewards}")
                    return np.argmax(option_probabilities) if rnd > EPSILON_FOR_EPSILON_GREEDY_ACTION else np.random.choice((np.arange(option_probabilities.shape[0])))

          def __init__(self):
              self.environment = Environment(3, 3)
              self.performed_iterations = 0
              self.initial_weights = np.ones(self.environment.shape)/self.environment.shape
              self.performed_pulls = np.zeros(self.environment.shape)
              self.cumulative_rewards = np.zeros(self.environment.shape)

          # def __nn_init__(self):
          #       self.environment = Environment()
          #
          #       #initialize weights as uniform
          #       self.weights = tf.Variable(tf.ones(self.environment.shape))
          #       #initialize model
          #       #inputs are characteristics of the given dataset
          #       # inputs = tf.keras.layers.Input(shape=self.environment.shape)
          #       hidden = tf.keras.layers.Dense(units=5,
          #                                      activation=tf.nn.relu)(inputs)
          #       outputs = tf.keras.layers.Dense(units=self.environment.shape,
          #                                       activation=tf.nn.softmax)(hidden)
          #       self._model = tf.keras.Model(inputs=[inputs], outputs=[outputs])
          #       self._model.compile(tf.optimizers.Adam(0.001),
          #                           loss=tf.losses.BinaryCrossentropy())

          def get_action_and_reward(self):
                    chosen_action = self._epsilon_greedy_action(self.Q())
                    reward = get_cvar_value(self.environment.environment[chosen_action])
                    self.performed_iterations += 1
                    self.performed_pulls[chosen_action] += 1
                    self.cumulative_rewards[chosen_action] += reward
                    return chosen_action, reward

          def train(self, n_actions: int):
              for iter in range(n_actions):
                  action, reward = self.get_action_and_reward()
                  logging.info(f"Action {action} reward {reward}")

          def Q(self):
              if self.performed_iterations < 20 or (not np.all(self.performed_pulls > 0)):
                return self.initial_weights
              else:
                return self.cumulative_rewards / self.performed_pulls



ag = Agent()
ag.train(100)
