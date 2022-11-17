import gym
import numpy as np
from gym import spaces
# gridworld
# possible branching - width - 3 - 8
# depth - 2 - 8
from random import random, uniform

from main import get_cvar_value, get_necessary_data
import logging

from src.configuration import MAX_NUMBER_LEAVES_IN_SCENARIO_TREE

valid_action_reward = 0.1
invalid_action_reward = -valid_action_reward

def _get_predictors_from_data(data):
    """
    Get predictors that might be useful from the data.
    Particularly:
        alpha,
        number of stocks and
        statistics of returns over whole period.
    """
    predictors = np.zeros(9)
    predictors[0] = uniform(0.9, 0.95) # alpha
    predictors[1] = len(data.columns)  # n_stocks
    returns = data.iloc[[0,-1],:].pct_change().iloc[-1] + 1
    predictors[2] = max(returns) #maximal return over whole period
    predictors[3] = min(returns) #minimal return over the whole period
    predictors[4] = returns.quantile(0.75)
    predictors[5] = returns.quantile(0.5)
    predictors[6] = returns.quantile(0.25)
    predictors[7] = returns.mean()
    predictors[8] = returns.var()
    return predictors

def _reward_func_v2(branching, data, alpha, train_or_test):
    """Calculates reward from chosen branching."""
    #the minus sign before the cvar value has to be added, since we
    #calculate cvar-alpha of the loss distribution - thus the smaller the
    #cvar value the better. But the reinforcement agent in fact maximises
    #rewards - thus we need to give it a positive reward such that
    #higher value -> better reward and that is exactly what the
    #minus sign does here
    num_scenarios = np.prod(branching)
    coef_minus_reward_scenario_tree_complexity = 1/1000
    penalty_for_complexity = coef_minus_reward_scenario_tree_complexity*num_scenarios
    return -get_cvar_value(branching, data, alpha, train_or_test) - penalty_for_complexity

class TreeBuildingEnv_v4(gym.Env):
    """
    Adaptation of gridworld environment for tree building purposes.
    First choosing depth of tree, then each step chooses next branching.
    Reward is returned when the tree is completely built.
    The observation space is a 9*9 table, where the first row represents possible
    depths of tree (only valid are 3-8), and successive rows represent
    the brachings in each level.
    """

    def __init__(self, train_or_test:str = "train"):
        self.train_or_test = train_or_test
        self.height = 9
        # first level in height is chosen depth of tree
        # next levels represent branching at each stage
        self.width = 9
        self.observation_space = spaces.Dict(state=spaces.MultiDiscrete([2]*self.height*self.width), predictors=spaces.Box(shape=(9,), low=0, high=1000))
        self.reset()
        # begin in start state

    def step(self, action):
        """
        Takes a step in the environment according to a chosen action.
        If invalid depth of tree is chosen, episode is ended with large negative reward.
        If invalid branching is chosen (according to maximum number of scenarios),
        the agent is forced to take the maximum possible branching in current stage and
        receives small negative reward.
        """
        if action not in self.action_space:
            raise NotImplementedError
        if self.done:
            raise NotImplementedError
        valid_actions = self.valid_actions()
        if valid_actions[action]==0: #checks if action is valid
            invalid_action = True
            if self.depth == 0: #invalid depth of tree
                self.done=True
                return self.get_current_observation_state(), -10, self.done, {"num_scen":self.current_num_scenarios} #if invalid depth of tree is chosen, end the episode
            else: #return maximum valid action - if the agent chooses a bad action, we force him to take maximum
                amax = np.argmax(valid_actions[::-1])
                action =len(valid_actions) - amax - 1
        else:
            invalid_action = False
        x = action
        y = self.current_y
        if y == 0: #first assignment chooses depth of tree
            self.depth = x
        self.current_y += 1
        self.S[y, x] = 1
        if self.current_y == np.argmax(self.S[0, :]) + 1: #all branchings have been chosen - end episode
            self.done = True
            return self.get_current_observation_state(), _reward_func_v2(self.get_branching(), self.data, self.predictors[0], self.train_or_test), self.done, {"num_scen":self.current_num_scenarios}
        return self.get_current_observation_state(), valid_action_reward if not invalid_action else invalid_action_reward, self.done, {}

    @property
    def remaining_depth(self):
        return self.depth - self.current_y + 1

    @property
    def current_num_scenarios(self):
        amax = np.argmax(self.S[1:, :], axis=1)
        curr_num_scenarios = np.prod(amax[amax > 0])
        return curr_num_scenarios

    @property
    def action_space(self):
        return spaces.Discrete(9)

    def is_action_valid(self, action):
        if action < 3 or action >= 9: #in other states permit only actions 3 - 8
            return False
        if (self.action_space is not None) and (self.depth > 0) and self.remaining_depth > 0:
            current_num_scenarios = self.current_num_scenarios
            if (
                current_num_scenarios * action * (3 ** (self.remaining_depth - 1))
                < MAX_NUMBER_LEAVES_IN_SCENARIO_TREE
            ):
                return True
        return False

    def valid_actions(self):
        if self.depth == 0:
            va = [False, False]
            for i in range(2, self.width):
                va.append(3**i < MAX_NUMBER_LEAVES_IN_SCENARIO_TREE)
            arr = np.array(va) * 1
            return np.array(arr, dtype=np.int8)
        else:
            va = [False, False, False]
            for i in range(3, self.width):
                va.append(self.is_action_valid(i))
            arr = np.array(va) * 1
            return np.array(arr, dtype=np.int8)

    def reset(self):
        self.done = False
        self.current_y = 0
        self.S = np.zeros((self.height, self.width))
        self.depth = 0
        self.data = get_necessary_data(self.train_or_test)
        self.predictors = _get_predictors_from_data(self.data)
        return self.get_current_observation_state()


    def get_branching(self):
        assert self.done
        br = np.argmax(self.S[1:, :], axis=1)
        return br[br > 0].tolist()

    def get_current_observation_state(self):
        return {"state": self.S.reshape(-1), "predictors": self.predictors}




