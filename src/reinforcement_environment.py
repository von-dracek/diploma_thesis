"""
Implementation of the environment
"""
import os
import shutil
from typing import Callable, List

import gym
import numpy as np
from gams import GamsWorkspace
from gym import spaces

from src.configuration import (
    MAX_NUMBER_LEAVES_IN_SCENARIO_TREE,
    MIN_NUMBER_LEAVES_IN_SCENARIO_TREE,
)
from src.utils import get_cvar_value, get_necessary_data

gams_tmpdir = "C:/Users/crash/Documents/Programming/diploma_thesis_merged/diploma_thesis_v2/gams_workdir/"

# possible branching - width - 3 - 7
# depth - 3 - 5


def delete_files_in_temp_directory(seed):
    direc = gams_tmpdir + str(seed)
    for filename in os.listdir(direc):
        file_path = os.path.join(direc, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print("Failed to delete %s. Reason: %s" % (file_path, e))


def _get_predictors_from_data(
    data, random_number_generator: np.random.default_rng, defined_alpha: float = None
):
    """
    Get predictors that might be useful from the data.
    Particularly:
        alpha,
        number of stocks and
        statistics of returns over whole period.
    """
    predictors = np.zeros(9)
    alphas = np.array([0.8, 0.85, 0.9, 0.95])
    predictors[0] = random_number_generator.choice(
        alphas
    )  # randomly sample alpha 0.8 - 0.95
    if defined_alpha is not None:
        predictors[0] = defined_alpha  # alpha
    print(f"Value of alpha is = {predictors[0]}")
    predictors[1] = len(data.columns)  # n_stocks
    returns = data.iloc[[0, -1], :].pct_change().iloc[-1] + 1
    predictors[2] = max(returns)  # maximal return over whole period
    predictors[3] = min(returns)  # minimal return over the whole period
    predictors[4] = returns.quantile(0.75)
    predictors[5] = returns.quantile(0.5)
    predictors[6] = returns.quantile(0.25)
    predictors[7] = returns.mean()
    predictors[8] = returns.var()
    return predictors


def _reward_func_pretraining(branching, data, alpha, train_or_test):
    raise NotImplementedError
    # return valid_action_reward


penalty_func_none = lambda x: 0
# quadratic penalty - 0 at 300 scenarios, 0.1 at 100 and 700 scenarios
penalty_func_quadratic = lambda x: (0.1 / (550**2)) * ((x - 650)) ** 2
penalty_func_linear = (
    lambda x: 0.1
    * (x - MIN_NUMBER_LEAVES_IN_SCENARIO_TREE)
    / (MAX_NUMBER_LEAVES_IN_SCENARIO_TREE - MIN_NUMBER_LEAVES_IN_SCENARIO_TREE)
)


def _reward_func_v2(gams_workspace, branching, data, alpha, penalty_func):
    """Calculates reward from chosen branching."""
    # the minus sign before the cvar value has to be added, since we
    # calculate cvar-alpha of the loss distribution - thus the smaller the
    # cvar value the better. But the reinforcement agent in fact maximises
    # rewards - thus we need to give it a positive reward such that
    # higher value -> better reward and that is exactly what the
    # minus sign does here
    num_scenarios = np.prod(branching)
    penalty_for_complexity = penalty_func(num_scenarios)

    value = get_cvar_value(gams_workspace, branching, data, alpha)

    return -value - penalty_for_complexity


class TreeBuildingEnv(gym.Env):
    """
    Adaptation of gridworld environment for tree building purposes.
    First choosing depth of tree, then each step chooses next branching.
    Reward is returned when the tree is completely built.
    The observation space is a 8*8 table, where the first row represents possible
    depths of tree (only valid are 3-5), and successive rows represent
    the brachings in each level (again only valid 3-7).
    """

    def __init__(
        self,
        reward_fn: Callable,
        train_or_test: str = "train",
        defined_tickers: List[str] = None,
        defined_alpha: float = None,
        train_or_test_time: str = "train",
        penalty_func: Callable = penalty_func_none,
    ):
        super(TreeBuildingEnv, self).__init__()

        self.train_or_test = train_or_test
        self.train_or_test_time = train_or_test_time
        self.height = 8
        if penalty_func is None:
            self.penalty_func = penalty_func_none
        else:
            self.penalty_func = penalty_func
        # first level in height is chosen depth of tree
        # next levels represent branching at each stage
        self.width = 8
        self.observation_space = spaces.Dict(
            state=spaces.MultiDiscrete([2] * self.height * self.width),
            predictors=spaces.Box(shape=(9,), low=0, high=1000),
        )
        self.reward_fn = reward_fn
        self.defined_tickers = defined_tickers
        # defined_tickers is a list of tickers to choose - do not apply random sampling, always choosing this set
        # not used when training agent - only used for graphs
        self.defined_alpha = (
            defined_alpha  # if you wish to fix alpha, use this paramater
        )

    def seed(self, seed):
        if seed is not None:
            self._seed = seed
            print(f"random seed is {seed} \n")
            self.random_generator = np.random.default_rng(seed)
            self.gams_workspace = GamsWorkspace(
                system_directory=r"C:\GAMS\40",
                working_directory=gams_tmpdir + str(seed),
            )
        else:
            return self._seed

    def step(self, action):
        """
        Takes a step in the environment according to a chosen action.
        If invalid depth of tree is chosen, episode is ended with large negative reward.
        If invalid branching is chosen (according to maximum number of scenarios),
        the agent is forced to take the maximum possible branching in current stage
        """
        if action not in self.action_space:
            raise NotImplementedError
        action = action + 3
        if self.done:
            raise NotImplementedError
        valid_actions = self.valid_actions()
        if valid_actions[action] == 0:  # checks if action is valid
            amax = np.argmax(valid_actions[::-1])
            action = len(valid_actions) - amax - 1
            assert action > 2 and action < 8
        x = action
        y = self.current_y
        if y == 0:  # first assignment chooses depth of tree
            self.depth = x
        self.current_y += 1
        self.S[y, x] = 1
        if (
            self.current_y == np.argmax(self.S[0, :]) + 1
        ):  # all branchings have been chosen - end episode
            self.done = True
            reward = self.reward_fn(
                self.gams_workspace,
                self.get_branching(),
                self.data,
                self.predictors[0],
                self.penalty_func,
            )
            print(f"Done, got reward {reward}")
            return (
                self.get_current_observation_state(),
                reward,
                self.done,
                {
                    "num_scen": self.current_num_scenarios,
                    "terminal_observation": self.get_current_observation_state(),
                },
            )
        reward = 0
        return self.get_current_observation_state(), reward, self.done, {}

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
        return spaces.Discrete(5)
        # return spaces.Discrete(9)

    def is_action_valid(self, action):
        if action < 3 or action >= 8:  # in other states permit only actions 3 - 7
            return False
        if (
            (self.action_space is not None)
            and (self.depth > 0)
            and self.remaining_depth > 0
        ):
            current_num_scenarios = self.current_num_scenarios
            if (
                current_num_scenarios * action * (3 ** (self.remaining_depth - 1))
                < MAX_NUMBER_LEAVES_IN_SCENARIO_TREE
            ) and (
                current_num_scenarios * action * (7 ** (self.remaining_depth - 1))
                > MIN_NUMBER_LEAVES_IN_SCENARIO_TREE
            ):
                return True
        return False

    def valid_actions(self):
        if self.depth == 0:
            va = [False, False, False] + ([True] * 3) + ([False] * 2)
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
        self.data = get_necessary_data(
            self.random_generator,
            self.train_or_test,
            self.defined_tickers,
            self.train_or_test_time,
        )
        self.predictors = _get_predictors_from_data(
            self.data, self.random_generator, self.defined_alpha
        )
        delete_files_in_temp_directory(self._seed)

        return self.get_current_observation_state()

    def get_branching(self):
        assert self.done
        br = np.argmax(self.S[1:, :], axis=1)
        return br[br > 0].tolist()

    def get_current_observation_state(self):
        return {"state": self.S.reshape(-1), "predictors": self.predictors}


#
# class DepthChoosingEnv(gym.Env):
#     """
#     Adaptation of gridworld environment for tree building purposes.
#     First choosing depth of tree, then each step chooses next branching.
#     Reward is returned when the tree is completely built.
#     The observation space is a 8*8 table, where the first row represents possible
#     depths of tree (only valid are 3-7), and successive rows represent
#     the brachings in each level (again only valid 3-7).
#     """
#
#     def __init__(self, reward_fn: Callable, train_or_test:str = "train", defined_tickers:List[str]=None, defined_alpha:float=None, evaluate:bool = None):
#         super(DepthChoosingEnv, self).__init__()
#         self.train_or_test = train_or_test
#         self.height = 1
#         # first level in height is chosen depth of tree
#         # next levels represent branching at each stage
#         self.width = 8
#         # self.observation_space = spaces.Dict(state=spaces.MultiDiscrete([2]*self.height*self.width), predictors=spaces.Box(shape=(9,), low=0, high=1000))
#         self.observation_space = spaces.Box(shape=(9,), low=0, high=1000)
#         self.reward_fn = reward_fn
#         self.defined_tickers = defined_tickers
#         #defined_tickers is a list of tickers to choose - do not apply random sampling, always choosing this set
#         #not used when training agent - only used for graphs
#         self.defined_alpha = defined_alpha #if you wish to fix alpha, use this paramater
#         self.valid_action_reward = 0.1 if evaluate is None else 0
#         self.invalid_action_reward = -self.valid_action_reward
#         # self.reset()
#         # begin in start state
#
#     def seed(self, seed):
#         if seed is not None:
#             self._seed = seed
#             print(f"random seed is {seed} \n")
#             self.random_generator = np.random.default_rng(seed)
#             self.gams_workspace = GamsWorkspace(system_directory=r"C:\GAMS\40", working_directory=gams_tmpdir + str(seed))
#         else:
#             return self._seed
#
#     def step(self, action):
#         """
#         Takes a step in the environment according to a chosen action.
#         If invalid depth of tree is chosen, episode is ended with large negative reward.
#         If invalid branching is chosen (according to maximum number of scenarios),
#         the agent is forced to take the maximum possible branching in current stage and
#         receives small negative reward.
#         """
#         if action not in self.action_space:
#             raise NotImplementedError
#         action = action + 3
#         if self.done:
#             raise NotImplementedError
#         x = action
#         self.S[0, x] = 1
#         self.done = True
#         reward = self.reward_fn(self.gams_workspace, self.get_branching(), self.data, self.predictors[0], self.train_or_test)
#         print(f"Done, got reward {reward}")
#         return self.get_current_observation_state(), reward, self.done, {"terminal_observation":self.get_current_observation_state()}
#
#     @property
#     def action_space(self):
#         return spaces.Discrete(3)
#
#
#     def reset(self):
#         self.done = False
#         self.S = np.zeros((self.height, self.width))
#         self.data = get_necessary_data(self.train_or_test, self.defined_tickers)
#         self.predictors = _get_predictors_from_data(self.data, self.random_generator, self.defined_alpha)
#
#         return self.get_current_observation_state()
#
#     def get_current_observation_state(self):
#         return np.array(self.predictors)
#
#     def get_branching(self):
#         action = np.argmax(self.S)
#         if action == 3:
#             return [5] * action #125
#         elif action == 4:
#             return [4,4,4,3] #192
#         else:
#             return [3] * np.argmax(self.S)
