import gym
import numpy as np
from gym import spaces
import tensorflow as tf
# gridworld
# possible branching - width - 3 - 8
# depth - 2 - 8

minimum_branching = 3
# maximum branching = 8
max_number_of_scenarios = 1000
reward = 0  # TODO calculate reward


def _reward_func(input):
    if input == 3:
        return 100
    else:
        return 0


class TreeBuildingEnv(gym.Env):
    def __init__(self):
        self.height = 9
        # first level in height is chosen depth of tree
        # next levels represent branching at each stage
        self.width = 9
        self.observation_space = spaces.Tuple(
            (spaces.Discrete(self.height), spaces.Discrete(self.width))
        )
        self.reset()
        # begin in start state

    def step(self, action):
        if action not in self.action_space:
            raise NotImplementedError
        if self.done:
            raise NotImplementedError
        x = action
        y = self.current_y
        self.current_y += 1
        self.S[y, x] = 1
        if y == 0: #first assignment chooses depth of tree
            self.depth = x
        if self.current_y > 0:
            if self.current_y == np.argmax(self.S[0, :]) + 1:
                self.done = True
                return self.S.reshape(1,-1), _reward_func(self.depth), self.done, {}
        return self.S.reshape(1,-1), -0.01, self.done, {}

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
        return spaces.Discrete(7, start=2)
    # @property
    # def action_space(self):
    #     if self.depth > 0:
    #         current_num_scenarios = self.current_num_scenarios
    #         if self.remaining_depth == 0:
    #             return None
    #         act_space_full = range(9)
    #         valid_num_list = []
    #         for number in act_space_full:
    #             if (
    #                 current_num_scenarios * number * (3 ** (self.remaining_depth - 1))
    #                 < max_number_of_scenarios
    #             ):
    #                 valid_num_list.append(number)
    #         start = max(min(valid_num_list), 3)
    #         length = min(len(valid_num_list) - 3, 6)
    #         act_space = spaces.Discrete(length, start=start)
    #         return act_space
    #     else:
    #         return spaces.Discrete(7, start=2)

    def is_action_valid(self, action):
        if action < 3 or action > 9: #in other states permit only actions 3 - 8
            return False
        if (self.action_space is not None) and (self.depth > 0) and self.remaining_depth > 0:
            current_num_scenarios = self.current_num_scenarios
            if (
                current_num_scenarios * action * (3 ** (self.remaining_depth - 1))
                < max_number_of_scenarios
            ):
                return True
        return False

    def valid_actions(self):
        if self.depth == 0:
            arr = np.array([0] + [1]*6, dtype=np.int8)
            return arr
        else:
            va = [False]
            for i in range(3, self.width):
                va.append(self.is_action_valid(i))
            arr = np.array(va) * 1
            return np.array(arr, dtype=np.int8)

    def reset(self):
        self.done = False
        self.current_y = 0
        self.S = np.zeros((self.height, self.width))
        self.depth = 0
        return self.S.reshape(1,-1), {}


    def get_branching(self):
        assert self.done
        br = np.argmax(self.S[1:, :], axis=1)
        return br[br > 0]


# env = TreeBuildingEnv()
# done = False

# next_state, reward, done, info = env.step(5)
# while not done:
#     valid_actions = env.valid_actions()
#     next_state, reward, done, info = env.step(env.action_space.sample(valid_actions))

# env.valid_actions()
# env.valid_actions()
# env.valid_actions()
# env.valid_actions()
# env.action_space.sample()
#
# # next_state, reward, done, info = env.step(env.action_space.sample())
#
# assert env.current_num_scenarios < 1000
# branching = env.get_branching()
# va = env.valid_actions()
