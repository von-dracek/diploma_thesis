import os
from typing import List
import logging
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import load_results
from stable_baselines3.common.results_plotter import ts2xy
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize, VecEnv, DummyVecEnv
from src.reinforcement_environment import TreeBuildingEnv, _reward_func_pretraining, _reward_func_v2
from src.configuration import ASSET_SET_1
from stable_baselines3 import A2C
import torch as th
from datetime import datetime
import numpy as np

from src.CustomVecMonitor import CustomVecMonitor
from src.reinforcement_agent_v3 import make_training_env, make_testing_env



if __name__ == '__main__':

    log_dir = "./tensorboard_logging/gym/"
    os.makedirs(log_dir, exist_ok=True)

    start_training=True
    n_eval_episodes = 20

    current_time = datetime.now().strftime("%Y-%m-%d %H,%M,%S")
    tensorboard_log = log_dir

    env = make_training_env(defined_tickers=ASSET_SET_1,evaluate=True)
    venv = DummyVecEnv(env_fns=[env] * 1)
    venv = CustomVecMonitor(venv, log_dir, info_keywords=("num_scen",))
    env = VecNormalize.load(os.path.join(log_dir, 'latest_env'), venv=venv)
    env.reset()
    a2c_model = A2C.load(os.path.join(log_dir, 'latest.zip'), env=env)

    # a2c_model.set_env(env)
    a2c_trained_reward = evaluate_policy(model=a2c_model, env=env, n_eval_episodes=n_eval_episodes, return_episode_rewards=True)
    rewards_a2c = env.unnormalize_reward(np.array(a2c_trained_reward[0]))
    logging.info(rewards_a2c)
    logging.info(f"Loss of a2c is {rewards_a2c.mean()}")
    pass
    # logging.info("Starting random agent")
    #
    # rewards = []
    # for episode in range(n_eval_episodes):
    #     done = False
    #     while not done:
    #         action = env.action_space.sample()
    #         action = np.array([action])
    #         obs, rew, done, info = env.step(action)
    #     env.reset()
    #     rew = env.unnormalize_reward(rew)
    #     rew = rew[0]
    #     rewards.append(rew)
    #     print(rewards)



    logging.info("Starting random agent")

    rewards = []
    for episode in range(n_eval_episodes):
        done = False
        while not done:
            action = env.action_space.sample()
            action = np.array([action])
            obs, rew, done, info = env.step(action)
        env.reset()
        rewards.append(rew[0])
        print(rewards)

    rewards_random = env.unnormalize_reward(np.array(rewards))

    logging.info(f"Reward of random agent is {rewards_random.mean()}")
    logging.info(f"Reward of a2c trained agent is {rewards_a2c.mean()}")

    # random_agent = A2C(policy="MultiInputPolicy", learning_rate=1e-4, use_rms_prop=True, env=env)
    # random_agent_reward = evaluate_policy(model=random_agent, env=env, n_eval_episodes=n_eval_episodes)
    # logging.info(f"Reward of a2c is {a2c_trained_reward}")
    # logging.info(f"Reward of random agent is {random_agent_reward}")
