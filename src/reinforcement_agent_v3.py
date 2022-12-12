import os
from typing import List, Callable

from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import load_results, Monitor
from stable_baselines3.common.results_plotter import ts2xy
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize, VecEnv, DummyVecEnv
from src.reinforcement_environment import TreeBuildingEnv, _reward_func_v2, penalty_func_quadratic, penalty_func_linear
from src.configuration import ASSET_SET_1
from stable_baselines3 import PPO
import torch as th
from datetime import datetime
import numpy as np
from src.CustomVecMonitor import CustomVecMonitor

#tensorboard --logdir C:\Users\crash\Documents\Programming\diploma_thesis_merged\diploma_thesis_v2\tensorboard_logging\gym


class SaveOnBestTrainingRewardCallback(BaseCallback):
    """
    Callback for saving a model (the check is done every ``check_freq`` steps)
    based on the training reward (in practice, we recommend using ``EvalCallback``).

    :param check_freq: (int)
    :param log_dir: (str) Path to the folder where the model will be saved.
      It must contains the file created by the ``Monitor`` wrapper.
    :param verbose: (int)
    """
    def __init__(self, check_freq: int, log_dir: str, model_name: str, verbose=1):
        super(SaveOnBestTrainingRewardCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, 'best_model')
        self.best_mean_reward = -np.inf

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:

          # Retrieve training reward
          x, y = ts2xy(load_results(self.log_dir), 'timesteps')
          if len(x) > 0:
              # Mean training reward over the last 250 episodes
              mean_reward = np.mean(y[-250:])
              if self.verbose > 0:
                print(f"Num timesteps: {self.num_timesteps}")
                print(f"Best mean reward: {self.best_mean_reward:.2f} - Last mean reward per episode: {mean_reward:.2f}")
              self.logger.record(f"Mean reward over last 250 episodes", mean_reward)

              current_time = datetime.now().strftime("%Y-%m-%d %H,%M,%S")
              # New best model, you could save the agent here
              if mean_reward > self.best_mean_reward:
                  self.best_mean_reward = mean_reward
                  # Example for saving best model
                  if self.verbose > 0:
                    print(f"Saving new best model after {self.n_calls} calls to {self.save_path}_{self.n_calls}, {current_time}.zip")
                  self.model.save(self.save_path + f"_{self.n_calls}, {current_time}")
              # self.model.save(os.path.join(self.log_dir, 'checkpoint') + f"_{self.n_calls}, {current_time}")
              # self.model.env.save(os.path.join(self.log_dir, 'checkpoint_env') + f"_{self.n_calls}, {current_time}")
              self.model.save(os.path.join(self.log_dir, 'latest'))
              self.model.env.save(os.path.join(self.log_dir, 'latest_env'))

        return True


def make_treebuilding_env(defined_tickers:List[str]=None, defined_alpha:float=None, train_or_test_time:str = None, train_or_test:str = None, penalty_func:Callable = None):
    print(f"Making env with {train_or_test_time=}, {train_or_test=}")
    return lambda : TreeBuildingEnv(_reward_func_v2, train_or_test, defined_tickers, defined_alpha, train_or_test_time, penalty_func)


if __name__ == '__main__':

    log_dir = "./tensorboard_logging/gym/"
    os.makedirs(log_dir, exist_ok=True)

    # start_training = True
    num_neurons = 128


    current_time = datetime.now().strftime("%Y-%m-%d %H,%M,%S")
    tensorboard_log = log_dir


    # if start_training:

    # Instantiate the env
    env = make_treebuilding_env(train_or_test="train", train_or_test_time="train", penalty_func=penalty_func_linear)
    # venv = DummyVecEnv(env_fns=[env] * 1)
    venv = SubprocVecEnv(env_fns=[env] * 6)
    venv = CustomVecMonitor(venv, log_dir, info_keywords=("num_scen",))
    env = VecNormalize(venv, norm_obs_keys=["predictors"], clip_obs=1000, clip_reward=1000, norm_reward=False,
                       norm_obs=True)
    env.seed(1337)

    #not normalising advantage - should have no effect, see https://github.com/DLR-RM/stable-baselines3/issues/485
    ppo_model = PPO(policy="MultiInputPolicy",
                    learning_rate=1e-3,
                    env=env,
                    n_steps=24*8,
                    policy_kwargs=dict(activation_fn=th.nn.ReLU, net_arch=[num_neurons*2, dict(vf=[num_neurons], pi=[num_neurons])]),
                    tensorboard_log=tensorboard_log,
                    seed=1337,
                    gamma=1)
    ppo_model.learn(100000, callback=SaveOnBestTrainingRewardCallback(check_freq=10, log_dir=log_dir, model_name=f"PPO_{num_neurons * 2}_{num_neurons}_neurons_no_penalty"), progress_bar=True, tb_log_name=f"PPO_{num_neurons * 2}_{num_neurons}_neurons_no_penalty", log_interval=1)

    # else:
    #     ppo_model = PPO.load(os.path.join(log_dir, 'latest.zip'))
    #     env = make_treebuilding_env(train_or_test="train", train_or_test_time="train")
    #     venv = SubprocVecEnv(env_fns=[env] * 6)
    #     venv = CustomVecMonitor(venv, log_dir, info_keywords=("num_scen",))
    #     env = VecNormalize.load(os.path.join(log_dir, 'latest_env'), venv=venv)
    #     env.seed(1234)
    #     ppo_model.set_env(env)
    #     ppo_model.learn(200000, callback=SaveOnBestTrainingRewardCallback(check_freq=10, log_dir=log_dir, model_name=f"A2C_{num_neurons * 2}_{num_neurons}_neurons"), progress_bar=True, tb_log_name=f"A2C_{num_neurons * 2}_{num_neurons}_neurons", log_interval=1, reset_num_timesteps=False)