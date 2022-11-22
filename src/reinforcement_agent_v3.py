import os
from typing import List

from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import load_results
from stable_baselines3.common.results_plotter import ts2xy

from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from src.reinforcement_environment import TreeBuildingEnv, _reward_func_pretraining, _reward_func_v2
from stable_baselines3 import A2C
import torch as th
from datetime import datetime
import numpy as np

from src.CustomVecMonitor import CustomVecMonitor


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
              # Mean training reward over the last 100 episodes
              mean_reward = np.mean(y[-100:])
              if self.verbose > 0:
                print(f"Num timesteps: {self.num_timesteps}")
                print(f"Best mean reward: {self.best_mean_reward:.2f} - Last mean reward per episode: {mean_reward:.2f}")
              self.logger.record(f"Mean reward over last 100 episodes", mean_reward)

              global current_time
              # New best model, you could save the agent here
              if mean_reward > self.best_mean_reward:
                  self.best_mean_reward = mean_reward
                  # Example for saving best model
                  if self.verbose > 0:
                    print(f"Saving new best model after {self.n_calls} calls to {self.save_path}_{self.n_calls}, {current_time}.zip")
                  self.model.save(self.save_path + f"_{self.n_calls}, {current_time}")
              self.model.save(os.path.join(self.log_dir, 'checkpoint') + f"_{self.n_calls}, {current_time}")
              self.model.env.save(os.path.join(self.log_dir, 'checkpoint_env') + f"_{self.n_calls}, {current_time}")
              self.model.save(os.path.join(self.log_dir, 'latest'))
              self.model.env.save(os.path.join(self.log_dir, 'latest_env'))

              # print("Logging mean reward to tensorboard")


        return True


def make_pretraining_env():
    raise NotImplementedError
    # return TreeBuildingEnv(_reward_func_pretraining)

def make_training_env(defined_tickers:List[str]=None, defined_alpha:float=None):
    return TreeBuildingEnv(_reward_func_v2, "train", defined_tickers, defined_alpha)

def make_testing_env(defined_tickers:List[str]=None, defined_alpha:float=None):
    return TreeBuildingEnv(_reward_func_v2, "test", defined_tickers, defined_alpha)


if __name__ == '__main__':

    log_dir = "./tensorboard_logging/gym/"
    os.makedirs(log_dir, exist_ok=True)

    start_training=False
    num_neurons = 64


    current_time = datetime.now().strftime("%Y-%m-%d %H,%M,%S")
    tensorboard_log = log_dir

    if start_training:

        # Instantiate the env
        env = make_training_env
        venv = SubprocVecEnv(env_fns=[env] * 4)
        venv = CustomVecMonitor(venv, log_dir, info_keywords=("num_scen",))

        env = VecNormalize(venv, norm_obs_keys=["predictors"])
        env.seed(1337)

        #not normalising advantage - should have no effect, see https://github.com/DLR-RM/stable-baselines3/issues/485
        a2c_model = A2C(policy="MultiInputPolicy", learning_rate=1e-4, use_rms_prop=True, env=env, n_steps=5, policy_kwargs=dict(activation_fn=th.nn.ReLU, net_arch=[num_neurons*2, dict(vf=[num_neurons], pi=[num_neurons])]), tensorboard_log=tensorboard_log, seed=1337)
        a2c_model.learn(1000, callback=SaveOnBestTrainingRewardCallback(check_freq=20, log_dir=log_dir, model_name=f"A2C_{num_neurons*2}_{num_neurons}_neurons"), progress_bar=True, tb_log_name=f"A2C_{num_neurons*2}_{num_neurons}_neurons", log_interval=4)
    else:
        a2c_model = A2C.load(os.path.join(log_dir, 'latest.zip'))
        env = make_training_env
        venv = SubprocVecEnv(env_fns=[env] * 4)
        venv = CustomVecMonitor(venv, log_dir, info_keywords=("num_scen",))
        env = VecNormalize.load(os.path.join(log_dir, 'latest_env'), venv=venv)
        a2c_model.set_env(env)
        a2c_model.learn(30000, callback=SaveOnBestTrainingRewardCallback(check_freq=20, log_dir=log_dir,
                                                                        model_name=f"A2C_{num_neurons * 2}_{num_neurons}_neurons"),
                        progress_bar=True, tb_log_name=f"A2C_{num_neurons * 2}_{num_neurons}_neurons", log_interval=4,
                        reset_num_timesteps=False)