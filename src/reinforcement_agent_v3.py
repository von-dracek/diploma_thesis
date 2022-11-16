import os
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import load_results
from stable_baselines3.common.results_plotter import ts2xy

from stable_baselines3.common.vec_env import SubprocVecEnv
from reinforcement_environment import TreeBuildingEnv_v4
from stable_baselines3 import A2C

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

              # New best model, you could save the agent here
              if mean_reward > self.best_mean_reward:
                  self.best_mean_reward = mean_reward
                  # Example for saving best model
                  global current_time
                  if self.verbose > 0:
                    print(f"Saving new best model after {self.n_calls} calls to {self.save_path}_{self.n_calls}, {current_time}.zip")
                  self.model.save(self.save_path + f"_{self.n_calls}, {current_time}")
              # print("Logging mean reward to tensorboard")


        return True


if __name__ == '__main__':

    log_dir = "./tensorboard_logging/gym/"
    os.makedirs(log_dir, exist_ok=True)

    # Instantiate the env
    env = TreeBuildingEnv_v4
    env = SubprocVecEnv(env_fns=[env]*6)
    env.seed(1337)
    env = CustomVecMonitor(env, log_dir, info_keywords=("num_scen",))
    current_time = datetime.now().strftime("%Y-%m-%d %H,%M,%S")

    tensorboard_log=log_dir

    a2c_model = A2C(policy="MultiInputPolicy", learning_rate=4e-4, use_rms_prop=True, normalize_advantage=True, env=env, n_steps=5, policy_kwargs=dict(net_arch=[2048, 1024, 512]), tensorboard_log=tensorboard_log)
    a2c_model.learn(10**9, callback=SaveOnBestTrainingRewardCallback(check_freq=20, log_dir=log_dir, model_name="A2C"), progress_bar=True, log_interval=4)



    # dqn_model = DQN("MlpPolicy",
    #             env,
    #             verbose=1,
    #             train_freq=32,
    #             gradient_steps=8,
    #             gamma=0.99,
    #             exploration_fraction=1,
    #             exploration_final_eps=0.01,
    #             target_update_interval=600,
    #             learning_starts=10000,
    #             batch_size=128,
    #             learning_rate=4e-3,
    #             policy_kwargs=dict(net_arch=[48, 24, 12]),
    #             tensorboard_log=tensorboard_log,
    #             seed=2)
    #
    #
    # dqn_model.learn(100000, callback=callback, progress_bar=True, log_interval=100)

    # ddqn_model = DQN("MlpPolicy",
    #             env,
    #             verbose=1,
    #             train_freq=32,
    #             gamma=0.99,
    #             exploration_fraction=0.5,
    #             exploration_final_eps=0.01,
    #             learning_starts=10000,
    #             batch_size=128,
    #             learning_rate=1e-4,
    #             policy_kwargs=dict(net_arch=[48, 12]),
    #             tensorboard_log=tensorboard_log,
    #             target_update_interval=100,
    #             seed=2)
    #
    #
    # ddqn_model.learn(100000, callback=callback,log_interval=100)

