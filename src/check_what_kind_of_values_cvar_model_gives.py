import logging
import pickle
import pandas as pd
import plotly.express as px

from src.reinforcement_environment import TreeBuildingEnv
from datetime import datetime

current_time = datetime.now().strftime("%Y-%m-%d %H,%M,%S")
perform_experiment = True
if __name__ == '__main__':
    if perform_experiment:
        env = TreeBuildingEnv
        env = env()
        env.seed(1337)
        rewards = []
        i = 0
        count_to = 100
        while i < count_to:
            logging.info(f"Calculating tree {i} out of {count_to}")
            env.reset()
            done = False
            while not done:
                action = env.action_space.sample()
                _, reward, done, _ = env.step(action)
            if not (reward == -10):
                i += 1
                rewards.append((env.current_num_scenarios,reward, env.get_branching()))



        with open(f'reward_distribution_from_mean_cvar_{current_time}.pickle', 'wb') as handle:
            pickle.dump(rewards, handle, protocol=pickle.HIGHEST_PROTOCOL)

        print(rewards)

    with open(f'reward_distribution_from_mean_cvar_{current_time}.pickle', 'rb') as handle:
        data = pickle.load(handle)
        data = pd.DataFrame(data)
        fig = px.scatter(data, x=0, y=1, title="Scatterplot of generated rewards", template="simple_white")
        fig.update_layout(
            xaxis_title="# scenarios", yaxis_title="Reward Value"
        )
        fig.write_html(f"scatterplot_scenario_tree_generated_rewards_{current_time}.html")

