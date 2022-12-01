import os
import pickle
from typing import List
import logging
import pandas as pd
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize, VecEnv, DummyVecEnv, VecMonitor
from src.reinforcement_environment import TreeBuildingEnv, _reward_func_pretraining, _reward_func_v2, \
    penalty_func_quadratic
from src.configuration import ASSET_SET_1
from stable_baselines3 import A2C, PPO
import torch as th
from datetime import datetime
import numpy as np
from plotly import graph_objects as go
from src.CustomVecMonitor import CustomVecMonitor
from src.reinforcement_agent_v3 import make_treebuilding_env
import plotly.express as px

dir_quaadratic_penalty = "./tensorboard_logging/gym_final_with_quadratic_penalty/agent_evaluation/"
dir_no_penalty = "./tensorboard_logging/gym_final_final_final_without_penalty/agent_evaluation/"

#choose directory here
dir = dir_quaadratic_penalty

if __name__ == '__main__':

    log_dir = dir
    os.makedirs(log_dir, exist_ok=True)

    n_eval_episodes = 500
    start_evaluation = False

    for train_or_test_tickers in ["train", "test"]:
        for train_or_test_time in ["train", "test"]:
            log_dir = dir

            if start_evaluation:
                current_time = datetime.now().strftime("%Y-%m-%d %H,%M,%S")
                tensorboard_log = log_dir
                env = make_treebuilding_env(train_or_test=train_or_test_tickers, train_or_test_time=train_or_test_time, penalty_func=penalty_func_quadratic if dir == dir_quaadratic_penalty else None)
                venv = DummyVecEnv(env_fns=[env] * 1)
                venv = VecMonitor(venv, log_dir, info_keywords=("num_scen",))
                env = VecNormalize.load(os.path.join(log_dir, 'latest_env'), venv=venv)
                env.seed(1337)
                # np.random.seed(1337)
                ppo_model = PPO.load(os.path.join(log_dir, 'latest.zip'), env=env)

                run_ppo = True
                if run_ppo:
                    final_ppo = []
                    for episode in range(n_eval_episodes):
                        print(f"Running episode {episode} out of {n_eval_episodes} PPO")
                        done = False
                        obs = env.reset()
                        while not done:
                            action = ppo_model.predict(obs)[0][0]
                            action = np.array([action])
                            obs, rew, done, info = env.step(action)
                        final_ppo.append({"obs":env.unnormalize_obs(obs), "rew":env.unnormalize_reward(rew), "done":done, "info":info})

                    rewards_ppo = [x["rew"][0] for x in final_ppo]
                    rewards_ppo = np.array(rewards_ppo)
                    logging.info(rewards_ppo)
                    logging.info(f"Loss of ppo is {rewards_ppo.mean()}")
                    predictors_ppo = [x["obs"]["predictors"] for x in final_ppo]


                logging.info("Starting random agent")

                env = make_treebuilding_env(train_or_test=train_or_test_tickers, train_or_test_time=train_or_test_time,  penalty_func=penalty_func_quadratic if dir == dir_quaadratic_penalty else None)
                venv = DummyVecEnv(env_fns=[env] * 1)
                venv = VecMonitor(venv, log_dir, info_keywords=("num_scen",))
                env = VecNormalize.load(os.path.join(log_dir, 'latest_env'), venv=venv)
                env.seed(1337)
                np.random.seed(1337) #to make sampling action space deterministic
                env.action_space.np_random.seed(1337)


                final_random = []
                for episode in range(n_eval_episodes):
                    print(f"Running episode {episode} out of {n_eval_episodes} Random")
                    done = False
                    obs = env.reset()
                    while not done:
                        valid_actions = env.env_method("valid_actions")[0]
                        valid_actions = pd.DataFrame(valid_actions > 0)
                        valid_actions = valid_actions[valid_actions[0]]
                        valid_actions = list(valid_actions.index)
                        action = np.random.choice(valid_actions) - 3  #env.action_space.sample()
                        action = np.array([action])
                        obs, rew, done, info = env.step(action)
                    final_random.append({"obs":env.unnormalize_obs(obs), "rew":env.unnormalize_reward(rew), "done":done, "info":info})

                rewards_random = [x["rew"][0] for x in final_random]
                rewards_random = np.array(rewards_random)
                logging.info(rewards_random)
                logging.info(f"Loss of random is {rewards_random.mean()}")

                #check evaluation is correct
                predictors_random = [x["obs"]["predictors"] for x in final_random]


                with open(f"{log_dir}ppo_rewards_before_assert_{train_or_test_tickers}_{train_or_test_time}_{current_time}.pckl", "wb") as f:
                    pickle.dump(final_ppo, f)

                with open(f"{log_dir}random_rewards_before_assert_{train_or_test_tickers}_{train_or_test_time}_{current_time}.pckl", "wb") as f:
                    pickle.dump(final_random, f)

                assert len(predictors_ppo) == len(predictors_random)
                assert all([np.allclose(a,b, rtol=1e-2, atol=1e-2) for a,b in zip(predictors_ppo, predictors_random)]), "predictors differ in evaluation between random agent and ppo"

                with open(f"{log_dir}ppo_rewards_evaluated_{train_or_test_tickers}_{train_or_test_time}_{current_time}.pckl", "wb") as f:
                    pickle.dump(final_ppo, f)

                with open(f"{log_dir}random_rewards_evaluated_{train_or_test_tickers}_{train_or_test_time}_{current_time}.pckl", "wb") as f:
                    pickle.dump(final_random, f)

                with open(f"{log_dir}ppo_rewards_evaluated_{train_or_test_tickers}_{train_or_test_time}_latest.pckl", "wb") as f:
                    pickle.dump(final_ppo, f)

                with open(f"{log_dir}random_rewards_evaluated_{train_or_test_tickers}_{train_or_test_time}_latest.pckl", "wb") as f:
                    pickle.dump(final_random, f)

                logging.info(f"Rewards of random agent array is {rewards_random}")
                logging.info(f"Rewards of ppo agent array is {rewards_ppo}")


                logging.info(f"Reward of random agent is {rewards_random.mean()}")
                logging.info(f"Reward of ppo trained agent is {rewards_ppo.mean()}")


            else:
                with open(f"{log_dir}ppo_rewards_evaluated_{train_or_test_tickers}_{train_or_test_time}_latest.pckl", "rb") as f:
                    ppo = pickle.load(f)
                with open(f"{log_dir}random_rewards_evaluated_{train_or_test_tickers}_{train_or_test_time}_latest.pckl", "rb") as f:
                    random = pickle.load(f)

                # #todo: delete these rows
                # with open(f"{log_dir}ppo_rewards_before_assert_train_test_2022-11-28 15,26,13.pckl", "rb") as f:
                #     ppo = pickle.load(f)
                # with open(f"{log_dir}random_rewards_before_assert_train_test_2022-11-28 15,26,13.pckl", "rb") as f:
                #     random = pickle.load(f)

                # predictors_ppo = [x["obs"]["predictors"] for x in ppo]
                # predictors_random = [x["obs"]["predictors"] for x in random_rewards]

                def parse_dataset(dataset):
                    return {"predictors":[x["obs"]["predictors"] for x in dataset],"rewards":[x["rew"][0] for x in dataset], "number_scenarios":[x["info"][0]["num_scen"] for x in dataset], "terminal_observations":[x["info"][0]["terminal_observation"] for x in dataset]}

                def terminal_observations_to_branching(terminal_observations):
                    states_reshaped = [x["state"].reshape(8,8) for x in terminal_observations]
                    branchings = [np.argmax(state[1:, :], axis=1) for state in states_reshaped]
                    branchings = [br[br > 0].tolist() for br in branchings]
                    return branchings


                ppo_parsed = parse_dataset(ppo)
                random_parsed = parse_dataset(random)
                ppo_parsed["branchings"] = terminal_observations_to_branching(ppo_parsed["terminal_observations"])
                random_parsed["branchings"] = terminal_observations_to_branching(random_parsed["terminal_observations"])

                # plot_titles=["Reward barplot train train", "Reward scatterplot train train"]
                log_dir = log_dir + f"graphs_{train_or_test_time}_{train_or_test_tickers}/"
                os.makedirs(log_dir, exist_ok=True)

                data_plot1 = pd.DataFrame({"reinforcement_agent":ppo_parsed["rewards"],
                                           "random_agent":random_parsed["rewards"],
                                           "random_agent_scenarios":random_parsed["number_scenarios"],
                                           "reinforcement_agent_scenarios":ppo_parsed["number_scenarios"],
                                          "reinforcement_agent_branchings":ppo_parsed["branchings"],
                                          "random_agent_branchings":random_parsed["branchings"]})

                assert all([np.allclose(a,b, rtol=1e-2, atol=1e-2) for a,b in zip(ppo_parsed['predictors'], random_parsed['predictors'])]), "predictors differ in evaluation between random agent and ppo"


                results_str = f"ppo_mean: {data_plot1['reinforcement_agent'].mean()} \n random_agent_mean: {data_plot1['random_agent'].mean()} "
                with open(f"{log_dir}_mean_results.txt", "w") as text_file:
                    text_file.write(results_str)

                fig = go.Figure()

                fig.add_trace(go.Box(fillcolor="white", marker=dict(color='blue', line=dict(width=2, color='DarkSlateGrey')),
                                     boxpoints="all", y=list(data_plot1["reinforcement_agent"]), name="Reinforcement Agent"),)
                fig.add_trace(go.Box(fillcolor="white", marker=dict(color='red', line=dict(width=2, color='DarkSlateGrey')),
                                     boxpoints="all", y=list(data_plot1["random_agent"]), name="Random Agent"))
                fig.update_layout(template="simple_white")

                fig.write_html(f"{log_dir}evaluation_1_{train_or_test_time}_{train_or_test_tickers}.html")
                fig.write_image(f"{log_dir}evaluation_1_{train_or_test_time}_{train_or_test_tickers}.pdf")


                fig = go.Figure() # make_subplots(rows=4, subplot_titles=plot_titles)



                fig.add_trace(go.Scatter(x=list(data_plot1["reinforcement_agent_scenarios"]),
                                         y=list(data_plot1["reinforcement_agent"]),
                                         marker=dict(color='blue', line=dict(width=2, color='DarkSlateGrey')),
                                         mode="markers", name="Reinforcement agent"))
                fig.add_trace(go.Scatter(x=list(data_plot1["random_agent_scenarios"]),
                                         y=list(data_plot1["random_agent"]),
                                         marker=dict(color='red', line=dict(width=2, color='DarkSlateGrey')),
                                         mode="markers", name="Random agent"),)
                fig.update_layout(template="simple_white")

                fig.write_html(f"{log_dir}evaluation_2_{train_or_test_time}_{train_or_test_tickers}.html")
                fig.write_image(f"{log_dir}evaluation_2_{train_or_test_time}_{train_or_test_tickers}.pdf")



                fig = go.Figure()  # make_subplots(rows=4, subplot_titles=plot_titles)



                fig.add_trace(go.Histogram(x=list(data_plot1["reinforcement_agent_scenarios"]),
                                         # y=list(data_plot1["reinforcement_agent"]),
                                         # marker=dict(color='blue', line=dict(width=2, color='DarkSlateGrey')),
                                         # mode="markers",
                                        name="Reinforcement agent", opacity=0.5, marker={"color":"blue"}),)
                fig.add_trace(go.Histogram(x=list(data_plot1["random_agent_scenarios"]),
                                         # y=list(data_plot1["random_agent"]),
                                         # marker=dict(color='red', line=dict(width=2, color='DarkSlateGrey')),
                                         # mode="markers",
                                        name="Random agent", opacity=0.5, marker={"color":"red"}),)
                fig.update_layout(barmode="overlay")
                fig.update_layout(template="simple_white")

                fig.write_html(f"{log_dir}evaluation_3_{train_or_test_time}_{train_or_test_tickers}.html")
                fig.write_image(f"{log_dir}evaluation_3_{train_or_test_time}_{train_or_test_tickers}.pdf")


                fig = go.Figure()  # make_subplots(rows=4, subplot_titles=plot_titles)



                bar_data_reinforcement_agent = data_plot1["reinforcement_agent_branchings"].apply(len).value_counts()
                fig.add_trace(go.Bar(x=list(bar_data_reinforcement_agent.index),
                                     y=bar_data_reinforcement_agent,
                                         # y=list(data_plot1["reinforcement_agent"]),
                                         # marker=dict(color='blue', line=dict(width=2, color='DarkSlateGrey')),
                                         # mode="markers",
                                        name="Reinforcement agent", opacity=0.5, marker={"color":"blue"}),)
                bar_data_random = data_plot1["random_agent_branchings"].apply(len).value_counts()
                fig.add_trace(go.Bar(x=list(bar_data_random.index),
                                     y=bar_data_random,
                                         # marker=dict(color='red', line=dict(width=2, color='DarkSlateGrey')),
                                         # mode="markers",
                                        name="Random agent", opacity=0.5, marker={"color":"red"}),)
                # fig.update_layout(barmode="overlay")
                fig.update_layout(template="simple_white")

                fig.write_html(f"{log_dir}evaluation_4_{train_or_test_time}_{train_or_test_tickers}.html")
                fig.write_image(f"{log_dir}evaluation_4_{train_or_test_time}_{train_or_test_tickers}.pdf")

                data_depth_boxplots = pd.DataFrame(pd.concat([data_plot1["random_agent"], data_plot1["reinforcement_agent"]]))
                data_depth_boxplots.columns = ["reward"]
                data_depth_boxplots["color_by"]= pd.concat([data_plot1["random_agent_scenarios"]*0, data_plot1["reinforcement_agent_scenarios"]*1]).apply(lambda x: "Random agent" if x == 0 else "Reinforcement agent")
                depths = pd.concat([data_plot1["random_agent_branchings"], data_plot1["reinforcement_agent_branchings"]]).apply(len)
                data_depth_boxplots["depth"] = depths
                fig = px.box(data_depth_boxplots, x="depth", y="reward", color="color_by", color_discrete_sequence=[ "red", "blue"])

                fig.update_layout(template="simple_white")

                fig.write_html(f"{log_dir}evaluation_5_{train_or_test_time}_{train_or_test_tickers}.html")
                fig.write_image(f"{log_dir}evaluation_5_{train_or_test_time}_{train_or_test_tickers}.pdf")


                # fig = go.Figure()
                #
                # fig.add_trace(go.Box(fillcolor="white", marker=dict(color='blue', line=dict(width=2, color='DarkSlateGrey')),
                #                      boxpoints="all", y=list(data_plot1["reinforcement_agent"]), name="Reinforcement Agent"),)
                # fig.add_trace(go.Box(fillcolor="white", marker=dict(color='red', line=dict(width=2, color='DarkSlateGrey')),
                #                      boxpoints="all", y=list(data_plot1["random_agent"]), name="Random Agent"))
                # fig.update_layout(template="simple_white")

                fig = px.box(data_depth_boxplots[data_depth_boxplots["color_by"]=="Random agent"], x="depth", y="reward", color_discrete_sequence=["red"])

                fig.update_layout(template="simple_white")

                fig.write_html(f"{log_dir}random_agent_boxplots_by_depth_{train_or_test_time}_{train_or_test_tickers}.html")
                fig.write_image(f"{log_dir}random_agent_boxplots_by_depth_{train_or_test_time}_{train_or_test_tickers}.pdf")
