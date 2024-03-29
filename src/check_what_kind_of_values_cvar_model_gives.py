"""Script used for checking generating figures in Section 4.6.1 -- Exploratory analysis"""

import logging
import pickle
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.configuration import ASSET_SET_1, ASSET_SET_2, ASSET_SET_3
from src.reinforcement_agent_v3 import make_treebuilding_env
from src.reinforcement_environment import TreeBuildingEnv

current_time = datetime.now().strftime("%Y-%m-%d %H,%M,%S")
perform_experiment = False
asset_sets = [ASSET_SET_1, ASSET_SET_2, ASSET_SET_3]
env_function_dict = {"train": make_treebuilding_env}  # , "test":make_testing_env}

if __name__ == "__main__":

    if perform_experiment:
        for asset_set_index, asset_set in enumerate(asset_sets):
            for env_name, env_function in env_function_dict.items():
                env = env_function(
                    defined_tickers=asset_set,
                    defined_alpha=0.9,
                    train_or_test="train",
                    train_or_test_time="train",
                )()

                env.seed(1337)
                np.random.seed(1337)  # to make sampling action space deterministic
                env.action_space.np_random.seed(1337)

                rewards = []
                i = 0
                count_to = 250
                while i < count_to:
                    logging.info(f"Calculating tree {i} out of {count_to}")
                    env.reset()
                    done = False
                    while not done:
                        valid_actions = env.valid_actions()
                        valid_actions = pd.DataFrame(valid_actions > 0)
                        valid_actions = valid_actions[valid_actions[0]]
                        valid_actions = list(valid_actions.index)
                        action = np.random.choice(valid_actions) - 3
                        obs, reward, done, info = env.step(action)
                    i += 1
                    rewards.append(
                        (env.current_num_scenarios, reward, env.get_branching())
                    )

                with open(
                    f"reward_distribution_from_mean_cvar_{env_name}_{asset_set_index}.pickle",
                    "wb",
                ) as handle:
                    pickle.dump(rewards, handle, protocol=pickle.HIGHEST_PROTOCOL)

                with open(
                    f"reward_distribution_from_mean_cvar_{env_name}_{asset_set_index}_{current_time}.pickle",
                    "wb",
                ) as handle:
                    pickle.dump(rewards, handle, protocol=pickle.HIGHEST_PROTOCOL)

                print(rewards)

            # with open(f'reward_distribution_from_mean_cvar_{current_time}.pickle', 'rb') as handle:
            #     data = pickle.load(handle)
            #     data = pd.DataFrame(data)
            #     fig = px.scatter(data, x=0, y=1, title="Scatterplot of generated rewards", template="simple_white")
            #     fig.update_layout(
            #         xaxis_title="# scenarios", yaxis_title="Reward Value"
            #     )
            #     fig.write_html(f"scatterplot_scenario_tree_generated_rewards_{current_time}.html")
    else:
        plots_to_produce = [
            f"reward_distribution_from_mean_cvar_{env_name}_{asset_set_index}"
            for (asset_set_index, _) in enumerate(asset_sets)
            for env_name in env_function_dict.keys()
        ]
        # plots_to_produce = ["reward_distribution_from_mean_cvar_2022-11-22 13,06,23", "reward_distribution_from_mean_cvar_2022-11-22 13,23,37", "reward_distribution_from_mean_cvar_2022-11-22 13,45,29", "reward_distribution_from_mean_cvar_2022-11-22 14,00,17", "reward_distribution_from_mean_cvar_2022-11-22 14,50,35", "reward_distribution_from_mean_cvar_2022-11-22 15,04,09"]
        titles = [(f"Asset set {i} - train period") for i in range(1, 4)]
        # titles = [x for tup in titles for x in tup]
        fig = make_subplots(rows=3, cols=1, subplot_titles=titles)
        for i, plot_string in enumerate(plots_to_produce):
            with open(f"{plot_string}.pickle", "rb") as handle:
                data = pickle.load(handle)
                data = pd.DataFrame(data)
                fig.add_trace(
                    go.Scatter(
                        x=list(data[0]),
                        y=list(data[1] / min(data[1])),
                        marker=dict(
                            color="grey", line=dict(width=2, color="DarkSlateGrey")
                        ),
                        mode="markers",
                    ),
                    row=i + 1,
                    col=1,
                )
        fig.update_layout(template="simple_white")
        for i in range(1, 4):
            for j in range(1, 3):
                fig.update_xaxes(title_text="# leaves", row=i, col=j)
                fig.update_yaxes(title_text="Normalized Reward", row=i, col=j)
        # fig.show()
        fig.update(layout_showlegend=False)
        fig.write_image(
            f"C:/Users/crash/Documents/Programming/diploma_thesis_merged/diploma_thesis_v2/scatterplots_scenarios_normalized_{current_time}.pdf"
        )
        fig.write_html(
            f"C:/Users/crash/Documents/Programming/diploma_thesis_merged/diploma_thesis_v2/scatterplots_scenarios_normalized_{current_time}.html"
        )

        del fig

        fig = make_subplots(rows=3, cols=1, subplot_titles=titles)

        for i, plot_string in enumerate(plots_to_produce):
            with open(f"{plot_string}.pickle", "rb") as handle:
                data = pickle.load(handle)
                data = pd.DataFrame(data)
                fig.add_trace(
                    go.Scatter(
                        mode="markers",
                        marker=dict(
                            color="grey", line=dict(width=2, color="DarkSlateGrey")
                        ),
                        x=list(data[2].apply(len)),
                        y=list(data[1] / min(data[1])),
                    ),
                    row=i + 1,
                    col=1,
                )
        fig.update_layout(template="simple_white")
        for i in range(1, 4):
            for j in range(1, 3):
                if j == 1:
                    fig.update_xaxes(title_text="Depth of tree", row=i, col=j)
                    fig.update_yaxes(title_text="Normalized Reward", row=i, col=j)
        # fig.show()
        fig.update(layout_showlegend=False)
        fig.write_image(
            f"C:/Users/crash/Documents/Programming/diploma_thesis_merged/diploma_thesis_v2/scatterplots_depth_normalized_{current_time}.pdf"
        )
        fig.write_html(
            f"C:/Users/crash/Documents/Programming/diploma_thesis_merged/diploma_thesis_v2/scatterplots_depth_normalized_{current_time}.html"
        )

        del fig

        fig = make_subplots(rows=3, cols=1, subplot_titles=titles)

        for i, plot_string in enumerate(plots_to_produce):
            with open(f"{plot_string}.pickle", "rb") as handle:
                data = pickle.load(handle)
                data = pd.DataFrame(data)
                fig.add_trace(
                    go.Box(
                        fillcolor="white",
                        marker=dict(
                            color="grey", line=dict(width=2, color="DarkSlateGrey")
                        ),
                        boxpoints="all",
                        x=list(data[2].apply(len)),
                        y=list(data[1] / min(data[1])),
                    ),
                    row=i + 1,
                    col=1,
                )
        fig.update_layout(template="simple_white")
        for i in range(1, 4):
            for j in range(1, 3):
                if j == 1:
                    fig.update_xaxes(title_text="Stage", row=i, col=j)
                    fig.update_yaxes(title_text="Normalized Reward", row=i, col=j)
        # fig.show()
        fig.update(layout_showlegend=False)
        fig.write_image(
            f"C:/Users/crash/Documents/Programming/diploma_thesis_merged/diploma_thesis_v2/boxplots_depth_normalized_{current_time}.pdf"
        )
        fig.write_html(
            f"C:/Users/crash/Documents/Programming/diploma_thesis_merged/diploma_thesis_v2/boxplots_depth_normalized_{current_time}.html"
        )

        fig = make_subplots(rows=3, cols=1, subplot_titles=titles)
        for i, plot_string in enumerate(plots_to_produce):
            with open(f"{plot_string}.pickle", "rb") as handle:
                data = pickle.load(handle)
                data = pd.DataFrame(data)
                fig.add_trace(
                    go.Scatter(
                        x=list(data[0]),
                        y=list(data[1]),
                        marker=dict(
                            color="grey", line=dict(width=2, color="DarkSlateGrey")
                        ),
                        mode="markers",
                    ),
                    row=i + 1,
                    col=1,
                )
        fig.update_layout(template="simple_white")
        for i in range(1, 4):
            for j in range(1, 3):
                if j == 1:
                    fig.update_xaxes(title_text="# leaves", row=i, col=j)
                    fig.update_yaxes(title_text="Normalized Reward", row=i, col=j)
        # fig.show()
        fig.update(layout_showlegend=False)
        fig.write_image(
            f"C:/Users/crash/Documents/Programming/diploma_thesis_merged/diploma_thesis_v2/scatterplots_scenarios_{current_time}.pdf"
        )
        fig.write_html(
            f"C:/Users/crash/Documents/Programming/diploma_thesis_merged/diploma_thesis_v2/scatterplots_scenarios_{current_time}.html"
        )

        del fig

        fig = make_subplots(rows=3, cols=1, subplot_titles=titles)

        for i, plot_string in enumerate(plots_to_produce):
            with open(f"{plot_string}.pickle", "rb") as handle:
                data = pickle.load(handle)
                data = pd.DataFrame(data)
                fig.add_trace(
                    go.Scatter(
                        mode="markers",
                        marker=dict(
                            color="grey", line=dict(width=2, color="DarkSlateGrey")
                        ),
                        x=list(data[2].apply(len)),
                        y=list(data[1]),
                    ),
                    row=i + 1,
                    col=1,
                )
        fig.update_layout(template="simple_white")
        for i in range(1, 4):
            for j in range(1, 3):
                if j == 1:
                    fig.update_xaxes(title_text="Stage", row=i, col=j)
                    fig.update_yaxes(title_text="Normalized Reward", row=i, col=j)
        # fig.show()
        fig.update(layout_showlegend=False)
        fig.write_image(
            f"C:/Users/crash/Documents/Programming/diploma_thesis_merged/diploma_thesis_v2/scatterplots_depth_{current_time}.pdf"
        )
        fig.write_html(
            f"C:/Users/crash/Documents/Programming/diploma_thesis_merged/diploma_thesis_v2/scatterplots_depth_{current_time}.html"
        )

        del fig

        fig = make_subplots(rows=3, cols=1, subplot_titles=titles)

        for i, plot_string in enumerate(plots_to_produce):
            with open(f"{plot_string}.pickle", "rb") as handle:
                data = pickle.load(handle)
                data = pd.DataFrame(data)
                fig.add_trace(
                    go.Box(
                        fillcolor="white",
                        marker=dict(color="grey", line=dict(width=2, color="black")),
                        boxpoints="all",
                        x=list(data[2].apply(len)),
                        y=list(data[1]),
                    ),
                    row=i + 1,
                    col=1,
                )
        fig.update_layout(template="simple_white")
        for i in range(1, 4):
            for j in range(1, 3):
                if j == 1:
                    fig.update_xaxes(title_text="Number of stages", row=i, col=j)
                    fig.update_yaxes(title_text="Reward", row=i, col=j)
        # fig.show()
        fig.update(layout_showlegend=False)
        fig.update_layout(height=1000, width=1000)
        fig.write_image(
            f"C:/Users/crash/Documents/Programming/diploma_thesis_merged/diploma_thesis_v2/boxplots_depth_{current_time}.pdf"
        )
        fig.write_html(
            f"C:/Users/crash/Documents/Programming/diploma_thesis_merged/diploma_thesis_v2/boxplots_depth_{current_time}.html"
        )
