"""
Script used to export graphs from tensorboard
"""
import logging
import os.path
import re
import time

import numpy as np
import plotly.express as px


def atoi(text):
    return int(text) if text.isdigit() else text


def natural_keys(text):
    """
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    """
    return [atoi(c) for c in re.split(r"(\d+)", text)]


def convert_tb_data(root_dirs, sort_by=None):
    """Convert local TensorBoard data into Pandas DataFrame.

    Function takes the root directory path and recursively parses
    all events data.
    If the `sort_by` value is provided then it will use that column
    to sort values; typically `wall_time` or `step`.

    *Note* that the whole data is converted into a DataFrame.
    Depending on the data size this might take a while. If it takes
    too long then narrow it to some sub-directories.

    Paramters:
        root_dir: (str) path to root dir with tensorboard data.
        sort_by: (optional str) column name to sort by.

    Returns:
        pandas.DataFrame with [wall_time, name, step, value] columns.

    """
    import os

    import pandas as pd
    from tensorflow.python.summary.summary_iterator import summary_iterator

    def convert_tfevent(filepath, run_name):
        df = pd.DataFrame(
            [
                parse_tfevent(e)
                for e in summary_iterator(filepath)
                if len(e.summary.value)
            ]
        )
        df["run"] = run_name
        return df

    def parse_tfevent(tfevent):
        return dict(
            wall_time=tfevent.wall_time,
            name=tfevent.summary.value[0].tag,
            step=tfevent.step,
            value=float(tfevent.summary.value[0].simple_value),
        )

    columns_order = ["run", "name", "step", "value"]

    out = []
    for root_dir in root_dirs:
        for (root, _, filenames) in os.walk(root_dir):
            for filename in filenames:
                if "events.out.tfevents" not in filename:
                    continue
                file_full_path = os.path.join(root, filename)
                run_name = root.split("/")[-2] + root.split("/")[-1]
                out.append(convert_tfevent(file_full_path, run_name))

    # Concatenate (and sort) all partial individual dataframes
    all_df = pd.concat(out)[columns_order]
    if sort_by is not None:
        all_df = all_df.sort_values(sort_by)

    return all_df.reset_index(drop=True)


if __name__ == "__main__":
    # dir_path = ["./tensorboard_logging/gym_final_final_final_without_penalty/", "./tensorboard_logging/gym_final_with_quadratic_penalty/"]
    # dir_path = ["./tensorboard_logging/gym_1200_with_linear_penalty_003_penalty_005/", "./tensorboard_logging/gym_1200_with_linear_penalty_003_penalty_0025/"]

    dir_path = [
        "./tensorboard_logging/gym_linear_005/",
        "./tensorboard_logging/gym_linear_0075/",
        "./tensorboard_logging/gym_linear_01/",
    ]

    df = convert_tb_data(dir_path)
    df = df.sort_values("step")
    df.loc[
        df["run"] == "gym_linear_005PPO_256_128_neurons_no_penalty_1", "run"
    ] = "Reinforcement agent - penalty, c=0.05"
    df.loc[
        df["run"] == "gym_linear_0075PPO_256_128_neurons_no_penalty_1", "run"
    ] = "Reinforcement agent - penalty, c=0.075"
    df.loc[
        df["run"] == "gym_linear_01PPO_256_128_neurons_no_penalty_1", "run"
    ] = "Reinforcement agent - penalty, c=0.1"
    df_groupby = df.groupby("name")
    for idx, group in df_groupby:
        logging.info(f"Generating graph {idx}")
        sorting = list(group["run"].unique())
        sorting.sort(key=natural_keys)
        if "ep_len_mean" in idx:
            yaxis_value = "Mean episode length over last 100 episodes"
        elif "ep_rew_mean" in idx:
            yaxis_value = "Mean reward over last 100 episodes"
        else:
            yaxis_value = "Value"
        line_dashing = []
        fig = px.line(
            group,
            x="step",
            y="value",
            color="run",
            title=yaxis_value,
            template="simple_white",
            category_orders={"run": sorting},
        )

        fig.update_traces(patch={"line": {"color": "black", "width": 2}})
        fig.update_traces(
            patch={"line": {"color": "rgb(117, 116, 111)", "width": 2, "dash": "dot"}},
            selector={"legendgroup": "Reinforcement agent - penalty, c=0.075"},
        )
        fig.update_traces(
            patch={"line": {"color": "rgb(117, 116, 111)", "width": 2, "dash": "dash"}},
            selector={"legendgroup": "Reinforcement agent - penalty, c=0.1"},
        )

        fig.update_layout(
            xaxis_title="# timesteps",
            yaxis_title=yaxis_value,
            legend_title="",
        )
        fig.layout.xaxis.tickfont.size = 20
        fig.layout.yaxis.tickfont.size = 20
        full_path = f"./tensorboard_logging/graphs/{idx}.html"
        if not os.path.exists(full_path):
            index = full_path.rfind("/")
            if not os.path.exists(full_path[:index]):
                os.makedirs(full_path[:index])
        fig.update_layout(height=800, width=800)
        fig.update_layout(
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=0.93,
                xanchor="right",
                x=1,
                traceorder="normal",
            )
        )
        fig.update_layout(
            font=dict(
                size=18,  # Set the font size here
            ),
            title=dict(
                font=dict(
                    size=20,
                )
            ),
        )
        fig.layout.legend.font.size = 20

        fig.data = sorted(fig.data, key=lambda x: x.name)
        fig.write_html(full_path)
        time.sleep(0.5)
        fig.write_image(full_path[:-5] + ".pdf")
        fig.write_image(full_path[:-5] + ".pdf")
