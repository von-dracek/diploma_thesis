import logging
import os.path

import numpy as np
import plotly.express as px

import re

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]


def convert_tb_data(root_dir, sort_by=None):
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
        df = pd.DataFrame([
            parse_tfevent(e) for e in summary_iterator(filepath) if len(e.summary.value)
        ])
        df["run"] = run_name
        return df

    def parse_tfevent(tfevent):
        return dict(
            wall_time=tfevent.wall_time,
            name=tfevent.summary.value[0].tag,
            step=tfevent.step,
            value=float(tfevent.summary.value[0].simple_value),
        )

    columns_order = ["run", 'name', 'step', 'value']

    out = []
    for (root, _, filenames) in os.walk(root_dir):
        for filename in filenames:
            if "events.out.tfevents" not in filename:
                continue
            file_full_path = os.path.join(root, filename)
            run_name = root.split("/")[-1]
            out.append(convert_tfevent(file_full_path, run_name))

    # Concatenate (and sort) all partial individual dataframes
    all_df = pd.concat(out)[columns_order]
    if sort_by is not None:
        all_df = all_df.sort_values(sort_by)

    return all_df.reset_index(drop=True)


if __name__ == "__main__":
    dir_path = "./tensorboard_logging/gym"
    df = convert_tb_data(f"{dir_path}")
    df = df.sort_values("step")
    df_groupby = df.groupby("name")
    for idx, group in df_groupby:
        logging.info(f"Generating graph {idx}")
        sorting = (list(group["run"].unique()))
        sorting.sort(key=natural_keys)
        fig = px.line(group, x="step", y="value", color="run", title=idx, template="simple_white", category_orders={"run":sorting})
        full_path = f"./tensorboard_logging/graphs/{idx}.html"
        if not os.path.exists(full_path):
            index = full_path.rfind("/")
            if not os.path.exists(full_path[:index]):
                os.makedirs(full_path[:index])
        fig.write_html(full_path)
    #todo: generate to files

    pass