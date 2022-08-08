import logging
from typing import Any, Dict, List, Tuple, Union

import numpy as np
import pandas as pd
from anytree import LevelGroupOrderIter, Node
from scipy.optimize import minimize


def create_empty_tree(sub_tree_structure: List, parent: Node = None) -> Node:
    """Recursively generates empty scenario tree based on branching structure"""
    if parent is None:
        parent = Node("0")
    children = []
    for s in range(sub_tree_structure[0]):
        children.append(Node(parent.name + str(s)))
    parent.children = children

    if len(sub_tree_structure) > 1:
        for child in parent.children:
            create_empty_tree(sub_tree_structure[1:], child)
    return parent


def fill_empty_tree_with_scenario_data(
    datasets: Dict[str, Dict[str, Union[pd.DataFrame, np.ndarray, np.ndarray]]], root: Node
) -> Node:
    tree_levels = [level for level in LevelGroupOrderIter(root)][
        1:
    ]  # dropping level with only root node -> root node has no scenario set
    assert len(tree_levels) == len(datasets), (
        "The number of levels in the tree "
        "does not correspond to number of datasets "
        "prepared by specified branching"
    )
    timestamps = sorted(datasets.keys())
    for level_idx, level in enumerate(tree_levels):
        # generate scenario
        dataset_corresponding_to_level = datasets[timestamps[level_idx]]
        number_of_nodes_in_level = len(level)
        logging.info(f"Starting scenario generation for level {level_idx}")
        result = generate_scenario_level(dataset_corresponding_to_level, number_of_nodes_in_level)
        scenarios = result["x"].reshape(-1, number_of_nodes_in_level)
        for node_idx, node in enumerate(level):
            node.returns = scenarios[:, node_idx]
            # fill value from scenario to node
            # scenarios is array of shape (n_stocks, n_nodes)
            # each column is set to given node and give the scenarios
            # for each stock in the given node
        logging.info(f"Level {level_idx} generated")
    return root


def get_tarmom_and_r_from_dataset_dict(
    dataset_dict: Dict[str, Union[pd.DataFrame, np.ndarray]]
) -> Tuple[np.ndarray, np.ndarray]:
    return dataset_dict["moments"], dataset_dict["R"]


# https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.497.113&rep=rep1&type=pdf
# Data-Driven Multi-Stage Scenario Tree Generation via Statistical Property and Distribution Matching
# Bruno A. Calfa∗, Anshul Agarwal†, Ignacio E. Grossmann∗, John M. Wassick†
def generate_scenario_level(
    dataset_dict: Dict[str, Union[pd.DataFrame, np.ndarray]], number_of_nodes: int
) -> Any:
    moments, r = get_tarmom_and_r_from_dataset_dict(dataset_dict)
    n_stocks = r.shape[0]  # number of stocks is the dimension of corr matrix
    starting_values = np.random.random_sample(size=(n_stocks * number_of_nodes)) * 0.04 + 0.98
    starting_values = np.reshape(starting_values, (-1, number_of_nodes))
    result = minimize(
        weighted_loss_function_diag,
        starting_values,
        args=(moments, r, number_of_nodes),
        options={"disp": True},
    )
    return result


def weighted_loss_function_diag(
    x: np.ndarray,
    TARMOM: np.ndarray,
    R: np.ndarray,
    n_nodes: int,
    weights: np.ndarray = np.ones(5),
) -> float:
    mean = np.mean(x, axis=1)
    variance = np.sum(np.power(x - np.tile(mean, (n_nodes, 1)).transpose(), 2), axis=1) / n_nodes
    third = np.sum(np.power(x - np.tile(mean, (n_nodes, 1)).transpose(), 3), axis=1) / n_nodes
    fourth = np.sum(np.power(x - np.tile(mean, (n_nodes, 1)).transpose(), 4), axis=1) / n_nodes

    R_discrete = np.corrcoef(x)

    loss = (
        weights[0] * np.sum(np.power(mean - TARMOM[0, :], 2))
        + weights[1] * np.sum(np.power(variance - TARMOM[1, :], 2))
        + weights[2] * np.sum(np.power(third - TARMOM[2, :], 2))
        + weights[3] * np.sum(np.power(fourth - TARMOM[3, :], 2))
        + weights[4] * np.sum(np.power(np.triu(R_discrete - R, 1), 2))
    )

    return loss
