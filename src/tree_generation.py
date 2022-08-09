import logging
from typing import Any, Dict, List, Tuple, Union

import numpy as np
import pandas as pd
from anytree import LevelGroupOrderIter, Node
from scipy.optimize import minimize
from scipy.optimize import LinearConstraint


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
    #need number_of_nodes more values to calculate probabilities
    num_variables = (n_stocks) * number_of_nodes + number_of_nodes
    starting_values = np.random.random_sample(size=(num_variables)) * 0.04 + 0.98
    linear_constraint_matrix = np.zeros((1+number_of_nodes,num_variables))
    #first constraint - probabilities sum up to 1
    #last row is assumed to be probability of each node
    linear_constraint_matrix[0,-number_of_nodes:] = 1
    for i in range(number_of_nodes):
        linear_constraint_matrix[i+1,-number_of_nodes+i] = 1

    tmp = np.reshape(linear_constraint_matrix, (-1, number_of_nodes))

    linear_constraint = LinearConstraint(linear_constraint_matrix, [1] + [0]*number_of_nodes, [1] + [1]*number_of_nodes)

    result = minimize(
        weighted_loss_function_diag,
        starting_values,
        method="trust-constr",
        args=(moments, r, number_of_nodes),
        constraints=[linear_constraint],
        options={"disp": True},
    )
    tmp = np.reshape(result.x, (-1, number_of_nodes))
    x = tmp[:-1,:]
    probs = tmp[-1,:]
    #todo:
    # return x, probs
    # x contains values, probs contain probabilities for each node
    return result

# https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.497.113&rep=rep1&type=pdf
# Data-Driven Multi-Stage Scenario Tree Generation via Statistical Property and Distribution Matching
# Bruno A. Calfa∗, Anshul Agarwal†, Ignacio E. Grossmann∗, John M. Wassick†
def weighted_loss_function_diag(
    x: np.ndarray,
    TARMOM: np.ndarray,
    R: np.ndarray,
    number_of_nodes: int,
    weights: np.ndarray = np.ones(5),
) -> float:
    x = np.reshape(x, (-1, number_of_nodes))
    probs = x[-1,:]
    # probs[1]=0#first column is assumed to be probabilities
    x = x[:-1,:]
    mean = np.sum((x*probs), axis=1)
    tiled_mean = np.tile(mean, (number_of_nodes, 1)).transpose()
    variance = np.sum((np.power(x - tiled_mean, 2)*probs), axis=1)
    third = np.sum((np.power(x - tiled_mean, 3)*probs), axis=1)
    fourth = np.sum((np.power(x - tiled_mean, 4)*probs), axis=1)

    #TODO: use also correlation matrix
    # R_discrete_manual = (x-tiled_mean).T*(x-tiled_mean)
    # covariance = np.cov(x, ddof=0, aweights=probs)
    # R_discrete = np.corrcoef(x)


    #loss contains also the weights as presented in the given paper (L2 MMP formulation)
    loss = (
        weights[0] * np.sum((1/np.power(TARMOM[0, :], 2))*np.power(mean - TARMOM[0, :], 2))
        + weights[1] * np.sum((1/np.power(TARMOM[1, :], 2))*np.power(variance - TARMOM[1, :], 2))
        + weights[2] * np.sum((1/np.power(TARMOM[2, :], 2))*np.power(third - TARMOM[2, :], 2))
        + weights[3] * np.sum((1/np.power(TARMOM[3, :], 2))*np.power(fourth - TARMOM[3, :], 2))
        # + weights[4] * np.sum((1/np.power(R, 2))*np.power(np.triu(R_discrete - R, 1), 2))
    )

    return loss
