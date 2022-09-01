import logging
from typing import Any, Dict, List, Tuple, Union

import numpy as np
import pandas as pd
from anytree import LevelGroupOrderIter, Node, LevelOrderGroupIter
from scipy.optimize import minimize
from scipy.optimize import LinearConstraint
from gekko import GEKKO
from src.configuration import TICKERS


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

def fill_empty_tree_with_probabilities(root: Node):
    for child in root.children:
        # child.probability = child.parent.probability / len(root.children)
        child.probability = 1 / len(root.children)
        child.path_probability = root.path_probability * child.probability
        fill_empty_tree_with_probabilities(child)
    return root

def fill_empty_tree_with_scenario_data(TARMOM: pd.DataFrame, R:pd.DataFrame, root: Node, branching: int) -> Node:
    tree_levels = [level for level in LevelOrderGroupIter(root)]  # dropping level with only root node -> root node has no scenario set
    root.probability = 1
    root.path_probability = 1
    root = fill_empty_tree_with_probabilities(root)
    tree_level_zip_branching = list(zip(tree_levels, branching)) #purposely zipping n levels with n-1 branching - last level is not iterated over
    for level, current_branching in tree_level_zip_branching:
        probability = np.unique([child.probability for node in level for child in node.children])
        probabilities = np.ones(current_branching) * probability
        generated_returns, generated_probs = generate_scenario_level_gekko_uniform_iid(TARMOM, R, probabilities, current_branching)
        for current_node in level:
            children = current_node.children
            for idx, child in enumerate(children):
                child.returns = generated_returns[:,idx % 7]

    assert abs(sum([node.path_probability for node in tree_levels[-1]])-1) < 1e-4
    for level, current_branching in tree_level_zip_branching:
        for current_node in level:
            if not current_node.is_root:
                children = current_node.children
                for idx, child in enumerate(children):
                    child.cumulative_returns = current_node.returns * child.returns

    return root

def get_tarmom_and_r_from_dataset_dict(
    dataset_dict: Dict[str, Union[pd.DataFrame, np.ndarray]]
) -> Tuple[np.ndarray, np.ndarray]:
    return dataset_dict["moments"], dataset_dict["R"]

# https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.497.113&rep=rep1&type=pdf
# Data-Driven Multi-Stage Scenario Tree Generation via Statistical Property and Distribution Matching
# Bruno A. Calfa∗, Anshul Agarwal†, Ignacio E. Grossmann∗, John M. Wassick†
def weighted_loss_function_diag(
    x: np.ndarray,
    probabilities: np.ndarray,
    TARMOM: np.ndarray,
    R: np.ndarray,
    number_of_nodes: int,
    m: GEKKO,
    weights: np.ndarray = np.ones(5),
) -> float:
    mean = m.sum((x*probabilities).T).reshape((-1,1))
    variance = m.sum((((x - mean) ** 2)*probabilities).T)
    third = m.sum(((((x - mean)**3)*probabilities)).T)
    fourth = m.sum((((x - mean)**4)*probabilities).T)


    #TODO: use also correlation matrix
    # R_discrete_manual = (x-tiled_mean)@((x-tiled_mean)*probs).T
    # covariance = np.cov(x, ddof=0, aweights=probs)
    # R_discrete = np.corrcoef(x)
    cov_discrete = ((x-mean)*probabilities @ (x-mean).T)
    diag_var = np.diag(variance**(-1/2))
    corr_discrete = (diag_var)@cov_discrete@(diag_var)
    print(corr_discrete)

    loss = m.sum([
        weights[0] * m.sum((mean.reshape(-1) - TARMOM[0, :])**2)
        ,5*weights[1] * m.sum((variance - TARMOM[1, :])**2)
        ,5*weights[2] * m.sum((third - TARMOM[2, :])**2)
        ,5*weights[3] * m.sum((fourth - TARMOM[3, :])**2)
        # ,50000* m.sum((corr_discrete - R)**2)
    ]
    )


    return loss

def generate_scenario_level_gekko_uniform_iid(TARMOM:np.ndarray, R:np.ndarray, probabilities:np.ndarray, number_of_nodes_in_level:int) -> Tuple[Any, np.ndarray]:
    n_stocks = R.shape[0]
    m = GEKKO(remote=False)
    m.options.MAX_MEMORY = 8
    m.options.SOLVER = 3
    m.options.MAX_ITER = 5000
    # m.options.RTOL = 1.0e-7
    x = m.Array(m.Var, (n_stocks, number_of_nodes_in_level), value=1, lb=0, ub=7) #return scenarios
    for i in range(n_stocks):
        for j in range(number_of_nodes_in_level):
            x[i,j] = m.Var(value=np.random.random_sample() * 0.04 + 1, lb=0, ub=6)

    p=probabilities
    res = m.Minimize(weighted_loss_function_diag(x, p, TARMOM, R, number_of_nodes_in_level, m))
    m.solve(disp=True)
    x = np.array([[__x.value for __x in _x] for _x in x]).reshape(x.shape)
    first_moments = (x*p).sum(axis=1).reshape((-1,1))
    variance = np.sum((((x - first_moments) ** 2)*p).T, axis=0)
    third = np.sum(((((x - first_moments)**3)*p)).T, axis=0)
    fourth = np.sum((((x - first_moments)**4)*p).T, axis=0)
    moments_from_tree = np.stack([first_moments.reshape(-1), variance, third, fourth])
    temp = TARMOM - moments_from_tree
    #TODO: computation of correlation matrix is not working properly, not here, not in loss function
    cov_discrete = ((x-first_moments)*probabilities @ (x-first_moments).T)
    diag_var = np.diag(variance**(-1/2))
    corr_discrete = (diag_var)@cov_discrete@(diag_var)
    print(corr_discrete)
    print(temp)

    return x, p

