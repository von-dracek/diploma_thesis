from typing import Any, List, Tuple

import numpy as np
from anytree import LevelOrderGroupIter, Node
from gekko import GEKKO

from src.build_gams import build_mm_model


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
        child.probability = 1 / len(root.children)
        child.path_probability = root.path_probability * child.probability
        fill_empty_tree_with_probabilities(child)
    return root


def fill_empty_tree_with_scenario_data(
    TARMOM: np.ndarray, R: np.ndarray, root: Node, branching: List[int]
) -> Node:
    tree_levels = list(LevelOrderGroupIter(root))
    root.probability = 1
    root.path_probability = 1
    root = fill_empty_tree_with_probabilities(root)
    tree_level_zip_branching = list(
        zip(tree_levels, branching)
    )  # purposely zipping n levels with n-1 branching - last level is not iterated over
    # looping over each tree level
    for level, current_branching in tree_level_zip_branching:
        # generating children from one node of current level and then copying these children
        # to other nodes on the current level
        # probability = np.unique([child.probability for node in level for child in node.children])
        # probabilities = np.ones(current_branching) * probability
        # generate children
        # s = prepare_moment_matching_model_str(current_branching,TARMOM, R)
        generated_returns, generated_probs = build_mm_model(current_branching, TARMOM, R)
        # (
        #     generated_returns,
        #     generated_probs,  # pylint: disable=W0612
        # ) = generate_scenario_level_gekko_uniform_iid
        # (TARMOM, R, probabilities, current_branching)
        # set values to children
        for current_node in level:
            children = current_node.children
            for idx, child in enumerate(children):
                child.returns = generated_returns[:, idx % current_branching]
                child.probability = generated_probs[idx % current_branching]

    assert abs(sum(node.path_probability for node in tree_levels[-1]) - 1) < 1e-4
    # calculate cumulative returns
    for level, current_branching in tree_level_zip_branching:
        for current_node in level:
            if not current_node.is_root:
                children = current_node.children
                for idx, child in enumerate(children):
                    child.cumulative_returns = current_node.returns * child.returns

    return root


# https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.497.113&rep=rep1&type=pdf
# Data-Driven Multi-Stage Scenario Tree Generation via Statistical Property
# and Distribution Matching
# Bruno A. Calfa∗, Anshul Agarwal†, Ignacio E. Grossmann∗, John M. Wassick†
def generate_scenario_level_gekko_uniform_iid(
    TARMOM: np.ndarray, R: np.ndarray, proba: np.ndarray, number_of_nodes: int
) -> Tuple[Any, np.ndarray]:
    n_stocks = R.shape[0]
    # tried also with remote=False and all available solvers (IPOPT, APOPT, BPOPT)
    m = GEKKO(remote=False)
    m.options.MAX_MEMORY = 8
    m.options.SOLVER = 3
    m.options.MAX_ITER = 100000
    np.random.seed(1337)
    # x - array of decision variables of dimension (n_stocks, n_nodes)
    x = m.Array(m.Var, (n_stocks, number_of_nodes), lb=0.0001)

    # normal initialisation - initial guess using sampling from normal distribution with given
    # mean and correlation matrix
    normal_initialiastion = np.random.multivariate_normal(
        TARMOM[0, :], R, size=(number_of_nodes)
    ).T

    # here I was trying to also use different probs for each child - didnt help
    # # array of node probabilities of each node - just initialise it
    # p = m.Array(m.Var, (number_of_nodes), lb=0.0001, ub=1)
    # # fill array of probabilities with values
    # for j in range(number_of_nodes):
    #     p[j] = m.Var(value=proba[j], lb=0.0001, ub=1)
    p = proba
    # node probabilities must sum to 1
    # m.Equation(m.sum(p) == 1)

    # fill the decision variables with initial values
    for i in range(n_stocks):
        for j in range(number_of_nodes):
            x[i, j] = m.Var(value=normal_initialiastion[i, j], lb=0.0001)

    # calculate first four moments from the values of decision variables

    means = [m.sum(x[i, :] * p) for i in range(x.shape[0])]
    variances = [m.sum(((x[i, :] - means[i]) ** 2) * p) for i in range(x.shape[0])]
    third = [m.sum(((x[i, :] - means[i]) ** 3) * p) for i in range(x.shape[0])]
    fourth = [m.sum(((x[i, :] - means[i]) ** 4) * p) for i in range(x.shape[0])]

    # compute loss
    loss = m.sum(
        [
            m.sum(([(means[i] - TARMOM[0, i]) ** 2 for i in range(x.shape[0])])),
            m.sum(([(variances[i] - TARMOM[1, i]) ** 2 for i in range(x.shape[0])])),
            # m.sum(([(third[i] - TARMOM[2, i]) ** 2 for i in range(x.shape[0])])),
            # m.sum(([(fourth[i] - TARMOM[3, i]) ** 2 for i in range(x.shape[0])])),
            # m.sum((corr_discrete - R) ** 2)
        ]
    )

    res = m.Minimize(loss)  # noqa: F841
    m.solve(disp=True)
    x = np.array([[__x.value for __x in _x] for _x in x]).reshape(x.shape)
    # p = np.array([_p.value for _p in p]).reshape(-1)
    first_moments = (x * p).sum(axis=1).reshape((-1, 1))
    variance = np.sum((((x - first_moments) ** 2) * p).T, axis=0)
    third = np.sum(((((x - first_moments) ** 3) * p)).T, axis=0)
    fourth = np.sum((((x - first_moments) ** 4) * p).T, axis=0)
    moments_from_tree = np.stack([first_moments.reshape(-1), variance, third, fourth])
    diff_in_moments = TARMOM - moments_from_tree  # noqa: F841  # just for visual inspection
    # cov_discrete = (x - first_moments) * p @ (x - first_moments).T
    # diag_var = np.diag(variance ** (-1 / 2))
    # corr_discrete = (diag_var) @ cov_discrete @ (diag_var)
    # diff_in_corr = corr_discrete - R

    return x, p
