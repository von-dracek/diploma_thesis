from typing import Any, List, Tuple

import numpy as np
from anytree import LevelOrderGroupIter, Node

from src.moment_matching_model import build_mm_model


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


def calculate_path_probabilities(root: Node):
    if root.parent is not None:
        root.path_probability = root.parent.path_probability * root.probability
    for child in root.children:
        calculate_path_probabilities(child)

def fill_empty_tree_with_scenario_data_moment_matching(
    TARMOM: np.ndarray, R: np.ndarray, root: Node, branching: List[int], gams_workspace
) -> Node:
    tree_levels = list(LevelOrderGroupIter(root))
    root.probability = 1
    root.path_probability = 1
    tree_level_zip_branching = list(
        zip(tree_levels, branching)
    )  # purposely zipping n levels with n-1 branching - last level is not iterated over
    # looping over each tree level
    for level, current_branching in tree_level_zip_branching:
        # generating children from one node of current level and then copying these children
        # to other nodes on the current level
        # generate childrenvalues
        generated_returns, generated_probs = build_mm_model(current_branching, TARMOM, R, gams_workspace)
        # set values to children
        for current_node in level:
            children = current_node.children
            for idx, child in enumerate(children):
                child.returns = generated_returns[:, idx % current_branching]
                child.probability = generated_probs[idx % current_branching]
    calculate_path_probabilities(root)

    assert abs(sum(node.path_probability for node in root.leaves) - 1) < 1e-3
    return root