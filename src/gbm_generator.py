import logging

import numpy as np
import pandas as pd
from anytree import Node
from k_means_constrained import KMeansConstrained
from tf_quant_finance.models import MultivariateGeometricBrownianMotion


def get_gbm_scenarios_from_TARMOM_and_R(TARMOM, R, branching):
    n_stocks = R.shape[0]
    means = TARMOM[0, :]
    volatilities = TARMOM[1, :]

    gbm = MultivariateGeometricBrownianMotion(
        dim=n_stocks, means=means, volatilities=volatilities, corr_matrix=R
    )
    paths = gbm.sample_paths(
        times=[x + 1 for x in list(range(len(branching)))], num_samples=np.prod(branching)
    )
    paths = paths._numpy()

    logging.info("Generated paths")

    levels = []
    level_branching = {}
    cluster_centers = {}  # noqa: F841
    for i in range(len(branching)):
        levels.append(pd.DataFrame(paths[:, i, :]))
    for i, level in enumerate(levels):
        branch = branching[i]
        if level_branching.get(i - 1) is not None:
            for _k, v in level_branching.items():
                if _k == 0 or (isinstance(_k, tuple) and _k[0] == i - 1):
                    data = level.loc[v.tolist(), :]
                    kmeansconstrained = KMeansConstrained(
                        n_clusters=branch, size_min=data.shape[0] // branch - 1
                    )
                    kmeansconstrained.fit(data)
                    level_branching[i, _k[i:]] = (
                        pd.DataFrame(kmeansconstrained.labels_, index=data.index)
                        .groupby(0)
                        .indices
                    )
        else:
            kmeansconstrained = KMeansConstrained(
                n_clusters=branch, size_min=level.shape[0] // branch - 1
            )
            kmeansconstrained.fit(level)
            level_branching[i] = (
                pd.DataFrame(kmeansconstrained.labels_, index=level.index).groupby(0).indices
            )
            # cluster_centers[i] = kmeansconstrained.cluster_centers_.to_dict()
        logging.info(f"Fitted kmeans level {i}")


def get_gbm_scenarios_from_TARMOM_and_R_recursive(TARMOM, R, branching):
    n_stocks = R.shape[0]
    means = TARMOM[0, :]
    volatilities = TARMOM[1, :]

    gbm = MultivariateGeometricBrownianMotion(
        dim=n_stocks, means=means, volatilities=volatilities, corr_matrix=R
    )
    paths = gbm.sample_paths(
        times=[x + 1 for x in list(range(len(branching)))], num_samples=np.prod(branching)
    )
    paths = paths._numpy()

    logging.info("Generated paths")

    dfs = []
    for i in range(len(branching)):
        dfs.append(pd.DataFrame(paths[:, i, :]))

    branching_dict = {}  # noqa: F841

    def get_kmeans(dataframes, branching, i, subset=None):
        if i == len(branching) - 1:
            return {str(k): {"value": dataframes[i].loc[k, :]} for k in subset}
        if i >= len(branching) - 1:
            raise ValueError
        data = dataframes[i]
        branch = branching[i]
        if subset is not None:
            _df = data.loc[subset, :]
        else:
            _df = data
        kmeansconstrained = KMeansConstrained(n_clusters=branch, size_min=_df.shape[0] // branch)
        kmeansconstrained.fit(_df)
        tmp = (
            pd.DataFrame(kmeansconstrained.labels_, index=_df.index)
            .groupby(0)
            .nth.groupby_object.groups
        )
        values = pd.DataFrame(kmeansconstrained.cluster_centers_)
        return_dict = {}
        for k, v in tmp.items():
            v = v.tolist()
            return_dict[str(k)] = get_kmeans(dataframes, branching, i + 1, subset=v)
            if isinstance(return_dict[str(k)], dict):
                return_dict[str(k)]["value"] = values.loc[k, :]
        return return_dict

    clustered_branchings = get_kmeans(dfs, branching, 0)

    root = Node("0")

    def recursively_build_tree_from_dict(clustering_dict, root: Node):
        for i, (k, v) in enumerate(clustering_dict.items()):
            if k == "value":
                continue
            new_node = Node(f"{root.name}{i}", parent=root)
            new_node.value = v["value"]
            recursively_build_tree_from_dict(v, new_node)

    recursively_build_tree_from_dict(clustered_branchings, root)
