import numpy as np
import pulp
from anytree import Node

from src.configuration import CVAR_ALPHA, TICKERS


def calculate_mean_cvar_over_leaves(root: Node) -> float:
    leaves = root.leaves
    S = len(leaves)
    R_0 = 3
    means = np.stack([leaf.cumulative_returns for leaf in leaves])
    means = means.mean(axis=0)
    cvar = pulp.LpProblem("Mean_Cvar_problem", pulp.LpMinimize)
    u = [pulp.LpVariable(f"U_{i}", lowBound=0) for i, _ in enumerate(leaves)]
    ksi = pulp.LpVariable("ksi")
    x = [pulp.LpVariable(f"X_{i}", lowBound=0) for i, _ in enumerate(TICKERS)]
    x_constraint = pulp.LpConstraint(pulp.lpSum(x), sense=0, rhs=1)
    better_than_risk_free_constraint = pulp.LpConstraint(pulp.lpDot(x, means), sense=1, rhs=R_0)
    u_constraints = [
        pulp.LpConstraint(
            us + pulp.lpDot(x, leaves[int(us.name[2:])].cumulative_returns) + ksi, rhs=0, sense=1
        )
        for us in u
    ]
    objective = ksi + (1 / ((1 - CVAR_ALPHA) * S)) * pulp.lpSum(u)
    cvar += objective
    cvar += x_constraint
    cvar += better_than_risk_free_constraint
    for u_cons in u_constraints:
        cvar += u_cons

    optimality = cvar.solve()
    optimal_x = {}
    for v in x:
        optimal_x[v.name] = v.varValue
    print(optimality)
