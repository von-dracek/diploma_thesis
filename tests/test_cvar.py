import pickle

import numpy as np
import pandas as pd
import pytest
from anytree import PreOrderIter
from tabulate import tabulate

from src.mean_cvar import calculate_mean_cvar_over_leaves


def _create_cvar_gams_model_str_for_test(root):
    leaves = root.leaves
    specified_row_name_length = 20
    scenarios = []
    for leaf in leaves:
        scenario_series = []
        for node in leaf.ancestors + (leaf,):
            if node.depth > 0:
                ser = pd.Series(
                    node.returns,
                    index=[
                        f"stock{n+1}"
                        + " "
                        * (specified_row_name_length - len(f"stock{n+1}") - len(f".T{node.depth}"))
                        + f".T{node.depth}"
                        for n in range(len(node.returns))
                    ],
                )
                scenario_series.append(ser)
        scenario = pd.concat(scenario_series)
        scenario.name = leaf.name
        scenarios.append(scenario)
    scenario_df = pd.concat(scenarios, axis=1)
    scenario_probabilities = {
        leaf.name: float(np.prod([node.probability for node in (leaf,) + leaf.ancestors]))
        for leaf in leaves
    }
    assert abs(sum(scenario_probabilities.values()) - 1) < 1e-4

    def get_siblings(root):
        siblings = [(a.name, root.depth, b.name) for a in root.leaves for b in root.leaves]
        for node in root.children:
            siblings += get_siblings(node)
        return siblings

    siblings = get_siblings(root)
    ssiblings = [
        {"node1": sibling[0], "time": sibling[1], "node2": sibling[2]}
        for sibling in siblings
        if sibling[0] != sibling[2]
    ]
    siblings = pd.DataFrame(ssiblings)
    sib = (
        siblings["node1"].astype(str)
        + "."
        + siblings["node2"].astype(str)
        + ".T"
        + siblings["time"].astype(str)
    )
    sib = sib.to_list()

    min_expected_return = 0
    _str = f"""
Option LP=CPLEX;


Set       s        / {", ".join([leaf.name for leaf in leaves])} / ;
Set       i        / {', '.join([f'stock{n + 1}' for n in range(len(node.returns))])} / ;
Set       t        / {', '.join([f'T{n}' for n in range((root.height + 1))])} / ;

Alias (s, ss);
Set siblings /{', '.join(sib)}/
;

Table
AssetYields(i, t, s)
{tabulate(scenario_df, headers="keys", tablefmt="plain", numalign="right", showindex=True, floatfmt=".4f")}
;

Parameter
ScenarioProbabilities(s)
/{', '.join([str(k) + " " + "{:.7f}".format(v) for k, v in scenario_probabilities.items()])}/
;
Parameter
InitialWealth /1/;
Parameter
MinimumExpectedReturn /{min_expected_return}/;
Parameter
alpha /0.95/;

Positive Variables x(i,s,t);
x.l(i,s,t)=1/card(i);
Variable loss;
Positive Variables y(s);
Variable wealth(s,t);
Variable z;
Variable v;
Variable TotalExcessScenarioReturns(s);
Variable TotalScenarioLoss(s);

Equations          rootvariablessumuptoinitialwealth, nonanticipativityofx, eqwealth, eqinitialwealth, eqassetyields, objective, yconstrs, ExcessScenarioReturns(s,t), EqTotalScenarioLoss(s), benchmark;

rootvariablessumuptoinitialwealth(s).. InitialWealth=e=sum(i,x(i,s,"T0"));
eqinitialwealth(s).. wealth(s,"T0") =e= InitialWealth;

eqwealth(s, t).. wealth(s,t) =e= sum(i,x(i,s,t));
eqassetyields(s,t)$(ord(t)>1).. wealth(s,t) =e= sum(i, x(i,s,t-1)*AssetYields(i,t,s));

ExcessScenarioReturns(s,t)$(ord(t)=card(t)).. TotalExcessScenarioReturns(s)=e=wealth(s,t)-1;
**Previous row has assumption.. wealth(s,"T0")=1!
EqTotalScenarioLoss(s)..TotalScenarioLoss(s) =e= -TotalExcessScenarioReturns(s);
yconstrs(s).. y(s)=g=TotalScenarioLoss(s)-z;

nonanticipativityofx(i,s,ss,t)$(siblings(s,ss,t))..  x(i,s,t) =e= x(i,ss,t);
benchmark.. sum(s,ScenarioProbabilities(s)*TotalExcessScenarioReturns(s)) =g= MinimumExpectedReturn;

objective.. loss=e=z+(1/(1-alpha))*sum(s,ScenarioProbabilities(s)*y(s));

Model problem / ALL /;

solve problem using LP minimising loss;
"""

    return _str


def test_mean_cvar():
    """If all scenarios return nothing (i.e. not investing and holding cash (while not accounting for inflation)) the loss value should be 0."""
    with open("./tests/root.pickle", "rb") as handle:
        root = pickle.load(handle)
    for node in PreOrderIter(root):
        if hasattr(node, "returns"):
            node.returns = node.returns * 0 + 1
    loss = calculate_mean_cvar_over_leaves(root, _create_cvar_gams_model_str_for_test)
    assert loss == pytest.approx(0)
