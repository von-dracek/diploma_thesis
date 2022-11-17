import io
from typing import Callable

import numpy as np
import pandas as pd
from anytree import Node
from gams import GamsWorkspace
from tabulate import tabulate
import logging

specified_row_name_length = 20


def create_cvar_gams_model_str(root: Node, alpha: float):
    leaves = root.leaves

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
    n_columns = scenario_df.shape[1]
    splits_by_cols = list(range(0, n_columns, 1500)) + [n_columns]
    scenario_dfs_splitted_list = [scenario_df.iloc[:, i:j] for i, j in zip(splits_by_cols, splits_by_cols[1:])]
    asset_yields_string = ''
    for i, df in enumerate(scenario_dfs_splitted_list):
        asset_yields_string += tabulate(df, headers="keys", tablefmt="plain", numalign="right", showindex=True,
                                        floatfmt=".4f")
        asset_yields_string += "\n"
        if i + 1 != len(scenario_dfs_splitted_list):
            asset_yields_string += "+"
            asset_yields_string += "\n"
    scenario_probabilities = {
        leaf.name: float(np.prod([node.probability for node in (leaf,) + leaf.ancestors]))
        for leaf in leaves
    }
    assert abs(sum(scenario_probabilities.values()) - 1) < 1e-4
    logging.info("Built asset yields table")
    def get_siblings(root):
        if len(root.leaves) > 0:
            a = root.leaves[0]
            siblings = [(a.name,root.depth, b.name) for b in root.leaves if a != b]
        else:
            siblings = []
        if root.children is not None:
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
    siblings['sorted_row'] = [sorted([a, b]) for a, b in zip(siblings.node1, siblings.node2)]
    siblings['sorted_row'] = siblings['time'].astype(str) + siblings['sorted_row'].astype(str)
    siblings = siblings.drop_duplicates(subset="sorted_row")
    siblings = siblings.drop("sorted_row", axis=1)
    logging.info("Got siblings set")

    sib = (
        siblings["node1"].astype(str)
        + "."
        + siblings["node2"].astype(str)
        + ".T"
        + siblings["time"].astype(str)
    )
    sib = sib.to_list()

    riskaversionparameter = 0.2
    sib = [l + ' \n' * (n % 300 == 0) for n, l in enumerate(sib)]
    scen_probs = ', \n'.join([str(k) + " " + "{:.7f}".format(v) for k, v in scenario_probabilities.items()])
    set_s = ", \n".join([leaf.name for leaf in leaves])
    set_siblings = ', \n'.join(sib)
    _str = f"""Option LP=CPLEX;

$onecho > cplex.opt
preind 1
names no
$offecho

Set       s        / {set_s} / ;
Set       i        / {', '.join([f'stock{n + 1}' for n in range(len(node.returns))])} / ;
Set       t        / {', '.join([f'T{n}' for n in range((root.height + 1))])} / ;

Alias (s, ss);
Set siblings /{set_siblings}/
;

Table
AssetYields(i, t, s)
{asset_yields_string}
;

Parameter
ScenarioProbabilities(s)
/{scen_probs}/
;
Parameter
InitialWealth /1/;
Parameter
RiskAversionParameter /{riskaversionparameter}/;
Parameter
alpha /{alpha}/;

Positive Variables x(i,s,t);
x.l(i,s,t)=1/card(i);
Variable loss;
Positive Variables y(s);
Variable wealth(s,t);
Variable z;
Variable v;
Variable TotalExcessScenarioReturns(s);

Equations          rootvariablessumuptoinitialwealth, nonanticipativityofx, eqwealth, eqinitialwealth, eqassetyields, objective, yconstrs, ExcessScenarioReturns(s,t);

rootvariablessumuptoinitialwealth(s).. InitialWealth=e=sum(i,x(i,s,"T0"));
eqinitialwealth(s).. wealth(s,"T0") =e= InitialWealth;

eqwealth(s, t)$(ord(t) <> card(t)).. wealth(s,t) =e= sum(i,x(i,s,t));
eqassetyields(s,t)$(ord(t)>1).. wealth(s,t) =e= sum(i, x(i,s,t-1)*AssetYields(i,t,s));

ExcessScenarioReturns(s,t)$(ord(t)=card(t)).. TotalExcessScenarioReturns(s)=e=wealth(s,t)-1;
**Previous row has assumption.. wealth(s,"T0")=1!
yconstrs(s).. y(s)=g=-TotalExcessScenarioReturns(s)-z;

nonanticipativityofx(i,s,ss,t)$(siblings(s,ss,t))..  x(i,s,t) =e= x(i,ss,t);

objective.. loss=e=sum(s,ScenarioProbabilities(s)*TotalExcessScenarioReturns(s))+RiskAversionParameter*(z+(1/(1-alpha))*sum(s,ScenarioProbabilities(s)*y(s)));

Model problem / ALL /;

problem.OptFile=1;

solve problem using LP minimising loss;
"""

    return _str


def calculate_mean_cvar_over_leaves(
    root: Node, alpha, create_model_str_func: Callable = create_cvar_gams_model_str
) -> float:
    gms = GamsWorkspace(system_directory=r"C:\GAMS\40")
    # with open('root.pickle', 'wb') as handle:
    #     pickle.dump(root, handle, protocol=pickle.HIGHEST_PROTOCOL)
    logging.info("Creating Cvar model string")
    cvar_model_str = create_model_str_func(root, alpha)
    # with open("cvar_model_string.txt", "w") as text_file:
    #     text_file.write(cvar_model_str)
    output_stream = io.StringIO()
    job = gms.add_job_from_string(cvar_model_str)
    logging.info("Solving Cvar model")
    job.run(output=output_stream)
    output = output_stream.getvalue()

    for rec in job.out_db["loss"]:
        loss = rec.level
    assert "Optimal solution found" in output or "** Feasible solution" in output
    assert "*** Status: Normal completion" in output
    logging.info("Cvar model solved")
    return loss


#3**8 scenarios - 50s in total
# building tree - 4s, building model string - 12s (11s building asset yields table, 1s nonanticipativity constraints), 34s solving model (25s building model, 9s solving)
#3**7 scenarios - 10s
# 3s building tree, building model string takes 4s (3,5s creating asset yields table, 1s creating nonanticipativity constraints), 3s solve