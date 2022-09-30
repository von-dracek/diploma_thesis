from copy import deepcopy

import numpy as np
from anytree import Node, PreOrderIter, LevelOrderGroupIter
import io

from mpire import WorkerPool
import pandas as pd
from gams import GamsWorkspace
from tabulate import tabulate
from src.configuration import CVAR_ALPHA, TICKERS

def paralellized_create_gams_model_str(root):
    n_jobs = 6  # TODO: parametrize this from command line
    specified_row_name_length = 20
    print(pd.to_datetime("now"))
    def create_scenario_df(leaf):
        scenario_series = []
        for node in (leaf.ancestors + (leaf,)):
            if node.depth > 0:
                ser = pd.Series(node.returns, index=[f"stock{n + 1}" + " " * (
                            specified_row_name_length - len(f"stock{n + 1}") - len(
                        f".T{node.depth}")) + f".T{node.depth}" for n in range(len(node.returns))])
                scenario_series.append(ser)
        scenario = pd.concat(scenario_series)
        scenario.name = leaf.name
        return scenario

    # with WorkerPool(n_jobs=n_jobs) as pool:
    #     scenarios = pool.map(create_scenario_df, root.leaves)
    scenarios = [create_scenario_df(leaf) for leaf in root.leaves]
    scenario_df = pd.concat(scenarios, axis=1)
    _str = f"""
    Table
    AssetYields(i, t, s)
    {tabulate(scenario_df, headers="keys", tablefmt="plain", numalign="right", showindex=True, floatfmt=".4f")}
    ;
    """
    print(pd.to_datetime("now"))
    return _str


def create_cvar_gams_model_str(root):
    leaves = root.leaves
    specified_row_name_length = 20
    scenarios = []
    print(pd.to_datetime("now"))
    for leaf in leaves:
        scenario_series = []
        for node in (leaf.ancestors + (leaf,)):
            if node.depth>0:
                ser = pd.Series(node.returns, index=[f"stock{n+1}"+" "*(specified_row_name_length-len(f"stock{n+1}")-len(f".T{node.depth}"))+f".T{node.depth}" for n in range(len(node.returns))])
                scenario_series.append(ser)
        scenario = pd.concat(scenario_series)
        scenario.name = leaf.name
        scenarios.append(scenario)
    scenario_df = pd.concat(scenarios, axis=1)
    scenario_probabilities = {leaf.name:float(np.prod([node.probability for node in (leaf,) + leaf.ancestors ])) for leaf in leaves}
    assert abs(sum(scenario_probabilities.values()) - 1) < 1e-4
    print(pd.to_datetime("now"))
    def get_siblings(root):
        siblings = [(a.name,root.depth, b.name) for a in root.leaves for b in root.leaves]
        for node in root.children:
            siblings += get_siblings(node)
        return siblings
    siblings = get_siblings(root)
    ssiblings = [{"node1":sibling[0], "time":sibling[1], "node2":sibling[2]} for sibling in siblings if sibling[0]!=sibling[2]]
    siblings = pd.DataFrame(ssiblings)
    # df = pd.DataFrame(False, index=[leaf.name for leaf in root.leaves], columns=[leaf.name for leaf in root.leaves])
    # dfs = []
    # for t in range(root.height + 1):
    #     df.loc[:,"time"] = t
    #     _df = deepcopy(df)
    #     dfs.append(_df)
    # df = pd.concat(dfs)
    # df = df.set_index("time", append=True)
    #
    # print(pd.to_datetime("now"))
    # # df.loc[siblings.index.to_list(), siblings.iloc[:, 0].to_list()] = True
    # for i, row in siblings.iterrows():
    #     df.loc[i,row.iloc[0]]=True
    # df = df * 1
    # df.index = [
    #     "   .T".join(map(str, col)) if not isinstance(col, str) else col for col in df.index
    # ]
    sib = siblings["node1"].astype(str) + "." + siblings["node2"].astype(str) + ".T" + siblings["time"].astype(str)
    sib = sib.to_list()
    _str = f"""
Option LP=CPLEX;
option limrow = 10000 ;


Set       s        / {", ".join([leaf.name for leaf in leaves])} / ;
Set       i        / {', '.join([f'stock{n+1}' for n in range(len(node.returns))])} / ;
Set       t        / {', '.join([f'T{n}' for n in range((root.height + 1))])} / ;
    
Alias (s, ss);
Set siblings /{', '.join(sib)}/
;

Table
AssetYields(i, t, s)
{tabulate(scenario_df,headers="keys", tablefmt="plain", numalign="right", showindex=True, floatfmt=".4f")}
;

Parameter
ScenarioProbabilities(s)
/{', '.join([str(k) + " " + "{:.7f}".format(v) for k,v in scenario_probabilities.items()])}/
;
Parameter
InitialWealth /1/;

Positive Variables x(i,s,t);
Variable loss;

Equations          rootvariablessumuptoinitialwealth, nonanticipativityofx, objective;

rootvariablessumuptoinitialwealth(s).. InitialWealth=e=sum(i,x(i,s,"T0"));

nonanticipativityofx(i,t,s,ss)$(siblings(s,ss,t))..  x(i,s,t) =e= x(i,ss,t);
objective.. loss=e= sum(i,(sum(s,sum(t,x(i,s,t)))));

Model problem / ALL /;

solve problem using LP minimising loss;
"""

    print(pd.to_datetime("now"))
    # print(_str)
    return _str

def calculate_mean_cvar_over_leaves(root: Node) -> float:
    # paralellized_create_gams_model_str(root)
    create_cvar_gams_model_str(root)
