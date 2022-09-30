from copy import deepcopy

import numpy as np
from anytree import Node, PreOrderIter, LevelOrderGroupIter
import io

from mpire import WorkerPool
import pandas as pd
from gams import GamsWorkspace
from tabulate import tabulate
from src.configuration import CVAR_ALPHA, TICKERS
import sys

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
    sib = siblings["node1"].astype(str) + "." + siblings["node2"].astype(str) + ".T" + siblings["time"].astype(str)
    sib = sib.to_list()
    _str = f"""
Option LP=CPLEX;


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
x.l(i,s,t)=1/card(i);
Variable loss;

Equations          rootvariablessumuptoinitialwealth, nonanticipativityofx, xlessthanone, objective;

rootvariablessumuptoinitialwealth(s).. InitialWealth=e=sum(i,x(i,s,"T0"));

nonanticipativityofx(i,s,ss,t)$(siblings(s,ss,t))..  x(i,s,t) =e= x(i,ss,t);
objective.. loss=e= sum(i,(sum(s,sum(t,x(i,s,t)))));
xlessthanone(i,s,t).. x(i,s,t) =l= 1

Model problem / ALL /;

solve problem using LP minimising loss;
"""

    print(pd.to_datetime("now"))
    # print(_str)
    return _str

def calculate_mean_cvar_over_leaves(root: Node) -> float:
    gms = GamsWorkspace(system_directory="/opt/gams/gams40.2_linux_x64_64_sfx")
    cvar_model_str = create_cvar_gams_model_str(root)
    output_stream = io.StringIO()
    job = gms.add_job_from_string(cvar_model_str)
    job.run(output=output_stream)
    x = {}
    p = {}
    output = output_stream.getvalue()

    for rec in job.out_db["loss"]:
        loss = rec.level
    assert "Optimal solution found" in output or "** Feasible solution" in output
    assert "*** Status: Normal completion" in output

