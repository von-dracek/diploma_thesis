from copy import deepcopy

import numpy as np
from anytree import Node, PreOrderIter, LevelOrderGroupIter
import io

from mpire import WorkerPool
import pandas as pd
from gams import GamsWorkspace, DebugLevel
from tabulate import tabulate
from src.configuration import CVAR_ALPHA, TICKERS
import sys

def create_cvar_gams_model_str(root):
    leaves = root.leaves
    specified_row_name_length = 20
    scenarios = []
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
MinimumExpectedReturn /0.5/;
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
    return loss




#Old way of generating model string - using L shaped algorithm
#     _str = f"""
# Option LP=CPLEX;
#
#
# Set       s        / {", ".join([leaf.name for leaf in leaves])} / ;
# Set       i        / {', '.join([f'stock{n+1}' for n in range(len(node.returns))])} / ;
# Set       t        / {', '.join([f'T{n}' for n in range((root.height + 1))])} / ;
#
# Alias (s, ss);
# Set siblings /{', '.join(sib)}/
# ;
#
# Table
# AssetYields(i, t, s)
# {tabulate(scenario_df,headers="keys", tablefmt="plain", numalign="right", showindex=True, floatfmt=".4f")}
# ;
#
# Parameter
# ScenarioProbabilities(s)
# /{', '.join([str(k) + " " + "{:.7f}".format(v) for k,v in scenario_probabilities.items()])}/
# ;
# Parameter
# InitialWealth /1/;
# Parameter
# lambda /1/;
# Parameter
# alpha /0.05/;
# Parameter
# benchmarkwealth /2/;
#
# Positive Variables x(i,s,t);
# x.l(i,s,t)=1/card(i);
# Variable loss;
# Variable wealth(s,t);
# Variable z;
# Variable v;
#
# Equations          rootvariablessumuptoinitialwealth, nonanticipativityofx, eqwealth, eqinitialwealth, eqassetyields, benchmark, objective;
#
# rootvariablessumuptoinitialwealth(s).. InitialWealth=e=sum(i,x(i,s,"T0"));
# eqinitialwealth(s).. wealth(s,"T0") =e= InitialWealth;
#
# eqwealth(s, t).. wealth(s,t) =e= sum(i,x(i,s,t));
# eqassetyields(s,t)$(ord(t)>1).. wealth(s,t) =e= sum(i, x(i,s,t-1)*AssetYields(i,t,s));
# benchmark(t)$(ord(t)=card(t))..sum(s,ScenarioProbabilities(s)*(benchmarkwealth-wealth(s,t) - z)) =l= v;
#
#
# nonanticipativityofx(i,s,ss,t)$(siblings(s,ss,t))..  x(i,s,t) =e= x(i,ss,t);
#
#
# objective(t)$(ord(t)=card(t)).. loss=e= -(1/lambda)*sum(s, ScenarioProbabilities(s)*(wealth(s,t))) + z +(1/(1-alpha))*v;
#
# Model problem / ALL /;
#
# solve problem using LP minimising loss;
# """