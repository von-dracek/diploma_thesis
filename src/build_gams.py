import io

import pandas as pd
from gams import GamsWorkspace
from tabulate import tabulate

#
# sys.path.append("/opt/gams/gams40.2_linux_x64_64_sfx/apifiles/Python/api_38")
# sys.path.append("/opt/gams/gams40.2_linux_x64_64_sfx/apifiles/Python/gams")


def prepare_moment_matching_model_str(n_nodes, TARMOM, R):
    n_stocks = R.shape[0]
    headers = [f"stock{n+1}" for n in range(n_stocks)]
    rowIDs = ["1", "2", "3", "4"]
    TARMOM = tabulate(
        TARMOM, showindex=rowIDs, headers=headers, tablefmt="plain", numalign="right"
    )
    R = tabulate(R, showindex=headers, headers=headers, tablefmt="plain", numalign="right")
    return f"""Option NLP=CONOPT;
Option Seed=1337;
Set       j        / node1*node{n_nodes} /
          i        / stock1*stock{n_stocks} /
          k        / 1*4 /;
Alias(i,s);

Variables x(i,j);
Variables loss;
Variables mean(i), variance(i), third(i), fourth(i), corr(i,s);
Positive Variables p(j);


Table     TARMOM(k,i)
{TARMOM}
;

Table TARCORR(i,s)
{R}
;
x.l(i,j)=1*uniform(0,2);
p.l(j)=1/card(j);
p.lo(j)=0.1;
**minimum probability of 10 percent
variance.l(i)=0.01;

Equations          objective, sumuptoone, m1, m2, m3, m4, correl_eq;

m1(i).. mean(i) =e= sum(j, x(i,j)*p(j));
m2(i).. variance(i) =e= sum(j, power((x(i,j)-mean(i)),2)*p(j));
m3(i).. third(i) =e= sum(j, power((x(i,j)-mean(i)),3)*p(j));
m4(i).. fourth(i) =e= sum(j, power((x(i,j)-mean(i)),4)*p(j));
correl_eq(i,s).. corr(i,s) =e= (sum(j, x(i,j)*x(s,j)*p(j))-mean(s)*mean(i))/(sqrt(variance(s)*variance(i)));

objective..      loss =e= sum(i,power(mean(i)-TARMOM("1",i),2))
                        + sum(i, power(variance(i)-TARMOM("2",i),2))
                        + sum(i, power(third(i)-TARMOM("3",i),2))
                        + sum(i, power(fourth(i)-TARMOM("4",i),2))
                        + sum(s,sum(i$(ord(i)>ord(s)), power(corr(i,s)-TARCORR(i,s),2)));
sumuptoone..     1 =e= sum(j, p(j));

Model problem / objective, sumuptoone, m1, m2, m3, m4, correl_eq /;

solve problem using NLP minimising loss;
        """  # noqa: E501



# https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.497.113&rep=rep1&type=pdf
# Data-Driven Multi-Stage Scenario Tree Generation via Statistical Property
# and Distribution Matching
# Bruno A. Calfa∗, Anshul Agarwal†, Ignacio E. Grossmann∗, John M. Wassick†

def build_mm_model(n_nodes, TARMOM, R):
    gms = GamsWorkspace(system_directory=r"C:\GAMS\40")
    model_str = prepare_moment_matching_model_str(n_nodes, TARMOM, R)
    output_stream = io.StringIO()
    job = gms.add_job_from_string(model_str)
    job.run(output=output_stream)
    x = {}
    p = {}
    for rec in job.out_db["x"]:
        x[rec.keys[0]] = {}
    for rec in job.out_db["x"]:
        x[rec.keys[0]][rec.keys[1]] = rec.level
    for rec in job.out_db["p"]:
        p[rec.keys[0]] = rec.level
    output = output_stream.getvalue()
    assert "** Optimal solution" in output or "** Feasible solution" in output
    assert "*** Status: Normal completion" in output
    assert all((_p >= 0) and (_p <= 1) for _p in p.values())
    x = pd.DataFrame.from_dict(x).T
    p = pd.Series(p)
    x = x.to_numpy()
    p = p.to_numpy()
    return x, p
