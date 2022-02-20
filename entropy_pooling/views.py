from sys import builtin_module_names
import pandas as pd
import numpy as np
import math
from cvxopt import matrix, solvers # pip install cvxopt



# TODO:
# Talk about the views.xlsx
# Implement weights for the views
# Test for bugs!
# Test for unexpected inputs. Anything can happen!
# We could do try/except for each row. The software will then alert and ignore weird rows.
# Infeasible constraints (e.g. Example Equity = 0.5 % and Example Equity > 0.6 % is an infeasible combination)
# Clean the code... it's very messy right now...



# Column names. LHS has code names used in this script. RHS are the ones used in the Excel-file
# This is not very good implementation, because this breaks if the used changes the column names in the views.xlsx.
# Should we use the column numbers instead?
cols = {'type'      : '*View type (mean/cov)',
        'asset1'    : '*Asset 1',
        'asset2'    : 'Asset 2 (only cov)',
        'eq_ineq'   : '*eq / ineq',
        'parameter' : '*Parameter (% or cov)',
        'asset3'    : 'Asset 3',
        'asset4'    : 'Asset 4 (only cov)',
        'weight'    : 'Weight (0-100%)'
}

def load(data = pd.read_excel("data.xlsx")):
    # Returns a tuple (A,b,C,d), which contains the constraints Ax = b and Cx >= d.
    # input 'data' contains the returns data

    df = pd.read_excel("views.xlsx")

    # Initialize output matrices and vectors
    A = np.zeros((0,len(data)))
    b = np.zeros((0,1))
    C = np.zeros((0,len(data)))
    d = np.zeros((0,1))

    # MEAN CONSTRAINTS

    for ind in range(len(df)):
        rf = row_format(ind,df)
        if rf == 'mean_eq':
            # Append values to matrix A and vector b
            A = np.vstack([A, +data.iloc[:][df.iloc[ind][cols['asset1']]]])
            b = np.vstack([b, +df.iloc[ind][cols['parameter']]])
        elif rf == "mean_geq":
            C = np.vstack([C, +data.iloc[:][df.iloc[ind][cols['asset1']]]])
            d = np.vstack([d, +df.iloc[ind][cols['parameter']]])
        elif rf == "mean_leq":
            C = np.vstack([C, -data.iloc[:][df.iloc[ind][cols['asset1']]]])
            d = np.vstack([d, -df.iloc[ind][cols['parameter']]])
        elif rf == "rel_mean_eq":
            mat_element = data.iloc[:][df.iloc[ind][cols['asset1']]] - (1 + df.iloc[ind][cols['parameter']])*data.iloc[:][df.iloc[ind][cols['asset3']]]
            A = np.vstack([A, +mat_element])
            b = np.vstack([b, 0.0])
        elif rf == "rel_mean_geq":
            mat_element = data.iloc[:][df.iloc[ind][cols['asset1']]] - (1 + df.iloc[ind][cols['parameter']])*data.iloc[:][df.iloc[ind][cols['asset3']]]
            C = np.vstack([C, +mat_element])
            d = np.vstack([d, 0.0])
        elif rf == "rel_mean_leq":
            mat_element = data.iloc[:][df.iloc[ind][cols['asset1']]] - (1 + df.iloc[ind][cols['parameter']])*data.iloc[:][df.iloc[ind][cols['asset3']]]
            C = np.vstack([C, -mat_element])
            d = np.vstack([d, 0.0])
    
    # Gather mean-constraints and solve a feasible set of mean values
    # These are needed for linear covariance constraints
    # We use cvxopt:
    # http://cvxopt.org/userguide/coneprog.html?highlight=lp#quadratic-cone-programs
    (A_mu, b_mu, G, h) = constraints_mu(data, df)
    q = np.matrix(-data.mean().to_numpy()).transpose()
    q = matrix(np.nan_to_num(q)) # nan values to 0
    P = matrix(np.eye(data.shape[1]))
    G = matrix(G)
    h = matrix(h)
    A_mu = matrix(A_mu)
    b_mu = matrix(b_mu)
    dims = {'l': G.size[0], 'q': [], 's': []} # component-wise inequality constraint (default for cvxopt)
    # Let's solve   min_mu ||mu_0 - mu||
    #               s.t.     A*mu == b
    #                        G*mu >= h (element wise)
    mu = solvers.coneqp(P, q, G, h, dims, A_mu, b_mu)
    mu_df = pd.DataFrame(mu['x'])
    mu_df = pd.DataFrame(data=mu_df.T.values, columns=data.columns)

    # COVARIANCE CONSTRAINTS

    for ind in range(len(df)):
        rf = row_format(ind,df)
        if rf == 'cov_eq':
            # df.mul(df2) multiplies two dataframes
            A = np.vstack([A, +cov_vector(data, mu_df, df.iloc[ind][cols['asset1']], df.iloc[ind][cols['asset2']])])
            b = np.vstack([b, +df.iloc[ind][cols['parameter']]])
        if rf == 'cov_geq':
            C = np.vstack([C, +cov_vector(data, mu_df, df.iloc[ind][cols['asset1']], df.iloc[ind][cols['asset2']])])
            d = np.vstack([d, +df.iloc[ind][cols['parameter']]])
        if rf == 'cov_leq':
            C = np.vstack([C, -cov_vector(data, mu_df, df.iloc[ind][cols['asset1']], df.iloc[ind][cols['asset2']])])
            d = np.vstack([d, -df.iloc[ind][cols['parameter']]])
        if rf == 'rel_cov_eq':
            row = cov_vector(data, mu_df, df.iloc[ind][cols['asset1']], df.iloc[ind][cols['asset2']])
            row = row - (1 + df.iloc[ind][cols['parameter']]) * cov_vector(data, mu_df, df.iloc[ind][cols['asset3']], df.iloc[ind][cols['asset4']])
            A = np.vstack([A, +row])
            b = np.vstack([b, 0.0])
        if rf == 'rel_cov_geq':
            row = cov_vector(data, mu_df, df.iloc[ind][cols['asset1']], df.iloc[ind][cols['asset2']])
            row = row - (1 + df.iloc[ind][cols['parameter']]) * cov_vector(data, mu_df, df.iloc[ind][cols['asset3']], df.iloc[ind][cols['asset4']])
            C = np.vstack([C, +row])
            d = np.vstack([d, 0.0])
        if rf == 'rel_cov_leq':
            row = cov_vector(data, mu_df, df.iloc[ind][cols['asset1']], df.iloc[ind][cols['asset2']])
            row = row - (1 + df.iloc[ind][cols['parameter']]) * cov_vector(data, mu_df, df.iloc[ind][cols['asset3']], df.iloc[ind][cols['asset4']])
            C = np.vstack([C, -row])
            d = np.vstack([d, 0.0])

    return (A,b,C,d)

def cov_vector(data, mu_df, name1, name2):
    a = data.iloc[:][name1] - mu_df[name1].values
    b = data.iloc[:][name2] - mu_df[name2].values
    return a.mul(b)

# Function that reads the format of the row (e.g. non-relative mean equality)
def row_format(ind, df = pd.read_excel("views.xlsx")):
    if df.iloc[ind][cols['type']]=='mean' and isinstance(df.iloc[ind][cols['asset3']], float) and math.isnan(df.iloc[ind][cols['asset3']]) and df.iloc[ind][cols['eq_ineq']]=='=':
        return "mean_eq"
    if df.iloc[ind][cols['type']]=='mean' and isinstance(df.iloc[ind][cols['asset3']], float) and math.isnan(df.iloc[ind][cols['asset3']]) and df.iloc[ind][cols['eq_ineq']]=='<':
        return "mean_leq"
    if df.iloc[ind][cols['type']]=='mean' and isinstance(df.iloc[ind][cols['asset3']], float) and math.isnan(df.iloc[ind][cols['asset3']]) and df.iloc[ind][cols['eq_ineq']]=='>':
        return "mean_geq"
    if df.iloc[ind][cols['type']]=='mean' and df.iloc[ind][cols['eq_ineq']]=='=':
        return "rel_mean_eq"
    if df.iloc[ind][cols['type']]=='mean' and df.iloc[ind][cols['eq_ineq']]=='>':
        return "rel_mean_geq"
    if df.iloc[ind][cols['type']]=='mean' and df.iloc[ind][cols['eq_ineq']]=='<':
        return "rel_mean_leq"
    if df.iloc[ind][cols['type']]=='cov' and isinstance(df.iloc[ind][cols['asset3']], float) and math.isnan(df.iloc[ind][cols['asset3']]) and df.iloc[ind][cols['eq_ineq']]=='=':
        return "cov_eq"
    if df.iloc[ind][cols['type']]=='cov' and isinstance(df.iloc[ind][cols['asset3']], float) and math.isnan(df.iloc[ind][cols['asset3']]) and df.iloc[ind][cols['eq_ineq']]=='<':
        return "cov_leq"
    if df.iloc[ind][cols['type']]=='cov' and isinstance(df.iloc[ind][cols['asset3']], float) and math.isnan(df.iloc[ind][cols['asset3']]) and df.iloc[ind][cols['eq_ineq']]=='>':
        return "cov_geq"
    if df.iloc[ind][cols['type']]=='cov' and df.iloc[ind][cols['eq_ineq']]=='=':
        return "rel_cov_eq"
    if df.iloc[ind][cols['type']]=='cov' and df.iloc[ind][cols['eq_ineq']]=='>':
        return "rel_cov_geq"
    if df.iloc[ind][cols['type']]=='cov' and df.iloc[ind][cols['eq_ineq']]=='<':
        return "rel_cov_leq"
    return ""

def constraints_mu(data = pd.read_excel("data.xlsx"), df = pd.read_excel("views.xlsx")):
    A = np.zeros((0,data.shape[1]))
    b = np.zeros((0,1))
    C = np.zeros((0,data.shape[1]))
    d = np.zeros((0,1))

    for ind in range(len(df)):
        rf = row_format(ind,df)
        if rf == 'mean_eq':
            # Append values to matrix A and vector b
            A = np.vstack([A, np.zeros((1,data.shape[1]))])
            A[-1][data.columns.get_loc(df.iloc[ind][cols['asset1']])] = +1
            b = np.vstack([b, df.iloc[ind][cols['parameter']]])
        elif rf == "mean_geq":
            C = np.vstack([C, np.zeros((1,data.shape[1]))])
            C[-1][data.columns.get_loc(df.iloc[ind][cols['asset1']])] = -1
            d = np.vstack([d, df.iloc[ind][cols['parameter']]])
        elif rf == "mean_leq":
            C = np.vstack([C, np.zeros((1,data.shape[1]))])
            C[-1][data.columns.get_loc(df.iloc[ind][cols['asset1']])] = +1
            d = np.vstack([d, -df.iloc[ind][cols['parameter']]])
        elif rf == "rel_mean_eq":
            A = np.vstack([A, np.zeros((1,data.shape[1]))])
            A[-1][data.columns.get_loc(df.iloc[ind][cols['asset1']])] = +1
            A[-1][data.columns.get_loc(df.iloc[ind][cols['asset3']])] = -(1 + df.iloc[ind][cols['parameter']])
            b = np.vstack([b, 0.0])
        elif rf == "rel_mean_geq":
            C = np.vstack([C, np.zeros((1,data.shape[1]))])
            C[-1][data.columns.get_loc(df.iloc[ind][cols['asset1']])] = -1
            C[-1][data.columns.get_loc(df.iloc[ind][cols['asset3']])] = +(1 + df.iloc[ind][cols['parameter']])
            d = np.vstack([d, 0.0])
        elif rf == "rel_mean_leq":
            C = np.vstack([C, np.zeros((1,data.shape[1]))])
            C[-1][data.columns.get_loc(df.iloc[ind][cols['asset1']])] = +1
            C[-1][data.columns.get_loc(df.iloc[ind][cols['asset3']])] = -(1 + df.iloc[ind][cols['parameter']])
            d = np.vstack([d, 0.0])
    return (A,b,C,d)