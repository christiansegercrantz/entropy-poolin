from sys import builtin_module_names
import pandas as pd
import numpy as np
import math
from cvxopt import matrix, solvers # pip install cvxopt
#from mosek import iparam



# TODO:
# Talk about the views.xlsx
# Implement weights for the views
# Test for bugs!
# Test for unexpected inputs. Anything can happen!
# We could do try/except for each row. The software will then alert and ignore weird rows.
# Infeasible constraints (e.g. Example Equity = 0.5 % and Example Equity > 0.6 % is an infeasible combination)
# Clean the code... it's very messy right now...



# Below are the column names. LHS contains the code names used in this script. RHS contains the ones used in the Excel-file.
# This is not very good implementation, because this breaks if the user changes any column name in the views.xlsx.
# Should we use the column numbers instead?
cols = {'type'      : '* View on',
        'asset1'    : '* Risk factor 1',
        'asset2'    : 'Risk factor 2 \n(applicable for corr)',
        'eq_ineq'   : '* Operator',
        'parameter' : '* Constant \n(alpha)',
        'asset3'    : 'Risk factor 3',
        'asset4'    : 'Risk factor 4 \n(applicable for corr)' # pitää lisätä vielä beta!
}

# load() IS THE MAIN FUNCTION
# It returns a tuple (A,b,C,d), which contain the constraints Ax = b and Cx >= d.

def load(data = pd.read_excel("data.xlsx")):

    df = pd.read_excel("views.xlsx")

    # Initialize output matrices and vectors
    A = np.ones((1,len(data))) # ones because sum(x_i) = 1
    b = np.ones((1,1)) # one because sum(x_i) = 1
    C = np.zeros((0,len(data)))
    d = np.zeros((0,1))

    # MEAN CONSTRAINTS

    for ind in range(len(df)):
        rf = row_format(ind,df)
        if rf == 'mean_eq':
            (A,b) = append_mean(A, b, data, df, ind, +1)
        elif rf == "mean_geq":
            (C,d) = append_mean(C, d, data, df, ind, +1)
        elif rf == "mean_leq":
            (C,d) = append_mean(C, d, data, df, ind, -1)
        elif rf == "rel_mean_eq":
            (A,b) = append_rel_mean(A, b, data, df, ind, +1)
        elif rf == "rel_mean_geq":
            (C,d) = append_rel_mean(C, d, data, df, ind, +1)
        elif rf == "rel_mean_leq":
            (C,d) = append_rel_mean(C, d, data, df, ind, -1)
    
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
    # Let's solve   min_mu ||mu_0 - mu||**2
    #               s.t.     A*mu == b
    #                        G*mu <= h (element wise) (note leq instead of geq!)
    print('Solving feasible mean values, which we give to covariance constraints...')
    mu = solvers.coneqp(P, q, G, h, dims, A_mu, b_mu)
    mu_df = pd.DataFrame(mu['x'])
    mu_df = pd.DataFrame(data=mu_df.T.values, columns=data.columns)

    # VOLATILITY CONSTRAINTS
    for ind in range(len(df)):
        rf = row_format(ind,df)
        if rf == 'vol_eq':
            (A,b) = append_vol(A, b, data, mu_df, df, ind, +1)
        elif rf == 'vol_geq':
            (C,d) = append_vol(C, d, data, mu_df, df, ind, +1)
        elif rf == 'vol_leq':
            (C,d) = append_vol(C, d, data, mu_df, df, ind, -1)
        elif rf == 'rel_vol_eq':
            (A,b) = append_rel_vol(A, b, data, mu_df, df, ind, +1)
        elif rf == 'rel_vol_geq':
            (C,d) = append_rel_vol(C, d, data, mu_df, df, ind, +1)
        elif rf == 'rel_vol_leq':
            (C,d) = append_rel_vol(C, d, data, mu_df, df, ind, -1)

    (A_vol, b_vol, G, h) = constraints_vol(data, df)
    q = np.matrix(-data.var().to_numpy()).transpose()
    q = matrix(np.nan_to_num(q)) # nan values to 0
    P = matrix(np.eye(data.shape[1]))
    G = matrix(G)
    h = matrix(h)
    A_vol = matrix(A_vol)
    b_vol = matrix(b_vol)
    dims = {'l': G.size[0], 'q': [], 's': []} # component-wise inequality constraint (default for cvxopt)
    # Let's solve   min_vol ||vol_0 - vol||**2
    #               s.t.     A*vol == b
    #                        G*vol <= h (element wise) (note leq instead of geq!)
    print('Solving feasible mean values, which we give to covariance constraints...')
    vol = solvers.coneqp(P, q, G, h, dims, A_vol, b_vol)
    vol_df = pd.DataFrame(vol['x'])
    vol_df = pd.DataFrame(data=vol_df.T.values, columns=data.columns)

    # COVARIANCE CONSTRAINTS

    for ind in range(len(df)):
        rf = row_format(ind,df)
        if rf == 'corr_eq':
            (A,b) = append_corr(A, b, data, mu_df, vol_df, df, ind, +1)
        elif rf == 'corr_geq':
            (C,d) = append_corr(C, d, data, mu_df, vol_df, df, ind, +1)
        elif rf == 'corr_leq':
            (C,d) = append_corr(C, d, data, mu_df, vol_df, df, ind, -1)
        elif rf == 'rel_corr_eq':
            (A,b) = append_rel_corr(A, b, data, mu_df, vol_df, df, ind, +1)
        elif rf == 'rel_corr_geq':
            (C,d) = append_rel_corr(C, d, data, mu_df, vol_df, df, ind, +1)
        elif rf == 'rel_corr_leq':
            (C,d) = append_rel_corr(C, d, data, mu_df, vol_df, df, ind, -1)

    return (A,b,C,d)

'''

'''

def cov_vector(data, mu_df, name1, name2):
    a = data.iloc[:][name1] - mu_df[name1].values
    b = data.iloc[:][name2] - mu_df[name2].values
    return a.mul(b)

def returns_to_monthly(r):
    return (r+1)**(1/12)-1

def append_mean(C, d, data, df, ind, sign):
    mat_row = data.iloc[:][df.iloc[ind][cols['asset1']]]
    # The last line appends new elements to C and d
    return (np.vstack([C, sign*mat_row]), np.vstack([d, returns_to_monthly(sign*df.iloc[ind][cols['parameter']])]))

def append_rel_mean(C, d, data, df, ind, sign):
    mat_row = data.iloc[:][df.iloc[ind][cols['asset1']]] - data.iloc[:][df.iloc[ind][cols['asset3']]]
    return (np.vstack([C, sign*mat_row]), np.vstack([d, returns_to_monthly(sign*df.iloc[ind][cols['parameter']])]))

def append_rel_corr(C, d, data, mu_df, vol_df, df, ind, sign):
    row =       cov_vector(data, mu_df, df.iloc[ind][cols['asset1']], df.iloc[ind][cols['asset2']]) / np.sqrt(vol_df[df.iloc[ind][cols['asset1']]].values * vol_df[df.iloc[ind][cols['asset2']]].values)
    row = row - cov_vector(data, mu_df, df.iloc[ind][cols['asset3']], df.iloc[ind][cols['asset4']]) / np.sqrt(vol_df[df.iloc[ind][cols['asset3']]].values * vol_df[df.iloc[ind][cols['asset4']]].values)
    return (np.vstack([C, sign*row]), np.vstack([d, sign*df.iloc[ind][cols['parameter']]]))

def append_corr(C, d, data, mu_df, vol_df, df, ind, sign):
    row = cov_vector(data, mu_df, df.iloc[ind][cols['asset1']], df.iloc[ind][cols['asset2']]) / np.sqrt(vol_df[df.iloc[ind][cols['asset1']]].values * vol_df[df.iloc[ind][cols['asset2']]].values)
    return (np.vstack([C, sign*row]), np.vstack([d, sign*df.iloc[ind][cols['parameter']]]))

def append_vol(C, d, data, mu_df, df, ind, sign): # really, a constraint of covariance
    row = cov_vector(data, mu_df, df.iloc[ind][cols['asset1']], df.iloc[ind][cols['asset1']])
    return (np.vstack([C, sign*row]), np.vstack([d, sign*df.iloc[ind][cols['parameter']]**2 / 12])) # 1/12 to annualize volatility

def append_rel_vol(C, d, data, mu_df, df, ind, sign):
    row =       cov_vector(data, mu_df, df.iloc[ind][cols['asset1']], df.iloc[ind][cols['asset1']])
    row = row - cov_vector(data, mu_df, df.iloc[ind][cols['asset3']], df.iloc[ind][cols['asset3']])
    return (np.vstack([C, sign*row]), np.vstack([d, sign*df.iloc[ind][cols['parameter']]**2 / 12])) # 1/12 to annualize volatility

# Function that reads the format of the row (e.g. non-relative mean equality)
def row_format(ind, df = pd.read_excel("views.xlsx")):
    # If Excel cell 'asset3' is left empty, we are dealing with absolute view
    # Otherwise relative view
    if isinstance(df.iloc[ind][cols['asset3']], float) and math.isnan(df.iloc[ind][cols['asset3']]):
        res = ""
    else:
        res = "rel_"
    
    if df.iloc[ind][cols['type']]=='Mean':
        res = res + 'mean_'
    elif df.iloc[ind][cols['type']]=='Vol':
        res = res + 'vol_'
    elif df.iloc[ind][cols['type']]=='Corr':
        res = res + 'corr_'
    
    if df.iloc[ind][cols['eq_ineq']]=='=':
        res = res + 'eq'
    elif df.iloc[ind][cols['eq_ineq']]=='<':
        res = res + 'leq'
    elif df.iloc[ind][cols['eq_ineq']]=='>':
        res = res + 'geq'

    return res

def append_mu(A, b, data, df, ind, sign):
    a = np.zeros((1,data.shape[1]))
    a[0][data.columns.get_loc(df.iloc[ind][cols['asset1']])] = sign
    return (np.vstack([A, a]), np.vstack([b, df.iloc[ind][cols['parameter']]]))

def append_rel_mu(A, b, data, df, ind, sign):
    a = np.zeros((1,data.shape[1]))
    a[0][data.columns.get_loc(df.iloc[ind][cols['asset1']])] =  sign
    a[0][data.columns.get_loc(df.iloc[ind][cols['asset3']])] = -sign
    return (np.vstack([A, a]), np.vstack([b, df.iloc[ind][cols['parameter']]]))

def constraints_mu(data = pd.read_excel("data.xlsx"), df = pd.read_excel("views.xlsx")):
    A = np.zeros((0,data.shape[1]))
    b = np.zeros((0,1))
    C = np.zeros((0,data.shape[1]))
    d = np.zeros((0,1))

    for ind in range(len(df)):
        rf = row_format(ind,df)
        if rf == 'mean_eq':
            (A,b) = append_mu(A, b, data, df, ind, +1)
        elif rf == "mean_geq":
            (C,d) = append_mu(C, d, data, df, ind, -1)
        elif rf == "mean_leq":
            (C,d) = append_mu(C, d, data, df, ind, +1)
        elif rf == "rel_mean_eq":
            (A,b) = append_rel_mu(A, b, data, df, ind, +1)
        elif rf == "rel_mean_geq":
            (C,d) = append_rel_mu(C, d, data, df, ind, -1)
        elif rf == "rel_mean_leq":
            (C,d) = append_rel_mu(C, d, data, df, ind, +1)
    return (A,b,C,d)

def append_sigma(A, b, data, df, ind, sign):
    a = np.zeros((1,data.shape[1]))
    a[0][data.columns.get_loc(df.iloc[ind][cols['asset1']])] = sign
    return (np.vstack([A, a]), np.vstack([b, df.iloc[ind][cols['parameter']]**2 / 12])) # really monthly variance

def append_rel_sigma(A, b, data, df, ind, sign):
    a = np.zeros((1,data.shape[1]))
    a[0][data.columns.get_loc(df.iloc[ind][cols['asset1']])] =  sign
    a[0][data.columns.get_loc(df.iloc[ind][cols['asset3']])] = -sign
    return (np.vstack([A, a]), np.vstack([b, df.iloc[ind][cols['parameter']]**2 / 12])) # really monthly variance

def constraints_vol(data = pd.read_excel("data.xlsx"), df = pd.read_excel("views.xlsx")):
    A = np.zeros((0,data.shape[1]))
    b = np.zeros((0,1))
    C = np.zeros((0,data.shape[1]))
    d = np.zeros((0,1))

    for ind in range(len(df)):
        rf = row_format(ind,df)
        if rf == 'vol_eq':
            (A,b) = append_sigma(A, b, data, df, ind, +1)
        elif rf == "vol_geq":
            (C,d) = append_sigma(C, d, data, df, ind, -1)
        elif rf == "vol_leq":
            (C,d) = append_sigma(C, d, data, df, ind, +1)
        elif rf == "rel_vol_eq":
            (A,b) = append_rel_sigma(A, b, data, df, ind, +1)
        elif rf == "rel_vol_geq":
            (C,d) = append_rel_sigma(C, d, data, df, ind, -1)
        elif rf == "rel_vol_leq":
            (C,d) = append_rel_sigma(C, d, data, df, ind, +1)
    return (A,b,C,d)