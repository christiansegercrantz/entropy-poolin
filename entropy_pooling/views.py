from sys import builtin_module_names
import pandas as pd
import numpy as np
import math
from cvxopt import matrix, solvers # pip install cvxopt
from IPython.utils import io
#from mosek import iparam

# TODO:
# Talk about the views.xlsx
# Implement weights for the views
# Test for bugs!
# Test for unexpected inputs. Anything can happen!
# We could do try/except for each row. The software will then alert and ignore weird rows.
# Infeasible constraints (e.g. Example Equity = 0.5 % and Example Equity = 0.6 % is an infeasible combination)
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
# It returns a tuple (A,b,C,d), which contain the constraints Ax = b and Cx <= d.

def load(data = pd.read_excel("Data/data.xlsx"), views_subsheet_name = 0, views_sheet_name = "Data/views.xlsx"): # load_debug, but output is supressed
    with io.capture_output() as captured:
        return load_debug(data, views_subsheet_name, views_sheet_name)

def load_debug(data = pd.read_excel("Data/data.xlsx"), views_subsheet_name = 0, views_sheet_name = "Data/views.xlsx"):

    df = pd.read_excel(views_sheet_name, sheet_name=views_subsheet_name)
    df = set_col_names(df) # adds columns with new names
    data = data/100 # data is in precentages
    # Initialize output matrices and vectors
    A = np.ones((1,len(data))) # ones because sum(x_i) = 1
    b = np.ones((1,1))         #  one because sum(x_i) = 1
    C = np.zeros((0,len(data)))
    d = np.zeros((0,1))

    # MEAN CONSTRAINTS

    # Add the constraints to output matrices and vectors...
    for ind in range(len(df)):
        rf = row_format(ind,df)
        if 'mean' in rf:
            (A,b,C,d) = append_mean(A, b, C, d, data, df, ind, rf)
    
    # Gather mean-constraints and solve a feasible (and near optimal) set of mean values
    # Fixed mean values are required for linear volatility constraints
    posterior_mean = solve_feasible_posterior(data, df, 'mean')

    # VOLATILITY CONSTRAINTS

    # Add the constraints to output matrices and vectors...
    for ind in range(len(df)):
        rf = row_format(ind,df)
        if 'var' in rf:
            (A,b,C,d) = append_var(A, b, C, d, data, posterior_mean, df, ind, rf)

    # Again, for linear correlation constraints, we need to fix the volatilities.
    # As with the mean values, we solve a feasible (near-)optimal solution
    posterior_var = solve_feasible_posterior(data, df, 'var')
    # COVARIANCE CONSTRAINTS

    # Add the constraints to output matrices and vectors...
    for ind in range(len(df)):
        rf = row_format(ind,df)
        if 'corr' in rf:
            (A,b,C,d) = append_corr(A, b, C, d, data, posterior_mean, posterior_var, df, ind, rf)

    data = data*100 # dunno, if this does anything
    return (A,b,C,d)

def set_col_names(data):
    df = data.copy(deep=True)
    cols = {'type'   : '* View on',
        'asset1'     : '* Risk factor 1',
        'asset2'     : 'Risk factor 2 \n(applicable for corr)',
        'asset3'     : 'Risk factor 3',
        'asset4'     : 'Risk factor 4 \n(applicable for corr)',
        'eq_ineq'    : '* Operator',
        'parameter'  : '* Constant \n(alpha)',
        'multiplier' : 'Multiplier \n(beta)'
    }
    for key, value in cols.items():
        df = df.rename(columns={value:key})
    return df

def cov_vector(data, posterior_mean, name1, name2):
    a = data.iloc[:][name1] - posterior_mean[name1].values # data.iloc[:][name1].mean() 
    b = data.iloc[:][name2] - posterior_mean[name2].values
    return a.mul(b)

def returns_to_monthly(r):
    #print(r)
    #print(type(r))
    #return (np.abs(r+1))**(1/12)-1
    return r / 12

def append_mean(A, b, C, d, data, df, ind, rf):
    sign = 1 - 2*('geq' in rf) # Either +1 or -1
    row = data.iloc[:][df.iloc[ind]['asset1']]
    element = sign*returns_to_monthly(df.iloc[ind]['parameter'])
    if 'rel' in rf:
        if (isinstance(df.iloc[ind]['multiplier'], float) and math.isnan(df.iloc[ind]['multiplier'])) or df.iloc[ind]['multiplier']=="-":
            multiplier = 1.0
        else:
            multiplier = df.iloc[ind]['multiplier']
        row = row - data.iloc[:][df.iloc[ind]['asset3']] * multiplier
        #element = element / 12
    
    # Append new row and element with vstack
    if ('leq' in rf) or ('geq' in rf):
        return (A,b,np.vstack([C, sign*row]), np.vstack([d, element]))
    else:
        return (np.vstack([A, sign*row]), np.vstack([b, element]),C,d)

def append_corr(A, b, C, d, data, posterior_mean, posterior_var, df, ind, rf):
    sign = 1 - 2*('geq' in rf) # Either +1 or -1
    var1 = posterior_var[df.iloc[ind]['asset1']].values
    var2 = posterior_var[df.iloc[ind]['asset2']].values
    #var1 = data.iloc[:][df.iloc[ind]['asset1']].var()
    #var2 = data.iloc[:][df.iloc[ind]['asset2']].var()
    row = cov_vector(data, posterior_mean, df.iloc[ind]['asset1'], df.iloc[ind]['asset2']) / np.sqrt(var1 * var2)
    if 'rel' in rf:
        if (isinstance(df.iloc[ind]['multiplier'], float) and math.isnan(df.iloc[ind]['multiplier'])) or df.iloc[ind]['multiplier']=="-":
            multiplier = 1.0
        else:
            multiplier = df.iloc[ind]['multiplier']
        var3 = posterior_var[df.iloc[ind]['asset3']].values
        var4 = posterior_var[df.iloc[ind]['asset4']].values
        row = row - multiplier * cov_vector(data, posterior_mean, df.iloc[ind]['asset3'], df.iloc[ind]['asset4']) / np.sqrt(var3 * var4)
    if ('leq' in rf) or ('geq' in rf):
        C_new = np.vstack([C, sign*row])
        d_new = np.vstack([d, sign*df.iloc[ind]['parameter']])
        return (A, b, C_new, d_new)
    else:
        A_new = np.vstack([A, sign*row])
        b_new = np.vstack([b, sign*df.iloc[ind]['parameter']])
        return (A_new, b_new, C, d)

def append_var(A, b, C, d, data, posterior_mean, df, ind, rf):
    sign = 1 - 2*('geq' in rf) # Either +1 or -1
    row = cov_vector(data, posterior_mean, df.iloc[ind]['asset1'], df.iloc[ind]['asset1'])
    #print('Hellurei')
    if 'rel' in rf:
        prior_variances = data.var()
        prior_vol_1 = np.sqrt(prior_variances[df.iloc[ind]['asset1']])
        prior_vol_3 = np.sqrt(prior_variances[df.iloc[ind]['asset3']])
        #print('Hellurei')
        #print(prior_vol_1)
        row = row * (1 - prior_vol_3**2 / (prior_vol_1 * prior_vol_3))
        if (isinstance(df.iloc[ind]['multiplier'], float) and math.isnan(df.iloc[ind]['multiplier'])) or df.iloc[ind]['multiplier']=="-":
            multiplier = 1.0
        else:
            multiplier = df.iloc[ind]['multiplier']
        row = row - cov_vector(data, posterior_mean, df.iloc[ind]['asset3'], df.iloc[ind]['asset3']) * (multiplier**2 - multiplier * prior_vol_1**2 / (prior_vol_1 * prior_vol_3) )
        #row = row - cov_vector(data, posterior_mean, df.iloc[ind]['asset3'], df.iloc[ind]['asset3'])
    if ('leq' in rf) or ('geq' in rf):
        C_new = np.vstack([C, sign*row])
        d_new = np.vstack([d, sign*df.iloc[ind]['parameter']**2 / 12]) # 1/12 to annualize volatility
        return (A, b, C_new, d_new)
    else:
        A_new = np.vstack([A, sign*row])
        b_new = np.vstack([b, sign*df.iloc[ind]['parameter']**2 / 12]) # 1/12 to annualize volatility
        return (A_new, b_new, C, d)

# Function that reads the format of the row (e.g. non-relative mean equality)
def row_format(ind, df = pd.read_excel("Data/views.xlsx")):
    # If Excel cell 'asset3' is left empty, we are dealing with absolute view
    # Otherwise relative view
    if (isinstance(df.iloc[ind]['asset3'], float) and math.isnan(df.iloc[ind]['asset3'])) or df.iloc[ind]['asset3']=="-":
        res = ""
    else:
        res = "rel_"
    
    if df.iloc[ind]['type']=='Mean':
        res = res + 'mean_'
    elif df.iloc[ind]['type']=='Vol':
        res = res + 'var_'
    elif df.iloc[ind]['type']=='Corr':
        res = res + 'corr_'
    
    if df.iloc[ind]['eq_ineq']=='=':
        res = res + 'eq'
    elif df.iloc[ind]['eq_ineq']=='<':
        res = res + 'leq'
    elif df.iloc[ind]['eq_ineq']=='>':
        res = res + 'geq'
    return res

def solve_feasible_posterior(data, df, type):
    # We use cvxopt:
    # http://cvxopt.org/userguide/coneprog.html?highlight=lp#quadratic-cone-programs
    if type == 'mean':
        (A_mean, b_mean, G_mean, h_mean) = post_constraints(data, df, 'mean')
        q_mean = np.matrix(-data.mean().to_numpy()).transpose()
    elif type == 'var':
        (A_mean, b_mean, G_mean, h_mean) = post_constraints(data, df, 'var')
        q_mean = np.matrix(-data.var().to_numpy()).transpose()
    q_mean = matrix(np.nan_to_num(q_mean)) # nan values to 0
    P_mean = matrix(np.eye(data.shape[1]))
    G_mean = matrix(G_mean); h_mean = matrix(h_mean); A_mean = matrix(A_mean); b_mean = matrix(b_mean)
    dims = {'l': G_mean.size[0], 'q': [], 's': []} # component-wise inequality constraint (default for cvxopt)
    # Let's solve   min_mean ||mu_0 - mu||**2
    #               s.t.     A*mu == b
    #                        G_mean*mu <= h_mean (element wise) (note leq instead of geq!)
    mu = solvers.coneqp(P_mean, q_mean, G_mean, h_mean, dims, A_mean, b_mean)
    posterior_mean = pd.DataFrame(mu['x'])
    posterior_mean = pd.DataFrame(data=posterior_mean.T.values, columns=data.columns)
    return posterior_mean

# CONSTRAINTS FOR CONIC OPTIMIZATION

def post_constraints(data, df, type):
    A = np.zeros((0,data.shape[1]))
    b = np.zeros((0,1))
    C = np.zeros((0,data.shape[1]))
    d = np.zeros((0,1))

    for ind in range(len(df)):
        rf = row_format(ind,df)
        if type in rf:
            (A,b,C,d) = post_const_append(A, b, C, d, data, df, ind, rf, type)
    return (A,b,C,d)

def post_const_append(A, b, C, d, data, df, ind, rf, type):
    sign = 1 - 2*('geq' in rf)
    a = np.zeros((1,data.shape[1]))
    a[0][data.columns.get_loc(df.iloc[ind]['asset1'])] =  sign
    if 'rel' in rf:
        a[0][data.columns.get_loc(df.iloc[ind]['asset3'])] = -sign
    if type == 'var':
        b_new = sign*df.iloc[ind]['parameter']**2 / 12 # really monthly variance
    elif type == 'mean':
        b_new = sign*returns_to_monthly(df.iloc[ind]['parameter']) # monthly mean
    if 'geq' in rf or 'leq' in rf:
        return (A, b, np.vstack([C, a]), np.vstack([d, b_new]))
    else:
        return (np.vstack([A, a]), np.vstack([b, b_new]), C, d)