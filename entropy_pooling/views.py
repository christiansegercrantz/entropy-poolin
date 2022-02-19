import pandas as pd
import numpy as np
import math

# Column names. LHS has code names used in this script. RHS are the ones used in the Excel-file
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

    # Thus far only mean-views are implemented
    
    df = pd.read_excel("views.xlsx")

    # Initialize output matrices and vectors
    A = np.zeros((0,len(data)))
    b = np.zeros((0,1))
    C = np.zeros((0,len(data)))
    d = np.zeros((0,1))

    for ind in range(len(df)):
        rf = row_format(ind,df)
        if rf == 'mean_eq':
            # Append values to matrix A and vector b
            A = np.vstack([A, data.iloc[:][df.iloc[ind][cols['asset1']]]])
            b = np.vstack([b, df.iloc[ind][cols['parameter']]])
        elif rf == "mean_geq":
            C = np.vstack([C, data.iloc[:][df.iloc[ind][cols['asset1']]]])
            d = np.vstack([d, df.iloc[ind][cols['parameter']]])
        elif rf == "mean_leq":
            C = np.vstack([C, -data.iloc[:][df.iloc[ind][cols['asset1']]]])
            d = np.vstack([d, -df.iloc[ind][cols['parameter']]])
        elif rf == "rel_mean_eq":
            mat_element = data.iloc[:][df.iloc[ind][cols['asset1']]] - (1 + df.iloc[ind][cols['parameter']])*data.iloc[:][df.iloc[ind][cols['asset3']]]
            A = np.vstack([A, mat_element])
            b = np.vstack([b, 0.0])
        elif rf == "rel_mean_geq":
            mat_element = data.iloc[:][df.iloc[ind][cols['asset1']]] - (1 + df.iloc[ind][cols['parameter']])*data.iloc[:][df.iloc[ind][cols['asset3']]]
            C = np.vstack([C, mat_element])
            d = np.vstack([d, 0.0])
        elif rf == "rel_mean_leq":
            mat_element = data.iloc[:][df.iloc[ind][cols['asset1']]] - (1 + df.iloc[ind][cols['parameter']])*data.iloc[:][df.iloc[ind][cols['asset3']]]
            C = np.vstack([C, -mat_element])
            d = np.vstack([d, 0.0])
    
    # Todo:
    # Gather mean-constraints and solve a feasible one for covariances
    # Implement covariance constraints

    return (A,b,C,d)


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
    return ""