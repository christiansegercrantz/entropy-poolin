import pandas as pd

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
    df = pd.read_excel("views.xlsx")
    # Asset names to data column numbers
    # All ineqs to >-sided ones
    for ind in len(df):
        if df[ind][cols['eq_ineq']] == '<':
            df[ind][cols['asset1']] = df[ind][cols['asset3']], df[ind][cols['asset3']] = df[ind][cols['asset1']]
            df[ind][cols['asset2']] = df[ind][cols['asset4']], df[ind][cols['asset4']] = df[ind][cols['asset2']]
            df[ind][cols['eq_ineq']] = '>'
    # Gather mean-constraints and solve a feasible one for covariances
    # Form the different matrices
    return df