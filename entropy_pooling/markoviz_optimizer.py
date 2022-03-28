import numpy as np
import pandas as pd
from scipy.optimize import minimize, LinearConstraint, Bounds
import matplotlib.pyplot as plt

def optimizer(scenarios, probabilities, mu_0, disp = True, vizualization = False):
    """Optimizes the weights put on each item the portfolio. This is done by minimizing the volatility of the portfolio at a given return procentage. Also visualized the markoviz model if requested.
    --------------------
    ### Input arguments:
        scenarios: Matrix
            A (S x N) matrix of the optimizable portfolio items
        probabilities: Array of floats <= 1
            A (S x 1) vector of prior probabilities
        mu_0: Float
            The return to optimize for, given in decimal as 50% = 0.5
        disp: Default 'True'
            the (K x S) matrix used to express the constraints Cx <= d
        vizualization: Default 'False'
            Plots the efficient prontier, the optimal protoflio, the original portfolio items and a cloud of randomly weighted items.
    --------------------        
    ### Returns: 
        res: 
            A OptimizeResult object from scipy, where x is the optimal weight of each item.
    """
    
    mu, covar = mean_and_var(scenarios, probabilities)
  
    m, n = covar.shape
    x0 = np.ones(m)/m
    
    def jac(x):
        return 2 * covar @ x
    
    def objective_function(x):
        return  x.T @ covar @ x
    
    bounds = Bounds(lb = np.zeros(m), ub = np.ones(m)) #[(0, 1) for i in range(m)]
    constraints = (LinearConstraint(np.ones(m), lb=1, ub=1), #Sum of weights 1
                  LinearConstraint(mu, lb=mu_0, ub=np.inf) #Greater or equal to a certain return level
                  ) 
    
    res = minimize(objective_function,
                   jac=jac,
                   x0 = x0,
                   bounds = bounds,
                   constraints = constraints,
                   tol=0.00001,
                   options = {"disp": disp})
    if vizualization:
        vizualization(covar, mu, optimal = res.x, frontier=True, scenarios = scenarios, weights = probabilities)
    return res


def mean_and_var(scenarios, probabilities):
    """Calculates the mean and variance of portfolio using the given probabilities
    --------------------
    ### Input arguments:
        scenarios: Matrix
            A (S x ) matrix of the optimizable portfolio items
        probabilities: Array of floats <= 1
            A (S x 1) vector of prior probabilities
    --------------------    
    ### Returns: 
        mu: (N x 1) vector
            The mean of each portfolio item
        
        covar: (N x N) matrix
            The covariance of the portfolio items"""
    
    m,n = scenarios.shape
    probabilities_reshaped = np.asarray(probabilities).reshape(m,)
    mu = np.average(scenarios, axis=0, weights = probabilities_reshaped)
    covar = np.cov(scenarios, rowvar = False, aweights = probabilities_reshaped) #* 252 #annualization constant
    return mu, covar

def vizualization(covar, mu, generated_points = 50000, frontier = True, optimal = None, scenarios = None, probabilities = None, mu_0 = None):
    """ Visualizes the markoviz model, the original portfolio items and the optimal points. Optimally calculates the optimal results and plots the efficient frontier
    --------------------
    ### Input arguments:
        covar:(N x N) matrix
            The covariance of the portfolio items
        mu: (N x 1) vector
            The mean of each portfolio item
        generated_points: Int, Default:50000,
            The amount of points generated in the cloud.
        frontier: Boolean, Default: 'True'
            Weather to plot the efficient frontier or not. If 'True' requires scenarios, weights and mu_0 to be defined.
        optimal: Default: None
            The optimal values for the weights of each portfolio items. If None, calculates the optimal weights, requires scenarios, weights and mu_0 to be defined.
        scenarios: Matrix |None, Default: None
            A (S x N) matrix of the optimizable portfolio items
        probabilities = Array |None, Default: None
            A (S x 1) vector of prior probabilities
        mu_0 = Float |None, Default: None
            The return to optimize for, given in decimal as 50% = 0.5

    """
    if optimal is None:
        assert(scenarios is not None), "You have to give in scenarios in order to find the optimal using the method"
        assert(probabilities is not None), "You have to give in weights in order to find the optimal using the method"
        assert(mu_0 is not None), "You have to give in the return lower bound mu_0 to find the optimal using the method"
        optimal = optimizer(scenarios, probabilities, mu_0 = mu_0, disp = False, vizualization = False)
    m,n = covar.shape
    port_returns = []
    port_vol = []
    for i in range(0, generated_points):
        y = np.random.rand(m,1)**10
        y = y/np.sum(y)
        port_returns.append(mu @ y)
        port_vol.append(y.T @ covar @ y)

    if frontier:
        assert(scenarios is None), "You have to give in scenarios in order to plot the frontier"
        assert(probabilities is None), "You have to give in weights in order to to plot the frontier"
        frontier_mu = np.array([])
        frontier_var = np.array([])
        for j in np.linspace(0, np.max(mu), 100):
            opt = optimizer(scenarios, probabilities, mu_0 = j, disp = False)
            frontier_mu = np.append(frontier_mu, mu @ opt.x)
            frontier_var = np.append( frontier_var, opt.x.T @ covar @ opt.x)
            
    fig, ax = plt.subplots()
    ax.scatter(port_vol, port_returns)
    ax.scatter(np.diag(covar), mu, color = "yellow");
    for i, txt in enumerate(scenarios.columns):
        ax.annotate(txt, (np.diag(covar)[i], mu[i]))
    ax.scatter(optimal.T @ covar @ optimal,mu @ optimal, color='red');
    ax.annotate("Optimal", (optimal.T @ covar @ optimal,mu @ optimal));
    if frontier:
        ax.plot(frontier_var, frontier_mu, color='red');
    plt.show();