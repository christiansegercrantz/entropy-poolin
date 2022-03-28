import numpy as np
import pandas as pd

def optimizer(scenarios, weights, mu_0, disp = True):
    from scipy.optimize import minimize, LinearConstraint, Bounds
    mu, covar = mean_and_var(scenarios, weights)
  
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
    return res


def mean_and_var(scenarios, weights):
    m,n = scenarios.shape
    weights_reshaped = np.asarray(weights).reshape(m,)
    mu = np.average(scenarios, axis=0, weights = weights_reshaped)
    covar = np.cov(scenarios, rowvar = False, aweights = weights_reshaped) #* 252 #annualization constant
    return mu, covar

def vizualization(covar, mu, generated_points = 50000, frontier = True, optimal = None, scenarios = None, weights = None, mu_0 = None):
    if optimal is None:
        assert(scenarios is None), "You have to give in scenarios in order to find the optimal using the method"
        assert(weights is None), "You have to give in weights in order to find the optimal using the method"
        assert(mu_0 is None), "You have to give in the return lower bound mu_0 to find the optimal using the method"
        optimal = optimizer(scenarios, weights, mu_0 = mu_0, disp = False)
    m,n = covar.shape
    port_returns = []
    port_vol = []
    for i in range(0, generated_points):
        y = np.random.rand(m,1)**10
        y = y/np.sum(y)
        port_returns.append(mu @ y)
        port_vol.append(y.T @ covar @ y)

    frontier_mu = np.array([])
    frontier_var = np.array([])
    if frontier:
        for j in np.linspace(0, np.max(mu), generated_points/500):
            opt = optimizer(scenarios, weights, mu_0 = j, disp = False)
            frontier_mu = np.append(frontier_mu, mu @ opt.x)
            frontier_var = np.append( frontier_var, opt.x.T @ covar @ opt.x)
            
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ax.scatter(port_vol, port_returns)
    ax.scatter(np.diag(covar), mu, color = "yellow");
    for i, txt in enumerate(scenarios.columns):
        ax.annotate(txt, (np.diag(covar)[i], mu[i]))
    ax.scatter(optimal.T @ covar @ optimal,mu @ optimal, color='red');
    ax.annotate("Optimal", (optimal.T @ covar @ optimal,mu @ optimal));
    ax.plot(frontier_var, frontier_mu, color='red');