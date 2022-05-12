import numpy as np
import pandas as pd
from scipy.optimize import minimize, LinearConstraint, Bounds
import matplotlib.pyplot as plt
from plotnine import ggplot, geom_point, aes, geom_line, labs, geom_text, position_jitter, theme, element_text, theme_linedraw, element_line, element_rect

def load_asset_deltas(filename, sheet_name = 0):
    """Uploads the data containing the asset delta matrix from the given Excel file.
    --------------------
    ### Input arguments:
        filename: String
            The name of the excel file that contains the (N x F) matrix of the factor sensitivites
            of the optimizable assets. (N = number of assets, F = number of factors)
            The data should contain a header (with factor names) and the first row
            contains the indexes (asset names)
        sheet_name (optional): String
            If the delta matrix is given inside a bigger Excel workbook, then extract the right sheet
    --------------------
    ### Returns:
        deltas: The (N x F) asset sensitivity ('delta') matrix (in numpy format, without row or column names)
        asset_names: List of the names of the assets included in the deltas matrix as indexers
    """

    deltas = pd.read_excel(filename, sheet_name, header = 0, index_col = 0)

    # If there are any missing values, convert them to zeros
    deltas = deltas.fillna(0)
    asset_names = list(deltas.index)
    # Convert delta matrix to a simple numpy array
    deltas = deltas.to_numpy()

    return deltas, asset_names

def asset_scenarios(factor_scenarios, asset_deltas, asset_names):
    """Computes the scenario-wise returns for each portfolio asset using the factor scenarios matrix and asset delta matrix.
    --------------------
    ### Input arguments:
        factor_scenarios: Matrix
        A (S x F) matrix of factor return scenarios
        asset_deltas: Matrix
        A (N x F) matrix of asset sensitivities ('deltas') to changes in factors
        asset_names: List
        A (N x 1) list of the asset names
    --------------------
    ### Returns:
        asset_scenarios:
        A (S x N) matrix containing the asset return scenarios
    """

    # Check that the F dimension matches
    assert factor_scenarios.shape[1] == asset_deltas.shape[1], "The number of factors (x dimension) is not the same for the input matrices."
    asset_scenarios = factor_scenarios @ asset_deltas.T
    asset_scenarios.columns = asset_names
    return asset_scenarios

def optimizer(scenarios, probabilities, mu_0, additional_constraints = None, allow_shorting = False, visualize = False, verbose = 0):
    """Optimizes the weights put on each item the portfolio. This is done by minimizing the volatility of the portfolio at a given return procentage. Also visualized the markoviz model if requested.
    --------------------
    ### Input arguments:
        scenarios: Matrix
            A (S x N) matrix of the optimizable portfolio items
        probabilities: Array of floats <= 1
            A (S x 1) vector of prior probabilities
        mu_0: Float
            The return to optimize for, given in decimal as 50% = 0.5
        additional_constraints: Tupple(Matrix,Array[Float],Array[Float]) | None, Default: None
            A tuple to define additional constraints.
            The first element ought to be a (#Additional_constriants x #Assets matrix) defining the additional constraint. The values are to be floats in the range [0,100]
            The second element ought to be a (#Additional_constriants x 1) vector defining the lower bounds of the constraints. The values are to be floats in the range [0,100]
            The third element ought to be a (#Additional_constriants x 1) vector defining the upper bounds of the constraints. The values are to be floats in the range [0,100]
        allow_shorting: Boolean, Default False
            Weather to allow shorting, i.e. to not constraint the variable to [0,1] but to ]-inf, inf[
        visualize: Boolean, Default False
            Plots the efficient prontier, the optimal protoflio, the original portfolio items and a cloud of randomly weighted items.
        verbose: {0,1,2} , Default 0
            Weather to display extra information about the optimization. 0 will display nothing, 1 will display the result of the optimiztion and and error message in case it was not succesful, 2 will additionally to 1 display convergence messages of the optimizer. 
    --------------------
    ### Returns:
        res:
            A OptimizeResult object from scipy, where x is the optimal weight of each item.
    """

    #In case the probabilities vector  is of shape (n,) instead of (n,1)
    if probabilities.ndim == 1:
        probabilities = probabilities.reshape(-1,1)

    #Get the mean vector and covariance matrix
    mu, covar = mean_and_var(scenarios, probabilities)

    m, n = covar.shape
    x0 = np.ones(m)/m

    def jac(x):
        return 2 * covar @ x

    def objective_function(x):
        return  x.T @ covar @ x

    if allow_shorting:
        bounds = Bounds(lb = -np.ones(m)*np.inf, ub = np.ones(m) * np.inf) #[(0, 1) for i in range(m)]
    else: 
        bounds = Bounds(lb = np.zeros(m), ub = np.ones(m) * np.inf)
    constraints = (LinearConstraint(np.ones(m), lb=1, ub=1), #Sum of weights 1
                   LinearConstraint(mu, lb=mu_0, ub=np.inf) #Greater or equal to a certain return level
                  )
    if additional_constraints is not None:
        new_constraint = (LinearConstraint(additional_constraints[0],
                                            lb = additional_constraints[1],
                                            ub = additional_constraints[2]
                                         ),)
        constraints = constraints + new_constraint
                          

    disp = True if verbose == 2 else False 
    
    res = minimize(objective_function,
                   method = 'SLSQP',
                   jac=jac,
                   x0 = x0,
                   bounds = bounds,
                   constraints = constraints,
                   tol=0.00001,
                   options = {"disp": disp})
    if verbose:
        print(f"The optimization was succesful: {res.success}")
        if not res.success:
            print(f"The optimization was terminated due to: \n{res.message}")
    
    if visualize:
        vizualization(covar, mu, optimal = res.x, frontier=True, scenarios = scenarios, probabilities = probabilities)
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

def vizualization(covar,
                  mu,
                  generated_points = 50000,
                  frontier = True,
                  optimal = None,
                  scenarios = None,
                  probabilities = None,
                  mu_0 = None):
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
    port_returns = np.array([])
    port_vol = np.array([])
    for i in range(0, generated_points):
        y = np.random.rand(m,1)**10
        y = y/np.sum(y)
        port_returns = np.append(port_returns, mu @ y)
        port_vol = np.append(port_vol, y.T @ covar @ y)

    if frontier:
        assert(scenarios is not None), "You have to give in scenarios in order to plot the frontier"
        assert(probabilities is not None), "You have to give in weights in order to to plot the frontier"
        frontier_mu = np.array([])
        frontier_var = np.array([])
        for j in np.linspace(0, np.max(mu), 100):
            opt = optimizer(scenarios, probabilities, mu_0 = j, verbose = 0)
            frontier_mu = np.append(frontier_mu, mu @ opt.x)
            frontier_var = np.append( frontier_var, opt.x.T @ covar @ opt.x)

    # fig, ax = plt.subplots()
    # ax.scatter(port_vol, port_returns)
    # ax.scatter(np.diag(covar), mu, color = "yellow");
    # for i, txt in enumerate(scenarios.columns):
    #     ax.annotate(txt, (np.diag(covar)[i], mu[i]))
    # ax.scatter(optimal.T @ covar @ optimal,mu @ optimal, color='red');
    # ax.annotate("Optimal", (optimal.T @ covar @ optimal,mu @ optimal));
    # if frontier:
    #     ax.plot(frontier_var, frontier_mu, color='red');
    # plt.show();

    

    generated_df = pd.DataFrame(list(zip(port_vol,
                                    port_returns)),
                        columns = ["Volatility",
                                   "Returns"])
    frontier_df = pd.DataFrame(list(zip(frontier_var,
                                    frontier_mu)),
                        columns = ["Volatility",
                                   "Returns"])
    #df = pd.DataFrame(list(zip(frontier_var,frontier_mu)),
    #                    columns = ["Volatility", "Returns"])
    (ggplot()
        + theme(legend_title = element_text(
                family = "Calibri",
                colour = "brown",
                face = "bold",
                size = 12))
        + theme(panel_grid_major = element_line(size = 0.5,
                                                linetype = 'solid',
                                                colour = "black"),
                panel_grid_minor = element_line(size = 0.25,
                                                linetype = 'solid',
                                                colour = "black"),
                panel_background = element_rect(fill = "white"))
        + labs(title="Optimal solution", y="Returns", x="Volatility")
        + geom_point(data = generated_df,
                     mapping = aes(x = "Volatility",
                        y = "Returns"),
                     color = "#7D8CC4"
                    )
        + geom_line(data = frontier_df,
                    mapping= aes(x = "Volatility",
                        y = "Returns"),
                    color = "#A61C3C"
                    )
        + geom_point(mapping = aes(x = optimal.T @ covar @ optimal, y=mu @ optimal),
                     color = "#242331",
                     #shape=13
                     )
        + geom_text(mapping = aes(x = optimal.T @ covar @ optimal, y=mu @ optimal),
                     label = "Optimal",
                     nudge_y = -0.02,
                     nudge_x = 3,
                     size = 7,
                     )
        + geom_point(aes(x = np.diag(covar), y = mu),
                     color = "#A27035"
                     )
        + geom_text(aes(x = np.diag(covar), y = mu),
                     label = scenarios.columns,
                     nudge_y = -0.02,
                     nudge_x = 3,
                     size = 7,
                     position=position_jitter()
                     )

    ).draw()
