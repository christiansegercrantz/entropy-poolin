import numpy as np
import pandas as pd
from scipy.optimize import minimize, LinearConstraint, Bounds
from plotnine import ggplot, geom_point, aes, geom_line, labs, geom_text, position_jitter, theme, element_text, theme_linedraw, element_line, element_rect, scale_y_continuous, scale_x_continuous

def load_factor_scenarios(filename, sheet_name = 0, scale_by_100 = False):
    """Uploads the factor scenario data and the prior scenario distribution.
    --------------------
    ### Input arguments:
        filenme: String
            The name of the Excel file that contains the factor scenario matrix and prior distribution (column 'Weight').
        sheet_name (optional): String
            User can specify the sheet name of the data in the Excel file.
        scale_by_100 (optional): Boolean
            Tells whether to divide the factor return data by 100 (1% -> 0.01) or not. Default is False
    --------------------
    ### Returns:
        scenarios: pd.DataFrame
            The (S x F) sized factor scenario data (S = number of scenarios, F = number of factors).
        prior: np.Array
            The vector of length S containing the scenario prior probabilities.
    """

    data = pd.read_excel(filename, sheet_name, header = 0).dropna(axis = 1, how = 'all')
    scenarios = data.drop(columns = ['Weight'])
    if scale_by_100:
        scenarios /= 100
    prior = data['Weight'].to_numpy()

    return scenarios, prior

def load_asset_deltas(filename, sheet_name = 0):
    """Uploads the data containing the asset delta matrix from the given Excel file.
    --------------------
    ### Input arguments:
        filename: String
            The name of the Excel file that contains the (F x N) matrix of the factor sensitivites
            of the optimizable assets. (N = number of assets, F = number of factors)
            The data should contain a header (with factor names) and the first row
            contains the indexes (asset names)
        sheet_name (optional): String
            If the delta matrix is given inside a bigger Excel workbook, then extract the right sheet
    --------------------
    ### Returns:
        deltas: numpy.ndarray
            The (F x N) asset sensitivity ('delta') matrix (in numpy format, without row or column names)
        asset_names: list<String>
            List of the names of the assets included in the deltas matrix as indexers
    """

    deltas = pd.read_excel(filename, sheet_name, header = 0, index_col = 0)

    # If there are any missing values, convert them to zeros
    deltas = deltas.fillna(0)
    asset_names = list(deltas.columns)
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
            A (F x N) matrix of asset sensitivities ('deltas') to changes in factors
    --------------------
    ### Returns:
        asset_scenarios:
            A (S x N) matrix containing the asset return scenarios
    """

    # Check that the F dimension matches
    assert factor_scenarios.shape[1] == asset_deltas.shape[0], "The number of factors (x dimension) is not the same for the input matrices."
    asset_scenarios = factor_scenarios @ asset_deltas
    asset_scenarios.columns = asset_names
    return asset_scenarios

def load_portfolio_constraints(filename, sheet_name = 0):
    """Uploads the data containing the portfolio constraints used in Markowitz optimization.
    The constraints must be given in format A|lb|ub, the user can distinguish between the components
    by leaving empty columns between the components in the excel sheet.
    --------------------
    ### Input arguments:
        filename: String
            The name of the (Excel) file that contains the constraints
        sheet_name (optional): String
            The name of the sheet where the constraints are written
    --------------------
    ### Returns:
        A:  numpy.ndarray
            The matrix containing all the left-hand coefficients of the constraint equations
        lb: numpy.ndarray
            The vector containing the right-hand lower bounds of the constraint inequations Ax >= lb
        ub: numpy.ndarray
            The vector containing the right-hand upper bounds of the constraint inequations Ax <= ub
    """

    constrs = pd.read_excel(filename, sheet_name, header = 0).dropna(axis = 1, how = 'all')
    A  = constrs.iloc[:,:-2]
    lb = constrs.iloc[:,-2]
    ub = constrs.iloc[:,-1]

    return A, lb, ub

def optimizer(scenarios, probabilities, mu_0, manual_constraints, visualize = False, verbose = 0):
  
    """Optimizes the weights put on each item the portfolio. This is done by minimizing the volatility of the portfolio at a given return procentage. Also visualized the markoviz model if requested.
    --------------------
    ### Input arguments:
        scenarios: Matrix
            A (S x N) matrix of the optimizable portfolio items
        probabilities: Array of floats <= 1
            A (S x 1) vector of prior probabilities
        mu_0: Float
            The return to optimize for given in euros
        total: Float | Default = 1.0
            Total usable funds for the optimization
        manual_constraints: Tupple(Matrix,Array[Float],Array[Float])
            A tuple to define the constraints. If not given, we assume only that the sum of all assets is 1 and the assets are bound to [0, inf( (subject to allow_shorting)
            The first element ought to be a (#Additional_constriants x #Assets matrix) defining the additional constraint. The values are to be floats in the range [0,100]
            The second element ought to be a (#Additional_constriants x 1) vector defining the lower bounds of the constraints. The values are to be floats in the range [0,100]
            The third element ought to be a (#Additional_constriants x 1) vector defining the upper bounds of the constraints. The values are to be floats in the range [0,100]
        allow_shorting: Boolean, Default False
            Weather to allow shorting, i.e. to not constraint the variable to [0,1] but to ]-inf, inf[. Only active if manual_constraints is None
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

    m, _ = covar.shape
    x0 = np.ones(m)/m

    def jac(x):
        return 2 * covar @ x

    def objective_function(x):
        return  np.sqrt(x.T @ covar @ x)

    # We split the constraints into equality and inequality constraints for efficient optimization.
    # Unclear if this is a speed up or not for the portfolio sizes in question
    equality_constraint_matrix = manual_constraints[0].copy(deep=True)
    inequality_constraint_matrix = manual_constraints[0].copy(deep=True)
    equality_constraint_lb = manual_constraints[1].copy(deep=True)
    inequality_constraint_lb = manual_constraints[1].copy(deep=True)
    equality_constraint_ub = manual_constraints[2].copy(deep=True)
    inequality_constraint_ub = manual_constraints[2].copy(deep=True)
    for i, _ in enumerate(manual_constraints[1]):
        if (manual_constraints[1][i] != manual_constraints[2][i]):
            equality_constraint_matrix.drop(index = i, inplace=True)
            equality_constraint_lb.pop(i)
            equality_constraint_ub.pop(i)
        else:
            inequality_constraint_matrix.drop(index = i, inplace=True)
            inequality_constraint_lb.pop(i)
            inequality_constraint_ub.pop(i)

    constraints = (LinearConstraint(equality_constraint_matrix,  #A in Ax=b
                                    lb = equality_constraint_lb, #Lower bound
                                    ub = equality_constraint_ub  #Upper Bound
                                    ),
                    LinearConstraint(inequality_constraint_matrix, #A in lb=<Ax=<ub
                                    lb = inequality_constraint_lb, #Lower bound
                                    ub = inequality_constraint_ub  #Upper Bound
                                    ),
                    LinearConstraint(mu, lb=mu_0, ub=np.inf)
                    )



    disp = True if verbose == 2 else False

    res = minimize(objective_function,
                   #method = 'SLSQP', #Change to L-BFGS-B,
                   jac=jac,
                   x0 = x0,
                   constraints = constraints,
                   tol=0.01,
                   options = {"disp": disp, 'maxiter': 10**4})
    if verbose:
        print(f"The optimization was succesful: {res.success}")
        if not res.success:
            print(f"The optimization was terminated due to: \n{res.message}")

    if visualize:
        vizualization(covar, mu, optimal = res.x, manual_constraints = manual_constraints, frontier=True, scenarios = scenarios, probabilities = probabilities)
    return res


def mean_and_var(scenarios, probabilities):
    """Calculates the mean and variance of portfolio using the given probabilities
    --------------------
    ### Input arguments:
        scenarios: Matrix
            A (S x N) matrix of the optimizable portfolio items
        probabilities: Array of floats <= 1
            A (S x 1) vector of prior probabilities
    --------------------
    ### Returns:
        mu: (N x 1) vector
            The mean of each portfolio item

        covar: (N x N) matrix
            The covariance of the portfolio items"""

    probabilities_reshaped = np.asarray(probabilities).reshape(-1,)
    mu = np.average(scenarios, axis=0, weights = probabilities_reshaped)
    covar = np.cov(scenarios, rowvar = False, aweights = probabilities_reshaped)
    return mu, covar

def vizualization(covar,
                  mu,
                  generated_points = 50000, 
                  frontier = True,
                  optimal = None,
                  scenarios = None,
                  probabilities = None,
                  manual_constraints = None,
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
        optimal = optimizer(scenarios, probabilities, manual_constraints = manual_constraints, mu_0 = mu_0, disp = False, vizualization = False)
    m,n = covar.shape


    if frontier:
        assert(scenarios is not None), "You have to give in scenarios in order to plot the frontier"
        assert(probabilities is not None), "You have to give in weights in order to to plot the frontier"
        assert(manual_constraints is not None), "You have to give menual constraints in order to plot the frontier"
        frontier_mu = np.array([])
        frontier_var = np.array([])
        max_mu = np.max(mu)
        for j in np.linspace(0, max_mu, 100):
            opt = optimizer(scenarios, probabilities, manual_constraints = manual_constraints, mu_0 = j, verbose = 0)
            frontier_mu = np.append(frontier_mu, mu @ opt.x)
            frontier_var = np.append(frontier_var, opt.x.T @ covar @ opt.x)
        
    port_returns = np.array([])
    port_vol = np.array([])
    for _ in range(0, generated_points): #Could probably be vectorized
        y = np.random.rand(m,1)**10
        y = y/np.sum(y)
        port_returns = np.append(port_returns, mu @ y)
        port_vol = np.append(port_vol, y.T @ covar @ y)

    generated_df = pd.DataFrame(list(zip(port_vol,
                                    port_returns)),
                        columns = ["Volatility",
                                   "Returns"])
    if frontier:
        frontier_df = pd.DataFrame(list(zip(frontier_var,
                                            frontier_mu)),
                            columns = ["Volatility",
                                    "Returns"])
    #df = pd.DataFrame(list(zip(frontier_var,frontier_mu)),
    #                    columns = ["Volatility", "Returns"])
    plot = (ggplot()
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
        + scale_y_continuous(labels=lambda l: ["%.1f%%" % (v * 100) for v in l])
        + scale_x_continuous(labels=lambda l: ["%.3f%%" % (v * 100) for v in l])
        + geom_point(data = generated_df,
                     mapping = aes(x = "Volatility",
                                   y = "Returns"),
                     color = "#7D8CC4"
                    )
        + geom_point(mapping = aes(x = optimal.T @ covar @ optimal , y=mu @ optimal),
                     color = "#242331",
                     #shape=13
                     )
        + geom_text(mapping = aes(x = optimal.T @ covar @ optimal, y=mu @ optimal),
                     label = "Optimal",
                     #nudge_y = -0.02,
                     #nudge_x = 0.5,
                     size = 7,
                     )
        + geom_point(aes(x = np.diag(covar), y = mu),
                     color = "#A27035"
                     )
        + geom_text(aes(x = np.diag(covar), y = mu),
                     label = scenarios.columns,
                     #nudge_y = -0.02,
                     #nudge_x = 0.5,
                     size = 7,
                     position=position_jitter()
                     )

    )
    if frontier:
        plot += geom_line(data = frontier_df,
                    mapping= aes(x = "Volatility",
                        y = "Returns"),
                    color = "#A61C3C"
                    )
    plot.draw()
