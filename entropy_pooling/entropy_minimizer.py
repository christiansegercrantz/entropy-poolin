# imports
import numpy as np
import scipy.optimize as opt

# TODO: define the type of input arguments: lists/arrays or numpy arrays or DataFrames???
# current code is made for numpy arrays, let's see if the syntax works for pd DFs

def full_confidence_posterior(p, A, a_lb, a_ub):
    # Computes the full-condifence posterior distribution by finding the constrained
    # entropy-minimizing set of variables
    #
    # Input arguments:
    # p: a (J x 1) vector of prior probabilities
    # A: the (K x J) matrix used to express the constraints
    # a_lb: lower bound vector (1 x K) for constraints
    # a_ub: upper bound vector (1 x K) for constraints

    # Separate equality and inequality constraints to different arrays
    is_equal = a_lb == u_lb
    H = A[is_equal]
    h = a_ub[is_equal]
    dim_h = len(h)

    # We want F to include only Fx <= f type of constraints. This is why we multiply all >= ones and
    # their coefficients by -1.
    F = A[not is_equal]
    F = np.concatenate((F, -1*F))
    f = np.concatenate((a_ub[is_equal], -1*a_lb[is_equal]))
    dim_f = len(f)

    # Nested function for computing the dual function values (only one input x to be scipy optimizer -compatible)
    def dual(x):
        l, v = x[:dim_h], x[dim_h:] # separate equality and inequality dual variables
        x = p * np.exp(- 1 - np.dot(F, l) - np.dot(H, v)) # primal solution (with l, v fixed)
        L = np.dot(x, np.log(x) - np.log(p)) + np.dot(l, np.dot(F, x) - f) + np.dot(v, np.dot(H, x) - h) # Dual value
        return L

    # Nested function for computing the Jacobian of the dual function (needed by scipy optimizer)
    def Jac(x):
        l, v = x[:dim_h], x[dim_h:] # separate equality and inequality dual variables
        # TODO: compute partial derivatives
        # TODO: stack all to one vector and return

    res = opt.minimize(fun = lambda x : -1 * dual(x), x0 = p, method = 'Newton-CG', jac = ???) # TODO: Define the Jecobian
    l_opt, v_opt = res.x[:dim_h], res.x[dim_h:]

    # Step 3: compute optimal primal variable values = posterior distribution
    posterior = p * np.exp(- 1 - np.dot(F, l_opt) - np.dot(H, v_opt))
    return posterior

def confidence_weighted_posterior(p_prior, p_post, c):
    # Computes the confidence-weighted posterior distribution,
    # in fact, computing a c-weighted average of the prior and posterior distributions
    #
    # Input arguments:
    # p_prior: the (J x 1) vector of prior probabilities
    # p_post: the (J x 1) vector of posterior probabilities, output from full_confidence_posterior
    # c: a scalar or (L x 1) vector giving the confidence weight(s).

    # Error handling: check that all components of c are within [0, 1]?

    if type(c) in [int, float]:
        p_c = (1 - c)*p_prior + c*p_post # TODO: hande cases of scalar and vector c.
    else:
        # TODO: implement
    
    return p_c
