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
    # We want F to include only Fx <= f type of constraints. This is why we multiply all >= ones and
    # their coefficients by -1. Then, the multipliers lambda are nonnegative
    is_equal = a_lb == u_lb

    F = A[not is_equal, :]
    F = np.concatenate((F, -1*F))
    f = np.concatenate((a_ub[not is_equal], -1*a_lb[not is_equal]))
    dim_f = len(f)

    H = A[is_equal, :]
    h = a_ub[is_equal]
    dim_h = len(h)
    dim_x = len(p)

    # Nested function for computing the dual function values (only one input x to be scipy optimizer -compatible)
    def dual(var):
        l, v = var[:dim_h], var[dim_h:] # separate equality and inequality dual variables
        x = p * np.exp(-1 - np.dot(np.transpose(F), l) - np.dot(np.transpose(H), v)) # primal solution (with l, v fixed)
        L = np.dot(x, np.log(x) - np.log(p)) + np.dot(l, np.dot(F, x) - f) + np.dot(v, np.dot(H, x) - h) # Dual value
        return L

    # Nested function for computing the Jacobian of the dual function (needed by scipy optimizer)
    def Jac(var):
        l, v = var[:dim_f], var[dim_f:] # separate equality and inequality dual variables

        # Interim results: (ln x - ln p), dxdl and dxdv
        x = p * np.exp(-1 - np.dot(np.transpose(F), l) - np.dot(np.transpose(H), v))
        lnx_lnp = - np.ones(dim_x) - np.dot(np.transpose(F), l) - np.dot(np.transpose(H), v)
        dxdl = -p * np.exp(-1) * np.exp(-1 * np.dot(np.transpose(H), v)) * np.exp(-1 * np.transpose(F), l) * np.transpose(F)
        dxdv = -p * np.exp(-1) * np.exp(-1 * np.dot(np.transpose(F), l)) * np.exp(-1 * np.transpose(H), v) * np.transpose(H)

        # Jacobian vectors wrt to lambda and nu separately
        Jacl = -1 * np.dot(x, np.transpose(F)) + np.dot(lnx_lnp, dxdx) + np.transpose(np.dot(F, x)) -
            np.transpose(f) + (np.dot(l, F) + np.dot(np.dot(v, H)), dxdl)
        Jacv = -1 * np.dot(x, np.transpose(H)) + np.dot(lnx_lnp, dxdv) + np.transpose(np.dot(H, x)) -
            np.transpose(h) + (np.dot(v, H) + np.dot(np.dot(l, F)), dxdv)

        Jac = np.concatenate((Jacl, Jacv))
        return Jac

    bounds = np.concatenate((np.repeat((0, None), dim_f), np.repeat((None, None), dim_f)))
    res = opt.minimize(fun = lambda x : -1 * dual(x), x0 = p, method = 'CG', jac = Jac, bounds = bounds)
    l_opt, v_opt = res.x[:dim_f], res.x[dim_f:]

    posterior = p * np.exp(-1 - np.dot(np.transpose(F), l_opt) - np.dot(np.transpose(H), v_opt))
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
        p_c = (1 - c)*p_prior + c*p_post
    else:
        p_c = np.outer(p_prior, 1 - c) + np.outer(p_post, c)

    return p_c
