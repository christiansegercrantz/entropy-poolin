# imports
import numpy as np
import scipy.optimize as opt

# TODO: define the type of input arguments: lists/arrays or numpy arrays or DataFrames???
# current code is made for numpy arrays, let's see if the syntax works for pd DFs

def full_confidence_posterior(p, A, b, C, d):
    # Computes the full-condifence posterior distribution by finding the constrained
    # entropy-minimizing set of variables
    #
    # Input arguments:
    # p: a (S x 1) vector of prior probabilities
    # A: the (J x S) matrix used to express the constraints Ax = b
    # b: upper bound vector (1 x J) for constraints
    # C: the (K x S) matrix used to express the constraints Cx <= d
    # d: lower bound vector (1 x K) for inequality constraints

    # Check that the dimensions of the input arguments match
    if not len(p) == A.shape[1]:
        raise Exception('A and p dimensions mismatch')
    if not len(p) == C.shape[1]:
        raise Exception('C and p dimensions mismatch')
    if not A.shape[1] == len(b):
        raise Exception('A and b dimensions mismatch')
    if not C.shape[1] == len(d):
        raise Exception('C and d dimensions mismatch')

    dim_b = len(b)
    dim_d = len(d)
    dim_x = len(p)

    # Nested function for computing the dual function values (only one input x to be scipy optimizer -compatible)
    def dual(var):
        l, v = var[:dim_d], var[dim_d:] # separate equality and inequality dual variables
        x = p * np.exp(-1 - np.dot(np.transpose(C), l) - np.dot(np.transpose(A), v)) # primal solution (with l, v fixed)
        L = np.dot(x, np.log(x) - np.log(p)) + np.dot(l, np.dot(C, x) - d) + np.dot(v, np.dot(A, x) - b) # Dual value
        return -1 * L # we maximize dual but scipt.opt only mnimizes, thus minus sign

    # Nested function for computing the Jacobian of the dual function (needed by scipy optimizer)
    def Jac(var):
        l, v = var[:dim_d], var[dim_d:] # separate equality and inequality dual variables

        # Interim results: (ln x - ln p), dx/dl and dx/dv
        x = p * np.exp(-1 - np.dot(np.transpose(C), l) - np.dot(np.transpose(A), v))
        lnx_lnp = - np.ones(dim_x) - np.dot(np.transpose(C), l) - np.dot(np.transpose(A), v)
        dxdl = -p * np.exp(-1) * np.exp(-1 * np.dot(np.transpose(A), v)) * np.exp(-1 * np.dot(np.transpose(C), l)) * C # TODO: transpose(C)?
        dxdv = -p * np.exp(-1) * np.exp(-1 * np.dot(np.transpose(C), l)) * np.exp(-1 * np.dot(np.transpose(A), v)) * A # TODO: transpose(A)?

        Jacl = -1 * np.dot(C, x) + np.dot(dxdl, lnx_lnp) + np.dot(C, x) - d + np.dot(dxdl, np.dot(l, C) + np.dot(v, A))
        Jacv = -1 * np.dot(A, x) + np.dot(dxdv, lnx_lnp) + np.dot(A, x) - b + np.dot(dxdv, np.dot(v, A) + np.dot(l, C))

        Jac = np.concatenate((Jacl, Jacv))
        return -1 * Jac # we minimize the negative of dual --> we need minus sign in gradient

    bounds = (((0, None), ) * dim_d) + (((None, None), ) * dim_b) # ineq Lagr multipliers nonneg, eq multipliers unrestricted
    res = opt.minimize(fun = lambda x : dual(x), x0 = np.ones(dim_d + dim_b), method = 'TNC', jac = Jac, bounds = bounds)
    print(res.x)
    print(Jac(res.x))
    l_opt, v_opt = res.x[:dim_d], res.x[dim_d:]

    posterior = p * np.exp(-1 - np.dot(np.transpose(C), l_opt) - np.dot(np.transpose(A), v_opt))
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
    if not len(p_prior) == len(p_posterior):
        raise Exception('Lengths of prior and posterior vectors do not match')

    if type(c) in [int, float]:
        if c < 0 or c > 1:
            raise Exception('Value of c must be between 0 and 1')
        p_weighted = (1 - c)*p_prior + c*p_post
    elif type(c) == np.ndarray:
        if np.any(c < 0) or np.any(c > 1):
            raise Exception('All values of c must be between 0 and 1')
        p_weighted = np.outer(p_prior, 1 - c) + np.outer(p_post, c)
    else:
        raise Exception('c has wrong type')

    return p_weighted
