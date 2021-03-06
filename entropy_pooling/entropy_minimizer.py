# imports
import numpy as np
import scipy.optimize as opt

def full_confidence_posterior(p, A, b, C, d, verbose = False):
    """Computes the full-condifence posterior distribution by finding the constrained entropy-minimizing set of variables ('posterior distribution')
    -------------------
    ### Input arguments:
        p: numpy.ndarray
            the S-element vector of prior probabilities
        A: numpy.ndarray
            the (J x S) matrix used to express the constraints Ax = b
        b: numpy.ndarray
            the J-element upper bound vector (1 x J) for equality constraints
        C: numpy.ndarray
            the (K x S) matrix used to express the constraints Cx <= d
        d: numpy.ndarray
            the K-element lower bound vector for inequality constraints
        verbose: Boolean, Default False
            weather to print the result or not
    -------------------
    ### Returns:
        posterior: numpy.ndarray
            the S-element vector of posterior probabilities
    """

    # Change p, b, d shapes to simple np.ndarray if they are more complex
    if p.ndim > 1:
        p = p.reshape(p.shape[0],)
    if b.ndim > 1:
        b = b.reshape(b.shape[0],)
    if d.ndim > 1:
        d = d.reshape(d.shape[0],)

    # Check that the dimensions of the input arguments match
    assert len(p) == A.shape[1], 'A and p dimensions mismatch'
    assert len(p) == C.shape[1], 'C and p dimensions mismatch'
    assert A.shape[0] == len(b), 'A and b dimensions mismatch'
    assert C.shape[0] == len(d), 'C and d dimensions mismatch'

    dim_b = len(b)
    dim_d = len(d)
    dim_x = len(p)

    # Nested function for computing the dual function values (only one input x to be scipy optimizer -compatible)
    def dual(var):
        l, v = var[:dim_d], var[dim_d:] # separate equality and inequality dual variables
        x = p * np.exp(-1 - C.T @ l - A.T @ v) # primal solution (with l, v fixed)
        L = x @ (np.log(x) - np.log(p)) + l @ (C @ x - d) + v @ (A @ x - b) # Dual value
        return -1 * L # we maximize dual but scipt.opt only mnimizes, thus minus sign

    # Nested function for computing the Jacobian of the dual function (needed by scipy optimizer)
    def Jac(var):
        l, v = var[:dim_d], var[dim_d:] # separate equality and inequality dual variables

        # Interim results: (ln x - ln p), dx/dl and dx/dv
        x = p * np.exp(-1 - C.T @ l - A.T @ v)
        lnx_lnp = - np.ones(dim_x) - C.T @ l - A.T @ v
        dxdl = -p * np.exp(-1) * np.exp(-1 * A.T @ v) * np.exp(-1 * C.T @ l) * C
        dxdv = -p * np.exp(-1) * np.exp(-1 * A.T @ v) * np.exp(-1 * C.T @ l) * A

        # Jacl = df(l, v)/dl; Jacv = df(l, v)/dv
        Jacl = dxdl @ lnx_lnp - d + dxdl @ (l @ C + v @ A)
        Jacv = dxdv @ lnx_lnp - b + dxdv @ (v @ A + l @ C)

        Jac = np.concatenate((Jacl, Jacv))
        return -1 * Jac # we minimize the negative of dual --> we need minus sign in gradient

    bounds = (((0, None), ) * dim_d) + (((None, None), ) * dim_b) # ineq Lagr multipliers nonneg, eq multipliers unrestricted
    res = opt.minimize(fun = dual, x0 = np.ones(dim_d + dim_b), method = 'TNC', jac = Jac, bounds = bounds)
    if verbose:
        print("Results")
        print("Optimal dual variable values: ", res.x)
        print("Jacobian matrix at optimum", Jac(res.x))
    l_opt, v_opt = res.x[:dim_d], res.x[dim_d:]

    posterior = p * np.exp(-1 - C.T @ l_opt - A.T @ v_opt)
    return posterior

def confidence_weighted_posterior(p_prior, p_post, c):
    """Computes the confidence-weighted posterior distribution, in fact, computing a c-weighted average of the prior and posterior distributions
    -------------------
    ### Input arguments:
    p_prior: numpy.ndarray
        the S-element vector of prior probabilities, expected type numpy.ndarray
    p_post:  numpy.ndarray
        the S-element vector of posterior probabilities, output from full_confidence_posterior, expected type numpy.ndarray
    c:       int or float or numpy.ndarray
        a scalar or C-element vector giving the confidence weight(s).
    -------------------
    ### Returns:
    p_weighted: numpy.ndarray
        the S-element vector or (S x C) element matrix containing the weighted probabilities
    """

    if p_prior.ndim > 1:
        p_prior = p_prior.reshape(len(p_prior),)
    if p_post.ndim  > 1:
        p_post  = p_post.reshape(len(p_post),)

    assert len(p_prior) == len(p_post), 'Lengths of prior and posterior vectors do not match'

    if type(c) in [int, float]:
        assert c >= 0 and c <= 1, 'Value of c must be between 0 and 1'
        p_weighted = (1 - c)*p_prior + c*p_post
    elif type(c) == np.ndarray:
        assert np.all(c >= 0) and np.all(c <= 1), 'All values of c must be between 0 and 1'
        p_weighted = np.outer(p_prior, 1 - c) + np.outer(p_post, c)
    else:
        raise Exception('c has wrong type')

    return p_weighted
