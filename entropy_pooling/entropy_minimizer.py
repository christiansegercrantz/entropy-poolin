# imports
import numpy as np
import scipy.optimize as opt

# TODO: define the type of input arguments: lists/arrays or numpy arrays or DataFrames???

def full_confidence_posterior(p, A, a_lb, a_ub):
    # Computes the full-condifence posterior distribution by minimizing the entropy
    #
    # Input arguments:
    # p: a (J x 1) vector of prior probabilities
    # A: the (K x J) matrix used to express the constraints
    # a_lb: lower bound vector (1 x K) for constraints
    # a_ub: upper bound vector (1 x K) for constraints
    
    # TODO: reform the constraints so that they can be expressed in arrays of <= and == constraints
    # Step 1: define the dual function G to be maximised (in terms of dual variables l = lambda, v = nu)
    def dual(l, v):
        # define first-order solution to dL/dx = 0
        x = p * np.exp(- 1 - np.dot(F, l) - np.dot(H, v))
        # and the dual function value is the value of the primal Lagrangian where x is as defined above
        # TODO: handle l, v, F, H, f, h as in the problem we have A, a_lb, A_ub
        L = np.dot(x, np.log(x) - np.log(p)) + np.dot(l, np.dot(F, x) - f) + np.dot(v, np.dot(H, x) - h)
        
    # Step 2: use some optimization package to find argmax G and G^*
    # TODO: turn min to max
    # res = opt.minimize(fun = dual(l, v), x0 = p, method = 'Newton-CG', jac = ???)
    # l_opt = res.x # TODO: handle l,v case
    # We get the optimal posterior probabilities by plugging in the optimal dual variables
    # posterior = p * np.exp(- 1 - np.dot(F, l_opt) - np.dot(H, v_opt))
    # return posterior
    
def confidence_weighted_posterior(p_prior, p_post, c):
    # Computes the confidence-weighted posterior distribution,
    # in fact, computing a c-weighted average of the prior and posterior distributions
    #
    # Input arguments:
    # p_prior: the (J x 1) vector of prior probabilities
    # p_post: the (J x 1) vector of posterior probabilities, output from full_confidence_posterior
    # c: a scalar or (L x 1) vector giving the confidence weight(s).
    
    p_c = (1 - c)*p_prior + c*p_post # TODO: hande cases of scalar and vector c.
    
    return p_c