import cupy as cp
import numpy as np
import time
import sporco.cupy.admm as spc
from sklearn.decomposition import *

def dict_learn_(X,n_components,alpha, max_iter=100, verbose=1):
    t0=time.time()
    code, S, dict=cp.linalg.svd(X)
    dict=S[:,np.newaxis]*dict
    r = len(dict)
    if n_components <= r:  # True even if n_components=None
        code = code[:, :n_components]
        dict = dict[:n_components, :]
    else:
        code = cp.c_[code, cp.zeros((len(code), n_components - r))]
        dict = cp.r_[dict, cp.zeros((n_components - r, dictshape[1]))]

    # Fortran-order dict, as we are going to access its row vectors
    dict = cp.array(dict, order='F')

    residuals = 0

    errors = []
    current_cost = cp.nan

    if verbose == 1:
        print('[dict_learning]', end=' ')

        # If max_iter is 0, number of iterations returned should be zero
    ii = -1

    for ii in range(max_iter):
        dt = (time.time() - t0)
        if verbose == 1:
            sys.stdout.write(".")
            sys.stdout.flush()
        elif verbose:
            print("Iteration % 3i "
                  "(elapsed time: % 3is, % 4.1fmn, current cost % 7.3f)"
                  % (ii, dt, dt / 60, current_cost))

        # Update code
        # code = sparse_encode(X, dictionary, algorithm=method, alpha=alpha,init=code, n_jobs=n_jobs, positive=positive_code, max_iter=method_max_iter, verbose=verbose)
        code=spc.BPDN(dict,X,0.3)
        code.solve()
        code=code.obfn_reg()[1]
        # Update dictionary
        dict, residuals = _update_dict(dictionary.T, X.T, code.T,verbose=verbose, return_r2=True)
        dict = dict.T

        # Cost function
        current_cost = 0.5 * residuals + alpha * cp.sum(cp.abs(code))
        errors.append(current_cost)

        if ii > 0:
            dE = errors[-2] - errors[-1]
            # assert(dE >= -tol * errors[-1])
            if dE < tol * errors[-1]:
                if verbose == 1:
                    # A line return
                    print("")
                elif verbose:
                    print("--- Convergence reached after %d iterations" % ii)
                break

    return code, dictionary, errors

