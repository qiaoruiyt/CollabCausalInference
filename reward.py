import numpy as np
import scipy.integrate as integrate
from scipy.stats import norm
from scipy.stats import bootstrap
from scipy.optimize import root_scalar
from utils import *


def compute_rho(X, v_ind, sh, verbose=False):
    """ Compute factor rho such that R1-R5 are satisfied.
    Args:
        X: a list of datasets in 2D-shaped numpy array
        v_ind: a list of individual values for each dataset in X
        sh: a list of Shapley values for each dataset in X

    Returns:
        rho_r5: the factor rho for satisfying R1-R5
    """

    sh_max = max(sh)
    v0 = min(v_ind)
    
    log_ratios_r5 = []
    for i in range(len(X)):
        if sh[i] < 0 or sh[i] == sh_max:
            # ignore cases with negative Shapley value
            continue
        log_r = np.log(-(v_ind[i]-v0)/v0)/np.log(sh[i]/sh_max)
        log_ratios_r5.append(log_r)

    if len(log_ratios_r5) == 0:
        print("No feasible rho detected. ")
        print("Shapley values:", sh)
        
    rho_r5 = min(log_ratios_r5) # rho for R1-R5
    if verbose:
        print("rho R1-R5: ", log_ratios_r5)
    return rho_r5


def compute_reward_value(sh, v_ind, rho):
    """ Compute the rho-shapley fair reward value
    Args:
        sh: a list of Shapley values for each dataset in X
        v_ind: a list of individual values for each dataset in X
        rho: the factor rho for satisfying R1-R5

    Returns:
        r: a list of rho-Shapley fair reward values for each dataset in X
    """
    sh_max = max(sh)
    v0 = min(v_ind)

    r =  [max(v_ind[i] - v0, (sh[i]/sh_max)**rho*(-v0)) + v0 for i in range(len(sh))]
    return r


def solve_quadratic_equation(a, b, c):
    s1 = (-b + (b ** 2 - 4 * a * c) ** 0.5) / (2 * a) 
    s2 = (-b - (b ** 2 - 4 * a * c) ** 0.5) / (2 * a) 
    return s1, s2


def solve_std_root(tau, target_ate, target_std, value, value_fn=negative_kl, min_std=1e-10, verbose=False):
    '''Root finding algorithm to solve for std
    Args:
        tau: the sample mean
        target_ate: the target ate
        target_std: the target std
        value: the reward value of the dataset
        value_fn: the data valuation function (also works for ATE estimates)

    Returns:
        std: a solution such that (tau, std) and (target_ate, target_std) should have the same value
    '''
    def f_to_solve(x):
        return value - value_fn(target_ate, target_std, tau, x)
    
    try:
        result = root_scalar(f_to_solve, bracket=[target_std, 1e5],method='brentq') 
        std = result.root
        return std
    except:
        try: 
            result = root_scalar(f_to_solve, bracket=[min_std, target_std],method='brentq') 
            std = result.root
            return std
        except Exception as e:
            print(e)
            return None


def sample_return(target_ate, target_std, std, r, offset=0, alpha=1.96, verbose=False):
    '''Sample a return in terms of (tau, std) such that the constraints are satisfied
    Args:
        target_ate: the target ate
        target_std: the target std
        std: the std of the sample (defined inversely proportional to the reward value)
        r: the reward value of the sample
        offset: the offset of the reward value
        alpha: the confidence level for the confidence interval
    
    Returns:
        tau: the sample mean
        se: the standard error of the sample mean
    '''
    # NOTE: works for kl divergence
    
    tau = -np.inf*np.sign(target_ate)
    se = None
    num_iter = 0
    v = (r-offset)
    perturbation = target_std*np.sqrt(2*(-v))
    perturb_bound = [target_ate-perturbation, target_ate+perturbation]
    while(np.sign(tau) != np.sign(target_ate)) or se is None:
        tau = -np.inf*np.sign(target_ate)
        while tau > perturb_bound[1] or tau < perturb_bound[0]:
            tau = np.random.normal(target_ate, std)
        if tau < target_ate:
            tau_for_se = target_ate + np.abs(target_ate - tau)
        else:
            tau_for_se = tau
        se = solve_std_root(tau_for_se, target_ate, target_std, v, verbose=verbose)
        num_iter += 1 
        if num_iter >= 1e5:
            raise Exception ("Cannot find a feasible sample in time. Something is wrong")
    return tau, se
