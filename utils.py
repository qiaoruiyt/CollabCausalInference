import os
import sys
import numpy as np
from scipy.stats import bootstrap, norm

from math import factorial
from itertools import chain, combinations
import matplotlib.pyplot as plt


def get_bootstrap_func(data, Estimator):
    def bf(idx):
        x = data[idx, :-2]
        t = data[idx, -2]
        y = data[idx, -1]
        est = Estimator(x, t, y, get_std=False)
        ate = est.ate
        return ate
    return bf


def _get_se_bootstrap(idx, data, Estimator, n_resamples=5000):
    bf = get_bootstrap_func(data, Estimator)
    res = bootstrap(idx, bf, vectorized=False, n_resamples=n_resamples)
    # return res.confidence_interval
    return res.standard_error


def get_std_bootstrap(x, t, y, Estimator, n_resamples=5000):
    '''
    Compute standard error of the ATE estimate using bootstrap
    '''
    idx = np.arange(x.shape[0])
    idx = (idx,)
    # t = np.expand_dims(t, 1)
    # y = np.expand_dims(y, 1)
    data = collate_causal_data(x,t,y)
    std = _get_se_bootstrap(idx, data, Estimator=Estimator, n_resamples=500)
    return std


def collate_causal_data(x, t, y):
    data = np.concatenate([x,t,y], axis=1)
    return data


def rkl_for_ate_estimate(estimate, target, z=1.96):
    # target is also in form of an ATE estimate
    y_hat = estimate.ate 
    y_hat_std = estimate.std
    y = target.ate 
    y_std = target.std
    ed = negative_kl(y, y_std, y_hat, y_hat_std)
    return ed 


def negative_kl(target, target_std, y_hat, std_hat):
    '''
    Compute -KL(q||p), where p is the target distribution, q is the estimated distribution
    '''
    return -gaussian_reverse_kl(target, target_std, y_hat, std_hat)


def gaussian_reverse_kl(mu1, sigma1, mu2, sigma2):
    kl = np.log(sigma1/sigma2) + (sigma2**2+(mu1-mu2)**2)/2/sigma1**2 - 1/2
    return kl


def abse_for_ate_estimate(estimate, target):
    y_hat = estimate.ate 
    y = target.ate 
    abse = -abs_err(y, y_hat)
    return abse


def abs_err(y, y_hat):
    return np.abs(y-y_hat)


def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return list(chain.from_iterable(combinations(s, r) for r in range(0, len(s)+1)))


def valuate_dataset(S, target, Estimator, value_fn=rkl_for_ate_estimate, offset=0):
    '''Perform valuation for the data of coalition S

    Args:
        S: a list of data points, each data point is a tuple (x, t, y)
        target: the target ATE estimate
        Estimator: the estimator class that will be used to estimate the ATE for S
        value_fn: the valuation function that will be used to compute the valuation of S based on the ATE estimate

    Returns:
        v: the valuation of the data of coalition S
    '''
    x, t, y = [], [], []
    for i in S:
        xi, ti, yi = i
        x.append(xi)
        t.append(ti)
        y.append(yi)
    x = np.concatenate(x, axis=0)
    t = np.concatenate(t, axis=0)
    y = np.concatenate(y, axis=0)

    estimate = Estimator(x, t, y)
    v = value_fn(estimate, target) + offset
    return v


def compute_shapley_coef(s, n):
    return factorial(s)*factorial(n-s-1)


def compute_shapley_idx(X, idx, dvf, v_ind=None):
    '''Compute Shapley value for a single dataset with index specified by idx

    Args:
        X: a list of datasets in 2D-shaped numpy array
        idx: the index of the dataset in X
        dvf: the data valuation function
        v_ind: the standalone valuation of each dataset in X

    Returns:
        sh: the Shapley value of the dataset with index idx
        marginal_contributions: the marginal contribution of each dataset in X to the Shapley value of the dataset with index idx
    '''

    indices = np.arange(len(X))
    indices = np.delete(indices, idx) # remove the index of the dataset with index idx
    n = len(X) # num of parties in X
    ps = powerset(indices)
    sh = 0

    if v_ind is None:
        v_ind = [dvf([X[i]]) for i in range(n)]
    v0 = min(v_ind)


    marginal_contributions = []

    # Compute the Shapley value
    for s_idx in ps:
        s_idx = np.array(s_idx)
        # Marginal contribution for the empty coalition
        if s_idx.shape[0] == 0: 
            v_i = v_ind[idx]
            if v_i < 0:
                sh += factorial(n-1)*(v_i - v0)
                # sh += factorial(n-1)*v_i
            else:
                sh += factorial(n-1)*(v_i - v0)
                # sh += factorial(n-1)*v_i
            continue

        # Marginal contribution for non-empty coalitions
        S = [X[i] for i in s_idx]
        S_union_i = S+[X[idx]]
        s = len(S) # num of parties in coalition S

        v_S =  dvf(S) 
        v_S_union_i = dvf(S_union_i)
        marginal_contrib = v_S_union_i - v_S
        sh += compute_shapley_coef(s, n)*marginal_contrib
        marginal_contributions.append(marginal_contrib)
    
    sh = sh/factorial(n)
    return sh, marginal_contributions


def get_dataset_vf(X, target, Estimator, valuate=valuate_dataset, value_fn=rkl_for_ate_estimate, verbose=False):

    def dvf(S, offset=0):
        return valuate(S, target, Estimator, value_fn, offset=offset)
        
    v_ind = []
    for Di in X: 
        v_ind.append(dvf([Di]))
    return dvf, v_ind


def compute_shapley(X, dvf, v_ind=None, verbose=False, log=False):
    '''Compute Shapley value for each dataset in X

    Args:
        X: a list of datasets in 2D-shaped numpy array
        dvf: the data valuation function

    Returns:
        sh_vec: a list of Shapley values for each dataset in X
    '''
    
    sh_vec = []
    mc_vec = []
    for i in range(len(X)):
        sh, mc = compute_shapley_idx(X, i, dvf, v_ind)
        sh_vec.append(sh)
        mc_vec.append(mc)

    if log:
        return sh_vec, mc_vec
    return sh_vec


def plot(v_ind, sh, return_value, dataset, estimator, figure_folder='figures/'):
    import matplotlib
    import matplotlib.pyplot as plt
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42
    matplotlib.rcParams['font.family'] = 'Helvetica'
    matplotlib.rcParams['font.size'] = 13

    if not os.path.exists(figure_folder):
        os.makedirs(figure_folder)
    linewidth=1
    markersize=8
    xs = np.arange(1, len(return_value)+1)
    xticks = np.arange(1, len(return_value)+1, step=1)
    plt.figure(dpi=150)
    plt.plot(xs, v_ind, 'yo-', label='standalone value', linewidth=linewidth, markersize=markersize)
    plt.plot(xs, return_value, 'bo-', label='reward value', linewidth=linewidth, markersize=markersize)
    plt.plot(xs, sh, 'r^-', label='shapley value', linewidth=linewidth, markersize=markersize)
    plt.xticks(xticks)
    plt.xlabel('Party Index')
    plt.ylabel('Valuation')
    plt.legend()
    plt.savefig(figure_folder+str(dataset).lower()+'_'+str(estimator).lower()+'_'+estimator.model_name.lower()+'.png')


def plot_gaussians(means, stds, colors=None, alpha=0.4, linewidth=1, markersize=8, xlabel='x', ylabel='Probability Density', title='Multi-Gaussian Distribution'):
    xmin = min(means) - max(stds)
    xmax = max(means) + max(stds)
    x = np.linspace(xmin, xmax, 1000)
    fig, ax = plt.subplots(dpi=150)
    if colors is None:
        cmap = plt.cm.get_cmap('viridis', len(means))
        colors = [cmap(i) for i in range(len(means))]
    for i, (mean, std) in enumerate(zip(means, stds)):
        y = np.exp(-(x-mean)**2 / (2*std**2)) / (std * np.sqrt(2*np.pi))
        # ax.fill_between(x, y, color=colors[i], edgecolor=None, alpha=alpha, label=f'$\mu={mean}$, $\sigma={std}$')
        ax.fill_between(x, y, color=colors[i], edgecolor=None, alpha=alpha)
        ax.axvline(mean, color=colors[i], linestyle='-', linewidth=linewidth, ymax=0.9)
            
    ax.set_ylim(bottom=0) 
    
    # ax.legend()
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    return ax
    # plt.show()
    
    
def visualize_estimates(ret, ground_truth, dataset=None, estimator=None, colors=['red', 'green', 'blue', 'yellow', 'cyan'], figure_folder='figures/'):   
    if not os.path.exists(figure_folder):
        os.makedirs(figure_folder)
    plt.figure(dpi=150)
    means = [i[0] for i in ret]
    stds = [i[1] for i in ret]
    ax = plot_gaussians(means, stds, colors=colors, xlabel='ATE Estimate', ylabel='Probability Density', title='Reward Distribution')
    ax.axvline(ground_truth, color='black', linestyle='--', linewidth=0.5, label='Ground Truth')
    plt.legend()
    if dataset is not None and estimator is not None:
        plt.savefig(figure_folder+'reward_'+str(dataset).lower()+'_'+str(estimator).lower()+'_'+estimator.model_name.lower()+'.png')
    else:
        print("Showing the plot...")
        plt.show()
    return plt