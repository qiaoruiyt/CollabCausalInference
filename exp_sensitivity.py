from datasets import *
import numpy as np

from argparse import ArgumentParser

import matplotlib
import matplotlib.pyplot as plt
import warnings
from ate_func import *
from utils import *
from datasets import IHDP, Synthetic, JOBS
from ate_func import POREstimator
from reward import compute_rho, compute_reward_value, sample_return
from utils import *
warnings.filterwarnings('ignore') 
import random
import mkl
from scipy.stats import kendalltau


def plot_full(arrays, ylabel='Valuation', dataset=None, figure_folder='sens_plots_full/'):
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42
    matplotlib.rcParams['font.size'] = 13

    if not os.path.exists(figure_folder):
        os.makedirs(figure_folder)
    linewidth=1
    markersize=8
    plt.figure(dpi=150)


    cmap = plt.cm.get_cmap('viridis', len(arrays))
    colors = [cmap(i) for i in range(len(arrays))]

    for i in range(len(arrays)):
        array = arrays[i]
        color = colors[i]
        if i == len(arrays)//2:
            color = 'red'
        xs = np.arange(1, len(array)+1)
        xticks = np.arange(1, len(array)+1, step=1)
        if color == 'red':
            plt.plot(xs, array, label='default', linewidth=linewidth, markersize=markersize, color=color)
        plt.plot(xs, array, label=None, linewidth=linewidth, markersize=markersize, color=color)

    plt.xticks(xticks)
    plt.xlabel('Party Index')
    plt.ylabel(ylabel)
    plt.legend()
    plt.savefig(figure_folder+str(dataset).lower()+'_'+ylabel.replace(' ', '_').lower()+'.png')

def plot(x, y, xlabel='Perturbation', ylabel='Valuation', dataset=None, figure_folder='sens_plots/'):
    # y is (n x m) array, where m is the number of parties
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42
    matplotlib.rcParams['font.size'] = 13

    if not os.path.exists(figure_folder):
        os.makedirs(figure_folder)
    linewidth=1
    markersize=2
    plt.figure(dpi=150)

    import matplotlib.ticker as mtick
    x = x-1

    for i in range(len(y[0])):
        yi = [yj[i] for yj in y]
        # color = colors[i]
        plt.plot(x, yi, label='party '+str(i+1), linewidth=linewidth, markersize=markersize)
        original = yi[len(yi)//2]
        plt.plot(x[len(x)//2], original, 'ro', label='original', linewidth=linewidth, markersize=markersize)
        plt.gca().xaxis.set_major_formatter(mtick.PercentFormatter(1.0))
        
        break

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.tight_layout()
    plt.savefig(figure_folder+str(dataset).lower()+'_'+ylabel.replace(' ', '_').lower()+'.png')

def run_sensitivity_full(args):
    # TODO: plot sensitivity to the accuracy of the surrogate of ATE 

    # We could use cosine similarity or kendall tau to measure the deviation of Shapley
    perturbation = np.arange(-10, 11, 0.2)*0.01+1
    vs = []
    shs = []
    rvs = []
    for i in perturbation:
        v_ind, sh, return_value, ret = sensitive_main(args, perturbation=i)
        vs.append(v_ind)
        shs.append(sh)
        rvs.append(return_value)
    
    plot(arrays=vs, ylabel='Valuation', dataset=args.dataset)
    plot(arrays=shs, ylabel='Shapley Value', dataset=args.dataset)
    plot(arrays=rvs, ylabel='Reward Value', dataset=args.dataset)

def run_sensitivity(args):
    perturbation = np.arange(-10, 11, 0.2)*0.02+1

    vs = []
    shs = []
    rvs = []
    for i in perturbation:
        v_ind, sh, return_value, ret = sensitive_main(args, perturbation=i)
        vs.append(v_ind)
        shs.append(sh)
        rvs.append(return_value)
    
    plot(perturbation, vs, ylabel='Valuation', dataset=args.dataset)
    plot(perturbation, shs, ylabel='Shapley Value', dataset=args.dataset)
    plot(perturbation, rvs, ylabel='Reward Value', dataset=args.dataset)

def run_edge(args):
    sensitive_main(args, edge=1e-3)


def sensitive_main(args, perturbation=1, edge=None):
    seed = args.seed
    np.random.seed(seed)
    random.seed(seed)

    if args.dataset == 'ihdp':
        dataset = IHDP(replications=args.reps)
    elif args.dataset == 'synthetic':
        dataset = Synthetic(n=100, replications=args.reps)
    elif args.dataset == 'tcga':
        dataset = TCGA(replications=args.reps)
    elif args.dataset == 'jobs':
        dataset = JOBS(replications=args.reps, binary=args.binary, filtered=args.filtered) 
    else: 
        raise Exception('unsupported dataset')
    n_partitions = args.n_partitions
    base_n = 0
    n_replications = dataset.replications 

    value_fn = negative_kl

    if args.estimator == 'por':
        Estimator = lambda x,t,y: POREstimator(x, t, y, model_name=args.model)
    elif args.estimator == 'ipw':
        Estimator = IPWEstimator
    elif args.estimator == 'aipw':
        Estimator = AIPWEstimator
    else:
        raise Exception('unsupported estimator')


    for i, (x, t, y, true_ate) in enumerate(dataset):
        X = []
        grand_estimate = Estimator(x, t, y)
        grand_ate = grand_estimate.ate
        # Mod for sensitivity analysis
        if edge is not None:
            grand_ate = edge
        else:
            grand_ate = grand_ate * perturbation
        grand_estimate.ate = grand_ate

        # print("Surrogate loss: ", np.abs(grand_ate-true_ate))
        
        t = t[:, np.newaxis]
        xall, tall, yall = x.copy(), t.copy(), y.copy()
        n = xall.shape[0]
        for j in range(base_n, n_partitions):
            cutoff_idx = j+1

            indices = np.arange(n*cutoff_idx//n_partitions) # inc
            indices = np.arange(n*j//n_partitions, n*cutoff_idx//n_partitions) #dis
            x = xall[indices]
            t = tall[indices]
            y = yall[indices]

            X.append((x,t,y))

        dvf, v_ind = get_dataset_vf(X, target=grand_estimate, Estimator=Estimator, verbose=True)

        sh = compute_shapley(X, dvf, v_ind)

        v0 = min(v_ind)
        rho = compute_rho(X, v_ind, sh)
        threshold = args.threshold
        if rho > threshold:
            rho = threshold
        r = compute_reward_value(sh, v_ind, rho)

        k = 2*grand_estimate.std
        r_max = max(r)
        reward_sampling_std = [k*np.sqrt((r_max - r[i])/(r_max-v0)) for i in range(len(r))]

        ret = [sample_return(grand_estimate.ate, grand_estimate.std, reward_sampling_std[i], r[i]) for i in range(len(r))]

        return_value = [value_fn(grand_ate, grand_estimate.std, ret[i][0], ret[i][1]) for i in range(len(r))]

        if edge is not None:
            visualize_estimates(ret, grand_ate, dataset, grand_estimate, figure_folder='edge_figures/')

        return v_ind, sh, return_value, ret

        break 


def test_salib(args): 
    from SALib.sample import saltelli
    from SALib.analyze import sobol
    from SALib.test_functions import Ishigami
    import numpy as np

    problem = {
    'num_vars': 1,
    'names': ['x1'],
    'bounds': [[-3.14159265359, 3.14159265359]]
    }
    param_values = saltelli.sample(problem, 1024)
    Y = np.zeros([param_values.shape[0]])

    print(param_values.shape)

    def test_function(x):
        return np.sin(x).squeeze()
        
    Y = test_function(param_values)
    
    Si = sobol.analyze(problem, Y)
    print(Si)
    print(Si['S1'])
    print(Si['ST'])


if __name__ == '__main__':
    warnings.filterwarnings('ignore') 
    parser = ArgumentParser()
    parser.add_argument('--reps', type=int, default=10)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--n_threads', type=int, default=8)
    parser.add_argument('--n_partitions', type=int, default=5)
    parser.add_argument('--dataset', type=str, default='synthetic', choices=['ihdp', 'synthetic', 'jobs', 'tcga'])
    parser.add_argument('--estimator', type=str, default='por', choices=['por', 'ipw', 'aipw'])
    parser.add_argument('--model', type=str, default='linear', choices=['linear', 'gp', 'xgboost'])
    parser.add_argument('--threshold', type=float, default=0.2)
    parser.add_argument('--binary', action='store_true')
    parser.add_argument('--filtered', action='store_true')
    parser.add_argument('--edge', action='store_true')
    parser.add_argument('--test', action='store_true')

    args = parser.parse_args()
    print(args)
    mkl.set_num_threads(args.n_threads)

    if args.test:
        test_salib(args)
    elif args.edge:
        run_edge(args)
    else:
        run_sensitivity(args)
    