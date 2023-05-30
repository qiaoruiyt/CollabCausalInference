from datasets import *
import numpy as np
import random
from argparse import ArgumentParser
import matplotlib.pyplot as plt
import warnings

from ate_func import *
from utils import *
from reward import compute_rho, compute_reward_value, sample_return

import logging 
logger = logging.getLogger(__name__)

import platform

def is_intel_cpu():
    return 'Intel' in platform.processor()
    
    
def main(args):
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
        
        try:
            print("Surrogate loss: ", np.abs(grand_ate-true_ate))
            print("Ground truth ATE: ", true_ate)
            print("Ground truth Surrogate: ", grand_ate)
        except:
            print("Ground truth Surrogate: ", grand_ate)
            print("True ATE not available")
        
        t = t[:, np.newaxis]
        xall, tall, yall = x.copy(), t.copy(), y.copy()
        n = xall.shape[0]
        for j in range(base_n, n_partitions):
            cutoff_idx = j+1

            # indices = np.arange(n*cutoff_idx//n_partitions) # inc
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
        # reward_sampling_std = [k*(r_max - r[i])/(r_max-v0) for i in range(len(r))]
        reward_sampling_std = [k*np.sqrt((r_max - r[i])/(r_max-v0)) for i in range(len(r))]

        ret = [sample_return(grand_estimate.ate, grand_estimate.std, reward_sampling_std[i], r[i]) for i in range(len(r))]

        return_value = [value_fn(grand_ate, grand_estimate.std, ret[i][0], ret[i][1]) for i in range(len(r))]

        print("Individual value:", v_ind)
        print("Shapley", sh)
        if rho > threshold:
            print("rho > threshold = {}, setting rho to the threshold to improve group welfare.".format(threshold))
        print("rho", rho)
        print("return values", ret)
        print("reward values", r)
        print("Actual return value:", return_value)

        plot(v_ind, sh, return_value, dataset, grand_estimate)
        visualize_estimates(ret, grand_ate, dataset, grand_estimate)

        # Currently we only run on the first replication to demonstrate the algorithm
        break 


if __name__ == '__main__':
    warnings.filterwarnings('ignore') 
    parser = ArgumentParser()
    parser.add_argument('--reps', type=int, default=10)
    parser.add_argument('--seed', type=int, default=3)
    parser.add_argument('--n_threads', type=int, default=8)
    parser.add_argument('--n_partitions', type=int, default=5)
    parser.add_argument('--dataset', type=str, default='jobs', choices=['ihdp', 'synthetic', 'jobs', 'tcga'])
    parser.add_argument('--estimator', type=str, default='por', choices=['por', 'ipw', 'aipw'])
    parser.add_argument('--model', type=str, default='linear', choices=['linear', 'gp', 'xgboost'])
    parser.add_argument('--threshold', type=float, default=0.2)
    parser.add_argument('--binary', action='store_true')
    parser.add_argument('--filtered', action='store_true', help='use filtered jobs dataset')
    parser.add_argument('--quiet', action='store_true', help='do not print and plot')

    args = parser.parse_args()
    print(args)
    if is_intel_cpu():
        import mkl
        mkl.set_num_threads(args.n_threads)

    main(args)