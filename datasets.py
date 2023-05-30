from mimetypes import init
from matplotlib.pyplot import axis
import numpy as np
from sklearn.model_selection import train_test_split

class IHDP(object):
    def __init__(self, path_data="datasets/IHDP/csv", replications=10):
        # there is only 10 hard simulated replications
        # If n_replications > 10, we will reuse them with shuffled indices
        self.path_data = path_data
        self.replications = replications
        # which features are binary
        self.binfeats = [6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]
        # which features are continuous
        self.contfeats = [i for i in range(25) if i not in self.binfeats]
        self.binary = False

    def __str__(self) -> str:
        return 'IHDP'

    def __iter__(self):
        for i in range(self.replications):
            # file_id = i % 10
            file_id = 0
            data = np.loadtxt(self.path_data + '/ihdp_npci_' + str(file_id + 1) + '.csv', delimiter=',')

            idx = np.arange(data.shape[0])
            np.random.shuffle(idx)
            data = data[idx]

            t, y, y_cf = data[:, 0][:, np.newaxis], data[:, 1][:, np.newaxis], data[:, 2][:, np.newaxis]
            mu_0, mu_1, x = data[:, 3][:, np.newaxis], data[:, 4][:, np.newaxis], data[:, 5:]
            x[:, 13] -= 1 # this binary feature is in {1, 2}
            
            true_ite = mu_1 - mu_0
            ate = np.mean(true_ite)
            yield x, t, y, ate

    def get_train_valid_test(self):
        for i in range(self.replications):
            data = np.loadtxt(self.path_data + '/ihdp_npci_' + str(i + 1) + '.csv', delimiter=',')
            t, y, y_cf = data[:, 0][:, np.newaxis], data[:, 1][:, np.newaxis], data[:, 2][:, np.newaxis]
            mu_0, mu_1, x = data[:, 3][:, np.newaxis], data[:, 4][:, np.newaxis], data[:, 5:]
            # this binary feature is in {1, 2}
            x[:, 13] -= 1
            idxtrain, ite = train_test_split(np.arange(x.shape[0]), test_size=0.1, random_state=1)
            itr, iva = train_test_split(idxtrain, test_size=0.3, random_state=1)
            train = (x[itr], t[itr], y[itr]), (y_cf[itr], mu_0[itr], mu_1[itr])
            valid = (x[iva], t[iva], y[iva]), (y_cf[iva], mu_0[iva], mu_1[iva])
            test = (x[ite], t[ite], y[ite]), (y_cf[ite], mu_0[ite], mu_1[ite])
            yield train, valid, test, self.contfeats, self.binfeats



class JOBS(object):
    def __init__(self, replications=1, filtered=False, binary=False) -> None:
        '''
        # The dataset is the full version designed by Lalonde, which is downloaded from https://users.nber.org/~rdehejia/data/.nswdata2.html
        #   Note that this version does not contain RE74 by default due to missing values 
        # An alternative version that removed entries missing RE74 in RCT is available in R (https://rdrr.io/cran/qte/man/lalonde.html) 
        #   Note that this contain one more feature column on RE74
        '''
        # Binary meaning only cares about employment vs unemployment, regardless of the earnings 
        self.binary = binary

        self.replications = replications
        if not filtered:
            rct_control = self.read_txt("datasets/JOBS/nsw_control.txt")
            rct_treated = self.read_txt("datasets/JOBS/nsw_treated.txt")
            obs = self.read_txt("datasets/JOBS/psid_controls.txt")
            # remove RE74 
            obs = np.delete(obs, -3, axis=1)
        else: 
            rct_control = self.read_txt("datasets/JOBS/nswre74_control.txt")
            rct_treated = self.read_txt("datasets/JOBS/nswre74_treated.txt")
            obs = self.read_txt("datasets/JOBS/psid_controls.txt")

        self.rct_treated = rct_treated
        self.rct_control = rct_control

        if binary:
            rct_treated[rct_treated[:,-1]>0, -1] = 1
            rct_treated[rct_treated[:,-1]==0, -1] = 0
            rct_control[rct_control[:,-1]>0, -1] = 1
            rct_control[rct_control[:,-1]==0, -1] = 0
            obs[obs[:,-1]>0, -1] = 1
            obs[obs[:,-1]==0, -1] = 0
        
        y1 = rct_treated[:, -1]
        y0 = rct_control[:, -1]
        ate = y1.mean() - y0.mean() 

        # The training data contains both rct_treated and obs, but not the rct_control so that the model has no knowledge about it
        data = np.concatenate([rct_treated, rct_control], axis=0)

        self.ate = ate 
        self.data = data

    def __str__(self) -> str:
        return 'JOBS'
    
    def __iter__(self):
        for i in range(self.replications):

            idx = np.arange(self.data.shape[0])
            np.random.shuffle(idx)
            data = self.data[idx]

            t = data[:, 0].astype(np.bool8)[:, np.newaxis]
            x = data[:, 1:-1]
            y = data[:, -1][:, np.newaxis]
            yield x, t, y, self.ate

    def get_data(self):
        t = self.data[:, 0].astype(np.bool8)
        x = self.data[:, 1:-1]
        y = self.data[:, -1]
        return x, t, y, self.ate

    def read_txt(self, filename):
        data = []
        with open(filename) as f:
            for line in f.readlines():
                datum = line.strip().split() 
                datum = [float(i) for i in datum]
                data.append(datum)
        data = np.array(data)
        return data

    def get_treated_data(self):
        return self.rct_treated[:, 1:-1]


class TCGA(object):
    def __init__(self, replications=1, num_features=50) -> None:
        '''
        # The dataset is adopted from https://github.com/d909b/perfect_match 
        # Originally it has 20531 RNA-sequence features but we only selected 500 of them as covariates
        '''
        self.replications = replications
        data = self.read_txt("datasets/TCGA/tcga_compact.txt")
        self.ate = None 
        self.data = data
        self.num_features = num_features 
        self.binary = True

    def __str__(self) -> str:
        return 'TCGA'
    
    def __iter__(self):
        for i in range(self.replications):

            idx = np.arange(self.data.shape[0])
            np.random.shuffle(idx)
            data = self.data[idx]

            t = data[:, 0].astype(np.bool8)[:, np.newaxis]
            x = data[:, 1:-1]
            if self.num_features > x.shape[1]:
                raise Exception("num_features specified is more than available {}".format(x.shape[1]))
            x = x[:, :self.num_features]
            y = data[:, -1][:, np.newaxis]
            yield x, t, y, self.ate

    def get_data(self):
        t = self.data[:, 0].astype(np.bool8)
        x = self.data[:, 1:-1]
        y = self.data[:, -1]
        return x, t, y, self.ate

    def read_txt(self, filename):
        data = []
        with open(filename) as f:
            for line in f.readlines():
                datum = line.strip().split() 
                datum = [float(i) for i in datum]
                data.append(datum)
        data = np.array(data)
        return data



class Synthetic(object):
    def __init__(self, n=100, p=0.5, replications=10, seed=0):
        self.binary = False
        self.replications = replications
        self.replicated_data = []
        np.random.seed(seed)
        # z = np.random.binomial(1, p, n)
        # for i in range(replications):
        #     z = np.random.binomial(1, p, n)
        #     x = np.random.normal(z, 9*z+25*(1-z))
        #     pt = 0.75*z+0.25*(1-z)
        #     t = np.random.binomial(1, pt)
        #     y = np.random.binomial(1, self.sigmoid(3*z+2*(2*t-1)))
        #     y_cf = np.random.binomial(1, self.sigmoid(3*z+2*(2*(1-t)-1)))
        #     mu_0 = self.sigmoid(3*z+2*(0-1))
        #     mu_1 = self.sigmoid(3*z+2*(2*1-1))
        #     data = np.stack([t, y, y_cf, mu_0, mu_1, x], axis=1)
        #     self.replicated_data.append(data)

        # simple confounder: such that x is the true confounder
        for i in range(replications):
            x = np.random.normal(0, 5, n)
            # x = np.random.binomial(1, p, n)
            pt = self.sigmoid(x)
            t = np.random.binomial(1, pt)
            y_prob = self.sigmoid(3*(-x+3*t-1))
            y_cf_prob = self.sigmoid(3*(-x+3*(1-t)-1))
            y = np.random.binomial(1, y_prob)
            y_cf = np.random.binomial(1, y_cf_prob)
            mu_0 = self.sigmoid(3*(-x+3*0-1))
            mu_1 = self.sigmoid(3*(-x+3*1-1))
            data = np.stack([t, y, y_cf, mu_0, mu_1, x], axis=1)
            self.replicated_data.append(data)

    def __str__(self) -> str:
        return 'Synthetic'

    def sigmoid(self, x):
        return 1/(1 + np.exp(-x))

    def __iter__(self):
        for i in range(self.replications):
            data = self.replicated_data[i]
            t, y, y_cf = data[:, 0], data[:, 1][:, np.newaxis], data[:, 2][:, np.newaxis]
            mu_0, mu_1, x = data[:, 3][:, np.newaxis], data[:, 4][:, np.newaxis], data[:, 5:]
            # yield (x, t, y), (y_cf, mu_0, mu_1)
            true_ite = mu_1 - mu_0
            ate = np.mean(true_ite)
            yield x, t, y, ate


    def get_all_data(self):
        data = np.concatenate(self.replicated_data, axis=0)
        t, y, y_cf = data[:, 0], data[:, 1][:, np.newaxis], data[:, 2][:, np.newaxis]
        mu_0, mu_1, x = data[:, 3][:, np.newaxis], data[:, 4][:, np.newaxis], data[:, 5:]
        return (x, t, y), (y_cf, mu_0, mu_1)

    def construct_unbiased_small_subset(self):
        x = np.array([0 for i in range(24)])
        x[:12] = 1
        t = np.array([0 for i in range(24)])
        t[:9] = 1
        t[-6:] = 1
        mu_1 = self.sigmoid(3*(-x+3*t-1))
        mu_0 = self.sigmoid(3*(-x+3*(1-t)-1))
        y = np.random.binomial(1, mu_0)
        y_cf = np.random.binomial(1, mu_1)

        data = np.stack([t, y, y_cf, mu_0, mu_1, x], axis=1)
        t, y, y_cf = data[:, 0][:, np.newaxis], data[:, 1][:, np.newaxis], data[:, 2][:, np.newaxis]
        mu_0, mu_1, x = data[:, 3][:, np.newaxis], data[:, 4][:, np.newaxis], data[:, 5:]
        return (x, t, y), (y_cf, mu_0, mu_1)


class CausalMLSythetic(object):
    def __init__(self, replications=10, mode=1, n=1000, p=5, sigma=1.0, adj=0.) -> None:
        from causalml.dataset import synthetic_data
        self.replications = replications
        self.replicated_data = []
        for i in range(replications):
            y, X, treatment, tau, b, e = synthetic_data(mode, n, p, sigma, adj) 
            self.replicated_data.append((X, treatment, y, tau, b, e))

    def __iter__(self):
        for i in range(self.replications):
            data = self.replicated_data[i]
            x, t, y, ite, b, e = data
            yield x, t, y, ite




if __name__ == "__main__":
    print("Test loading datasets")
    # print("\nSynthetic Dataset from Causal ML......")
    # dataset = CausalMLSythetic()
    # for x, t, y, ite in dataset:
    #     print(x[0])
    #     print(t[0])
    #     print(y[0])
    #     print(ite[0])
    #     # print(b[0])
    #     break

    # print("\nJOBS......")
    # dataset = JOBS() 
    # print(dataset.ate)
    # dataset = JOBS(binary=True)
    # print(dataset.ate)

    print("\nTCGA")
    dataset = TCGA() 
    for x, t, y, _ in dataset:
        print(x[0])
        print(t[0])
        print(y[0])
