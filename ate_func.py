import numpy as np
from sklearn.gaussian_process.kernels import RBF, DotProduct
from sklearn.gaussian_process import GaussianProcessClassifier, GaussianProcessRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression

from utils import *

import logging 
logger = logging.getLogger(__name__)


def compute_var_ate(y1, y0):
    # Compute the variance of ATE
    v = np.var(y1)/y1.shape[0]+np.var(y0)/y0.shape[0]
    return v


def get_y0_y1(x, t, model, binary=False):
    t0 = np.zeros_like(t)
    t1 = np.ones_like(t)
    x0 = np.concatenate([x, t0], axis=1)
    x1 = np.concatenate([x, t1], axis=1)
    if binary:
        y0 = model.predict_proba(x0)
        y1 = model.predict_proba(x1)
    else:
        y0 = model.predict(x0)
        y1 = model.predict(x1)
    return y0, y1


def compute_ate_dr(t, w, y, y0, y1):
    # The doubly robust estimator for ATE
    ate_aipw = np.mean(t*(y-y1)/w) - np.mean((1-t)*(y-y0)/(1-w)) + np.mean(y1-y0)
    return ate_aipw


def binary_kl(y, y_hat):
    return y*np.log(y) - y*np.log(y_hat) + (1-y)*np.log(1-y) - (1-y)*np.log(1-y_hat)


def binary_cross_entropy(y, y_hat):
    return -y*np.log(y_hat) - (1-y)*np.log(1-y_hat)


def compute_discrepancy(gp, gp_hat, x):
    t1 = gp.predict_proba(x)
    t2 = gp_hat.predict_proba(x)
    d = binary_cross_entropy(t1, t2)
    return d.mean()


class POREstimator:
    '''
    The estimator class for Potential Outcome Regression (POR) 
    We provide two options for POR:
    1. Single model POR: fit a single model for both t=0 and t=1
    2. Two model POR: fit two models for t=0 and t=1 separately
    The implementation supports both regression and classification (y is binary)
    We also provide three options for the base model:
    1. Linear 
    2. Gaussian process (allows uncertainty quantification)
    3. XGBoost
    '''
    def __init__(self, x, t, y, single_model=False, binary=False, normalized=True, model_name='linear', get_std=True) -> None:
        if normalized and not binary:
            # By default, only y is normalized
            # xm, xs = np.mean(x, axis=0, keepdims=True), np.std(x, axis=0, keepdims=True)
            xm, xs = 0, 1
            ym, ys = np.mean(y), np.std(y)
        else: 
            xm, xs, ym, ys = 0, 1, 0, 1 
        
        self.xm = xm 
        self.xs = xs
        self.ym = ym 
        self.ys = ys

        x_normalized = (x - self.xm) / self.xs
        y_normalized = (y - self.ym) / self.ys

        self.single_model = single_model
        self.model_name = model_name

        if model_name == 'gp':
            # k1 = 1.0 * RBF(1.0)
            k1 = 1.0 * DotProduct(1.0)
            alpha=0.5
            if binary:
                args = {'kernel':k1, "random_state":0}
                model_class = GaussianProcessClassifier
            else:
                args = {'kernel':k1, "alpha":alpha, "random_state":0}
                model_class = GaussianProcessRegressor

        elif model_name == 'linear': 
            # use linear model
            args = {}
            if binary:
                model_class = LogisticRegression
            else:
                model_class = LinearRegression

        elif model_name == 'xgboost':
            from xgboost import XGBRegressor
            args = {'n_estimators':100, 'max_depth':2, 'learning_rate':0.1}
            model_class = XGBRegressor
            if binary:
                args['objective'] = 'binary:logistic'
            else:
                args['objective'] = 'reg:squarederror'
        
        
        # The test set is the counterfactual, which is constructed when computing ATE
        if single_model:
            X_train = np.concatenate([x_normalized, t], axis=1) # X_train 
            y_train = y_normalized
            model = model_class(**args).fit(X_train, y_train)
            y0, y1 = get_y0_y1(x_normalized, t, model, binary)
            self.model = model
            
        else: 
            t = t.squeeze()
            X0 = x_normalized[t==0]
            X1 = x_normalized[t==1]
            y0_train = y_normalized[t==0]
            y1_train = y_normalized[t==1]

            m0 = model_class(**args).fit(X0, y0_train)
            m1 = model_class(**args).fit(X1, y1_train)
            if binary:
                y0 = m0.predict_proba(x_normalized)
                y1 = m1.predict_proba(x_normalized)
            else:
                y0 = m0.predict(x_normalized)
                y1 = m1.predict(x_normalized)

            self.m0 = m0
            self.m1 = m1
            

        y0, y1 = self.unnormalize_y(y0), self.unnormalize_y(y1)
        ate = np.mean(y1-y0)

        self.y0 = y0 
        self.y1 = y1 
        self.ate = ate
        if get_std:
            self.std = np.sqrt(compute_var_ate(y1, y0)) 

    def unnormalize_y(self, y):
        return y*self.ys + self.ym

    def compute_ate(self, x):
        if self.single_model:
            raise NotImplementedError 
        else: 
            
            x_normalized = (x - self.xm) / self.xs
            y0 = self.m0.predict(x_normalized) 
            y1 = self.m1.predict(x_normalized)

            y0, y1 = y0 * self.ys + self.ym, y1 * self.ys + self.ym 
            ate = np.mean(y1-y0)
        return ate

    def compute_std(self, x, t):
        X0 = x[t==0]
        X1 = x[t==1]
        y0 = self.m0.predict(X0)
        y1 = self.m1.predict(X1)
        std = np.sqrt(compute_var_ate(y1, y0))
        return std

    def __str__(self) -> str:
        return 'por'

        
class IPWEstimator:
    '''
    The estimator class for Inverse Probability Weighting (IPW)
    The implementation supports only classification (y is binary)
    We also provide two options for the base model:
    1. Linear 
    2. Gaussian process (allows uncertainty quantification)
    '''
    def __init__(self, x, t, y, normalized=True, model_name='linear', get_std=True) -> None:
        self.model_name = model_name

        if model_name == 'gp':
            k1 = 1.0 * RBF(1.0)
            k1 = 1.0 * DotProduct(1.0)
            args = {'kernel':k1, "random_state":0}
            model_class = GaussianProcessClassifier

        elif model_name == 'linear': 
            # use linear model
            args = {}
            model_class = LogisticRegression
                
        X_train = x
        y_train = t
        m = model_class(**args).fit(X_train, y_train)
        
        w = m.predict_proba(x)[:, 1] # compute propensity score
        w = self.weight_clipping(w)
        ate = np.mean(t*y/w - (1-t)*y/(1-w)) 
        y1 = t*y/w 
        y0 = (1-t)*y/(1-w)
        if normalized:
            ate = np.sum(y1)/np.sum(t/w) - np.sum(y0)/np.sum((1-t)/(1-w))
        else:
            ate = np.mean(y1 - y0)

        self.ate = ate
        self.w = w
        if get_std:
            self.std = get_std_bootstrap(x, t, y, IPWEstimator)

    def weight_clipping(self, w, val=0.01):
        w[w<=val] = val
        w[w>=1-val] = 1-val
        return w

    def __str__(self) -> str:
        return 'ipw'


class AIPWEstimator:
    '''
    The estimator class for Augmented Inverse Probability Weighting (AIPW)
    The implementation supports only classification (y is binary)
    We also provide two options for the base model:
    1. Linear 
    2. Gaussian process (allows uncertainty quantification)
    '''
    def __init__(self, x, t, y, single_model=False, binary=False, normalized=True, model_name='linear', get_std=True) -> None:
        self.model_name = model_name
        por = POREstimator(x, t, y, single_model, binary, normalized, model_name, get_std=False)
        ipw = IPWEstimator(x, t, y, normalized, model_name, get_std=False)

        w = ipw.w 
        y0 = por.y0
        y1 = por.y1
        ate = compute_ate_dr(t, w, y, y0, y1)
        self.ate = ate 
        if get_std:
            self.std = get_std_bootstrap(x, t, y, AIPWEstimator)

    def __str__(self) -> str:
        return 'aipw'