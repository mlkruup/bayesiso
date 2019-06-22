from __future__ import division
#import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils import indexable, column_or_1d
from scipy.optimize import minimize_scalar
from sklearn.linear_model import LogisticRegression


import warnings

from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin, clone
from sklearn.utils import check_X_y, check_array, indexable, column_or_1d
from math import log
from scipy.optimize import fmin_bfgs
from sklearn.preprocessing import LabelEncoder
import autograd.numpy as np
from autograd import grad


def _my_logistic_regression_2d(df, y):
    y = column_or_1d(y)

    F = df
    tiny = np.finfo(np.float).tiny

    #prior0 = float(np.sum(y <= 0))
    prior0 = float(np.sum(y <= 0.5))

    prior1 = y.shape[0] - prior0

    T = y
    T1 = 1. - T

    def objective(AB):
        E = np.exp(AB[0] * F[:, 0] + AB[1] * F[:, 1] + AB[2])
        P = 1. / (1. + E)
        l = -(T * np.log(P + tiny) + T1 * np.log(1. - P + tiny))
        return l.sum()

    AB0 = np.array([0., 0., log((prior0 + 1.) / (prior1 + 1.))])  # ei tea kas on õiged mida valida
    AB_ = fmin_bfgs(objective, AB0, fprime=grad(objective), disp=False)
    return [[AB_[0], AB_[1]]], [AB_[2]]


def _my_logistic_regression_1d(df, y):
    df = column_or_1d(df)
    y = column_or_1d(y)

    F = df
    tiny = np.finfo(np.float).tiny

    #prior0 = float(np.sum(y <= 0))
    prior0 = float(np.sum(y <= 0.5))

    prior1 = y.shape[0] - prior0

    T = y
    T1 = 1. - T

    def objective(AB):
        E = np.exp(AB[0] * F + AB[1])
        P = 1. / (1. + E)
        l = -(T * np.log(P + tiny) + T1 * np.log(1. - P + tiny))
        return l.sum()

    AB0 = np.array([0., log((prior0 + 1.) / (prior1 + 1.))])  # ei tea kas on õiged mida valida
    AB_ = fmin_bfgs(objective, AB0, fprime=grad(objective), disp=False)
    return [[AB_[0]]], [AB_[1]]


class _MyLogisticRegression:

    def __init__(self, x_dim):
        self.x_dim = x_dim

    def fit(self, X, y):
        if self.x_dim == 1:
            self.coef_, self.intercept_ = _my_logistic_regression_1d(X, y)
        elif self.x_dim == 2:
            self.coef_, self.intercept_ = _my_logistic_regression_2d(X, y)
        return self

    def predict_proba(self, T):
        if self.x_dim == 1:
            T = column_or_1d(T)
            return 1. / (1. + np.exp(self.coef_[0][0] * T + self.intercept_[0]))
        elif self.x_dim == 2:
            return 1. / (1. + np.exp(self.coef_[0][0] * T[:, 0] + self.coef_[0][1] * T[:, 1] + self.intercept_[0]))

def _beta_calibration(df, y, sample_weight=None, sklearn_lr = True):
    warnings.filterwarnings("ignore")

    df = column_or_1d(df).reshape(-1, 1)
    df = np.clip(df, 1e-16, 1-1e-16)
    y = column_or_1d(y)

    x = np.hstack((df, 1. - df))
    x = np.log(x)
    x[:, 1] *= -1

    if sklearn_lr:
        lr = LogisticRegression(C=99999999999)
    else:
        lr = _MyLogisticRegression(x_dim = 2)
    lr.fit(x, y)
    coefs = lr.coef_[0]
    if not sklearn_lr:
        coefs = [-1 * _c for _c in coefs]
    #print(coefs)


    if coefs[0] < 0:
        x = x[:, 1].reshape(-1, 1)
        if sklearn_lr:
            lr = LogisticRegression(C=99999999999)
        else:
            lr = _MyLogisticRegression(x_dim = 1)
        lr.fit(x, y)
        coefs = lr.coef_[0]
        if not sklearn_lr:
            coefs = [-1 * _c for _c in coefs]
        a = 0
        b = coefs[0]
    elif coefs[1] < 0:
        x = x[:, 0].reshape(-1, 1)
        if sklearn_lr:
            lr = LogisticRegression(C=99999999999)
        else:
            lr = _MyLogisticRegression(x_dim = 1)
        lr.fit(x, y)
        coefs = lr.coef_[0]
        if not sklearn_lr:
            coefs = [-1 * _c for _c in coefs]
        a = coefs[0]
        b = 0
    else:
        a = coefs[0]
        b = coefs[1]
    inter = lr.intercept_[0]
    if not sklearn_lr:
        inter = inter * -1

    #print(coefs, inter)

    m = minimize_scalar(lambda mh: np.abs(b*np.log(1.-mh)-a*np.log(mh)-inter),
                        bounds=[0, 1], method='Bounded').x
    #m2 = minimize_scalar(lambda mh: np.abs(b*np.log(1.-mh)-a*np.log(mh)+inter),
    #                    bounds=[0, 1], method='Bounded').x
    #m3 = minimize_scalar(lambda mh: (b*np.log(1.-mh)-a*np.log(mh)-inter)**2,
    #                    bounds=[0, 1], method='Bounded').x
    #m4 = minimize_scalar(lambda mh: (b*np.log(1.-mh)-a*np.log(mh)+inter)**2,
    #                    bounds=[0, 1], method='Bounded').x

    #print(m, m2, m3, m4)


    map = [a, b, m]
    return map, lr


class _BetaCal(BaseEstimator, RegressorMixin):
    """Beta regression model with three parameters introduced in
    Kull, M., Silva Filho, T.M. and Flach, P. Beta calibration: a well-founded
    and easily implemented improvement on logistic calibration for binary
    classifiers. AISTATS 2017.

    Attributes
    ----------
    map_ : array-like, shape (3,)
        Array containing the coefficients of the model (a and b) and the
        midpoint m. Takes the form map_ = [a, b, m]

    lr_ : sklearn.linear_model.LogisticRegression
        Internal logistic regression used to train the model.
    """
    def fit(self, X, y, sklearn_lr = True, sample_weight=None):
        """Fit the model using X, y as training data.

        Parameters
        ----------
        X : array-like, shape (n_samples,)
            Training data.

        y : array-like, shape (n_samples,)
            Training target.

        sample_weight : array-like, shape = [n_samples] or None
            Sample weights. If None, then samples are equally weighted.

        Returns
        -------
        self : object
            Returns an instance of self.
        """
        X = column_or_1d(X)
        y = column_or_1d(y)
        X, y = indexable(X, y)

        self.map_, self.lr_ = _beta_calibration(X, y, sample_weight, sklearn_lr = sklearn_lr)
        #print(self.map_)

        return self

    def predict(self, S):
        """Predict new values.

        Parameters
        ----------
        S : array-like, shape (n_samples,)
            Data to predict from.

        Returns
        -------
        S_ : array, shape (n_samples,)
            The predicted values.
        """
        df = column_or_1d(S).reshape(-1, 1)
        df = np.clip(df, 1e-16, 1-1e-16)

        x = np.hstack((df, 1. - df))
        x = np.log(x)
        x[:, 1] *= -1
        if self.map_[0] == 0:
            x = x[:, 1].reshape(-1, 1)
        elif self.map_[1] == 0:
            x = x[:, 0].reshape(-1, 1)

        pred = self.lr_.predict_proba(x)
        if len(pred.shape) == 1:
            return self.lr_.predict_proba(x) #[:, 1] - commented out
        elif pred.shape[1] == 2:
            return self.lr_.predict_proba(x)[:,1]


def _beta_am_calibration(df, y, sample_weight=None, sklearn_lr = True):
    warnings.filterwarnings("ignore")

    df = column_or_1d(df).reshape(-1, 1)
    df = np.clip(df, 1e-16, 1-1e-16)
    y = column_or_1d(y)

    x = np.log(df / (1. - df))

    if sklearn_lr:
        lr = LogisticRegression(C=99999999999)
    else:
        lr = _MyLogisticRegression(x_dim = 1)
    lr.fit(x, y)
    coefs = lr.coef_[0]
    inter = lr.intercept_[0]
    a = coefs[0]
    b = a
    m = 1.0 / (1.0 + np.exp(inter / a))
    map = [a, b, m]
    return map, lr


class _BetaAMCal(BaseEstimator, RegressorMixin):
    """Beta regression model with two parameters (a and m, fixing a = b)
    introduced in Kull, M., Silva Filho, T.M. and Flach, P. Beta calibration:
    a well-founded and easily implemented improvement on logistic calibration
    for binary classifiers. AISTATS 2017.

    Attributes
    ----------
    map_ : array-like, shape (3,)
        Array containing the coefficients of the model (a and b) and the
        midpoint m. Takes the form map_ = [a, b, m], where a = b

    lr_ : sklearn.linear_model.LogisticRegression
        Internal logistic regression used to train the model.
    """
    def fit(self, X, y, sklearn_lr = True, sample_weight=None):
        """Fit the model using X, y as training data.

        Parameters
        ----------
        X : array-like, shape (n_samples,)
            Training data.

        y : array-like, shape (n_samples,)
            Training target.

        sample_weight : array-like, shape = [n_samples] or None
            Sample weights. If None, then samples are equally weighted.

        Returns
        -------
        self : object
            Returns an instance of self.
        """
        X = column_or_1d(X)
        y = column_or_1d(y)
        X, y = indexable(X, y)

        self.map_, self.lr_ = _beta_am_calibration(X, y, sample_weight, sklearn_lr = sklearn_lr)

        return self

    def predict(self, S):
        """Predict new values.

        Parameters
        ----------
        S : array-like, shape (n_samples,)
            Data to predict from.

        Returns
        -------
        S_ : array, shape (n_samples,)
            The predicted values.
        """
        df = column_or_1d(S).reshape(-1, 1)
        df = np.clip(df, 1e-16, 1-1e-16)

        x = np.log(df / (1. - df))
        pred = self.lr_.predict_proba(x)
        if len(pred.shape) == 1:
            return self.lr_.predict_proba(x) #[:, 1] - commented out
        elif pred.shape[1] == 2:
            return self.lr_.predict_proba(x)[:,1]


def _beta_ab_calibration(df, y, sample_weight=None, sklearn_lr = True):
    warnings.filterwarnings("ignore")

    df = column_or_1d(df).reshape(-1, 1)
    df = np.clip(df, 1e-16, 1-1e-16)
    y = column_or_1d(y)

    x = np.hstack((df, 1. - df))
    x = np.log(2 * x)

    lr = LogisticRegression(fit_intercept=False, C=99999999999) # not implemented
    lr.fit(x, y)
    coefs = lr.coef_[0]
    a = coefs[0]
    b = -coefs[1]
    m = 0.5
    map = [a, b, m]
    return map, lr


class _BetaABCal(BaseEstimator, RegressorMixin):
    """Beta regression model with two parameters (a and b, fixing m = 0.5)
    introduced in Kull, M., Silva Filho, T.M. and Flach, P. Beta calibration:
    a well-founded and easily implemented improvement on logistic calibration
    for binary classifiers. AISTATS 2017.

    Attributes
    ----------
    map_ : array-like, shape (3,)
        Array containing the coefficients of the model (a and b) and the
        midpoint m. Takes the form map_ = [a, b, m], where m = 0.5

    lr_ : sklearn.linear_model.LogisticRegression
        Internal logistic regression used to train the model.
    """
    def fit(self, X, y, sklearn_lr = True, sample_weight=None):
        """Fit the model using X, y as training data.

        Parameters
        ----------
        X : array-like, shape (n_samples,)
            Training data.

        y : array-like, shape (n_samples,)
            Training target.

        sample_weight : array-like, shape = [n_samples] or None
            Sample weights. If None, then samples are equally weighted.

        Returns
        -------
        self : object
            Returns an instance of self.
        """
        X = column_or_1d(X)
        y = column_or_1d(y)
        X, y = indexable(X, y)

        self.map_, self.lr_ = _beta_ab_calibration(X, y, sample_weight, sklearn_lr = sklearn_lr)

        return self

    def predict(self, S):
        """Predict new values.

        Parameters
        ----------
        S : array-like, shape (n_samples,)
            Data to predict from.

        Returns
        -------
        S_ : array, shape (n_samples,)
            The predicted values.
        """
        df = column_or_1d(S).reshape(-1, 1)
        df = np.clip(df, 1e-16, 1-1e-16)

        x = np.hstack((df, 1. - df))
        x = np.log(2 * x)
        pred = self.lr_.predict_proba(x)
        if len(pred.shape) == 1:
            return self.lr_.predict_proba(x) #[:, 1] - commented out
        elif pred.shape[1] == 2:
            return self.lr_.predict_proba(x)[:,1]
