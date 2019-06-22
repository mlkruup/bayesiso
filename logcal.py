from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin, clone
from sklearn.utils import check_X_y, check_array, indexable, column_or_1d
from math import log
from scipy.optimize import fmin_bfgs
from sklearn.preprocessing import LabelEncoder
import numpy as np


def _my_sigmoid_calibration(df, y, sample_weight=None):
    """Probability Calibration with sigmoid method (Platt 2000)
    Parameters
    ----------
    df : ndarray, shape (n_samples,)
        The decision function or predict proba for the samples.
    y : ndarray, shape (n_samples,)
        The targets.
    sample_weight : array-like, shape = [n_samples] or None
        Sample weights. If None, then samples are equally weighted.
    Returns
    -------
    a : float
        The slope.
    b : float
        The intercept.
    References
    ----------
    Platt, "Probabilistic Outputs for Support Vector Machines"
    """
    df = column_or_1d(df)
    y = column_or_1d(y)

    F = df  # F follows Platt's notations
    tiny = np.finfo(np.float).tiny  # to avoid division by 0 warning

    # Bayesian priors (see Platt end of section 2.2)
    prior0 = float(np.sum(y <= 0))
    prior1 = y.shape[0] - prior0

    #if platt_correction:
    #    T = np.zeros(y.shape)
    #    T[y > 0] = (prior1 + 1.) / (prior1 + 2.)
    #    T[y <= 0] = 1. / (prior0 + 2.)
    #else:

    T = y
    T1 = 1. - T

    def objective(AB):
        # From Platt (beginning of Section 2.2)
        E = np.exp(AB[0] * F + AB[1])
        P = 1. / (1. + E)
        l = -(T * np.log(P + tiny) + T1 * np.log(1. - P + tiny))
        if sample_weight is not None:
            return (sample_weight * l).sum()
        else:
            return l.sum()

    def grad(AB):
        # gradient of the objective function
        E = np.exp(AB[0] * F + AB[1])
        P = 1. / (1. + E)
        TEP_minus_T1P = P * (T * E - T1)
        if sample_weight is not None:
            TEP_minus_T1P *= sample_weight
        dA = np.dot(TEP_minus_T1P, F)
        dB = np.sum(TEP_minus_T1P)
        return np.array([dA, dB])

    AB0 = np.array([0., log((prior0 + 1.) / (prior1 + 1.))])
    AB_ = fmin_bfgs(objective, AB0, fprime=grad, disp=False)
    return AB_[0], AB_[1]


class _MySigmoidCalibration(BaseEstimator, RegressorMixin):
    """Sigmoid regression model.
    Attributes
    ----------
    a_ : float
        The slope.
    b_ : float
        The intercept.
    """

    def fit(self, X, y, sample_weight=None):
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

        self.a_, self.b_ = _my_sigmoid_calibration(X, y)
        return self

    def predict(self, T):
        """Predict new data by linear interpolation.
        Parameters
        ----------
        T : array-like, shape (n_samples,)
            Data to predict from.
        Returns
        -------
        T_ : array, shape (n_samples,)
            The predicted data.
        """
        T = column_or_1d(T)
        return 1. / (1. + np.exp(self.a_ * T + self.b_))