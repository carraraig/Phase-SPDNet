"""
=================================
 Functional Connectivity Helpers
=================================
This file contains helper functions for the functional connectivity example
"""

from pyriemann.estimation import Coherences
import numpy as np
# from pyriemann.utils.base import nearest_sym_pos_def
from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone
from sklearn.preprocessing import LabelEncoder
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import balanced_accuracy_score
from sklearn.ensemble import StackingClassifier

class Connectivities(TransformerMixin, BaseEstimator):
    """Getting connectivity features from epoch"""

    def __init__(self, method="ordinary", fmin=8, fmax=35, fs=None):
        self.method = method
        self.fmin = fmin
        self.fmax = fmax
        self.fs = fs
        self._coh = Coherences(
            coh=self.method,
            fmin=self.fmin,
            fmax=self.fmax,
            fs=self.fs,
        )

    def fit(self, X, y=None):
        self._coh = Coherences(
            coh=self.method,
            fmin=self.fmin,
            fmax=self.fmax,
            fs=self.fs,
        )
        return self

    def transform(self, X):
        X_coh = self._coh.fit_transform(X)
        X_con = np.mean(X_coh, axis=-1, keepdims=False)
        return X_con



class NearestSPD(TransformerMixin, BaseEstimator):
    """Transform square matrices to nearest SPD matrices"""

    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return nearest_sym_pos_def(X)

    def fit_transform(self, X, y=None):
        return self.transform(X)



def _nearest_sym_pos_def(S, reg=1e-6):
    """Find the nearest SPD matrix.
    Parameters
    ----------
    S : ndarray, shape (n, n)
        Square matrix.
    reg : float
        Regularization parameter.
    Returns
    -------
    P : ndarray, shape (n, n)
        Nearest SPD matrix.
    """
    A = (S + S.T) / 2
    _, s, V = np.linalg.svd(A)
    H = V.T @ np.diag(s) @ V
    B = (A + H) / 2
    P = (B + B.T) / 2

    if is_pos_def(P):
        # Regularize if already PD
        ei, ev = np.linalg.eigh(P)
        if np.min(ei) / np.max(ei) < reg:
            P = ev @ np.diag(ei + reg) @ ev.T
        return P

    spacing = np.spacing(np.linalg.norm(A))
    I = np.eye(S.shape[0])  # noqa
    k = 1
    while not is_pos_def(P, fast_mode=False):
        mineig = np.min(np.real(np.linalg.eigvals(P)))
        P += I * (-mineig * k ** 2 + spacing)
        k += 1

    # Regularize
    ei, ev = np.linalg.eigh(P)
    if np.min(ei) / np.max(ei) < reg:
        P = ev @ np.diag(ei + reg) @ ev.T
    return P


def nearest_sym_pos_def(X, reg=1e-6):
    """Find the nearest SPD matrices.
    A NumPy port of John D'Errico's `nearestSPD` MATLAB code [1]_,
    which credits [2]_.
    Parameters
    ----------
    X : ndarray, shape (..., n, n)
        Square matrices, at least 2D ndarray.
    reg : float
        Regularization parameter.
    Returns
    -------
    P : ndarray, shape (..., n, n)
        Nearest SPD matrices.
    Notes
    -----
    .. versionadded:: 0.3.1
    References
    ----------
    .. [1] `nearestSPD
        <https://www.mathworks.com/matlabcentral/fileexchange/42885-nearestspd>`_
        J. D'Errico, MATLAB Central File Exchange
    .. [2] `Computing a nearest symmetric positive semidefinite matrix
        <https://www.sciencedirect.com/science/article/pii/0024379588902236>`_
        N.J. Higham, Linear Algebra and its Applications, vol 103, 1988
    """
    return np.array([_nearest_sym_pos_def(x, reg) for x in X])


def is_pos_def(X, fast_mode=False):
    """ Check if all matrices are positive definite.
    Check if all matrices are positive definite, fast verification is done
    with Cholesky decomposition, while full check compute all eigenvalues
    to verify that they are positive.
    Parameters
    ----------
    X : ndarray, shape (..., n, n)
        The set of square matrices, at least 2D ndarray.
    fast_mode : boolean, default=False
        Use Cholesky decomposition to avoid computing all eigenvalues.
    Returns
    -------
    ret : boolean
        True if all matrices are positive definite.
    """
    if fast_mode:
        try:
            np.linalg.cholesky(X)
            return True
        except np.linalg.LinAlgError:
            return False
    else:
        return is_square(X) and np.all(_get_eigenvals(X) > 0.0)



def is_square(X):
    """ Check if matrices are square.
    Parameters
    ----------
    X : ndarray, shape (..., n, n)
        The set of square matrices, at least 2D ndarray.
    Returns
    -------
    ret : boolean
        True if matrices are square.
    """
    return X.ndim >= 2 and X.shape[-2] == X.shape[-1]


def _get_eigenvals(X):
    """ Private function to compute eigen values. """
    n = X.shape[-1]
    return np.linalg.eigvals(X.reshape((-1, n, n)))