"""Helper functions and data munging."""

import pickle
import numpy as np
import pandas as pd
import sys
import logging
from copy import copy

logger = logging.getLogger(__name__)

def assert_columns(db, columns):
    if not is_column(db, columns):
        logger.error('not all columns {} found'.format(columns))
        sys.exit(1)


def assert_square(X):
    if len(X.shape) != 2 or X.shape[0] != X.shape[1]:
        raise Exception("Input matrix must be square")


def is_probability(x):
    """True if x is a float between 0 and 1, otherwise false."""
    if x >= 0 and x <= 1:
        return True
    return False


def is_column(df, column):
    """
    True if column is in pandas dataframe df. If column is a list, checks all of
    them.
    """
    if type(column) == str:
        if column in df.columns:
            return True
        return False

    elif type(column) == list:
        for c in column:
            if not is_column(df, c):
                return False
        return True


def is_even(n):
    """True if n is even, else false."""
    if n % 2 == 0:
        return True
    return False


def split_columns(variable):
    """
    Splits the input variable, which is either None or a comma delimited string.
    """
    if variable:
        return(variable.split(','))
    return(variable)


def clean(X):
    """
    Replaces nan and inf values in numpy array with zero. If any columns are all
    0, removes them completely.
    """
    X[np.isnan(X)] = 0
    X[np.isinf(X)] = 0
    logger.debug('X matrix has {} bad values (replaced with 0)'.format(np.sum(X == 0)))

    idx_zero = np.where(np.sum(np.abs(X), axis=0) == 0)[0] # find all zero cols

    if len(idx_zero) > 0:
        logger.debug('removing {} columns in X that are all 0'.format(len(idx_zero)))
        idx = np.arange(X.shape[1])
        idx = np.setdiff1d(idx, idx_zeros)
        X = X[:, idx]

    return(X)


def reorder(X, idx, symm=False):
    """
    Reorders the rows of a matrix. If symm is True, this simultaneously reorders
    the columns and rows of a matrix by the given index.
    """
    if symm:
        assert_square(X)
        X = X[:, idx]

    if X.shape[0] != len(idx):
        logger.warn('reorg IDX length {} does not match the rows of X {}'.format(len(idx), X.shape[0]))

    X = X[idx, :]

    return(X)


def full_rank(X):
    """Ensures input matrix X is not rank deficient."""
    if len(X.shape) == 1:
        return True

    k = X.shape[1]
    rank = np.linalg.matrix_rank(X)
    if rank < k:
        return False

    return True


def scrub_data(x):
    """
    Removes NaNs from a vector, and raises an exception if the data is empty.
    """
    x[np.isnan(x)] = 0
    if np.sum(x) == 0:
        raise Exception('vector contains no information')

    return x


def gather_dv(db, columns):
    """
    Returns a numpy vector of the predicted column. Cutoff is a percentage
    (0 < p < 0.5). If cutoff specified, returns a binary vector (0 = lower than
    cutoff, 1 = higher than cutoff). The maximum cutoff is 50%, or the median
    of the sample.
    """
    for i, col in enumerate(columns):
        tmp = np.array(db[col])
        if i == 0:
            y = tmp
        else:
            y = np.vstack((y, tmp))

    return(y)


def make_dv_groups(y, cutoff):
    """
    Accepts a numpy vector of the dependent variable y. All scores lower than
    the submitted percentile cutoff are set to 0, and the rest are set to 1.
    Used to turn continuous variables into groups for outlier detection.
    """
    logger.info('partitioning y at the {}th percentile'.format(cutoff*100))
    cutoff = np.percentile(y, cutoff*100)
    idx_lo = np.where(y < cutoff)[0]
    idx_hi = np.where(y >= cutoff)[0]
    y[idx_lo] = 0
    y[idx_hi] = 1

    return y


def pickle_it(my_data, save_path):
    f = open(save_path, 'wb')
    pickle.dump(my_data, f)
    f.close()


