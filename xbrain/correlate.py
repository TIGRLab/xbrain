"""
Functions for generating intersubject correlation features.
"""
import os, sys
import logging
import numpy as np
import pandas as pd
from scipy.stats.stats import pearsonr
import matplotlib
matplotlib.use('Agg')   # Force matplotlib to not use any Xwindows backend
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

def read_timeseries(db, row, col):
    """
    Return numpy array of the timeseries defined in the ith row, named column
    of input database db.
    """
    timeseries_file = db.iloc[row][col]
    try:
        return(np.genfromtxt(timeseries_file, delimiter=','))
    except:
        raise IOError('failed to parse timeseries {}'.format(timeseries_file))


def pct_signal_change(ts):
    """Converts each timeseries (column of matrix ts) to % signal change."""
    means = np.tile(np.mean(ts, axis=0), [ts.shape[0], 1])
    return(((ts-means)/means) * 100)


def zscore(ts):
    """Converts each timeseries to have 0 mean and unit variance."""
    means = np.tile(np.mean(ts, axis=0), [ts.shape[0], 1])
    stdev = np.tile(np.std(ts, axis=0), [ts.shape[0], 1])
    return((ts-means)/stdev)

def calc_xbrain(template_db, db, timeseries):
    """
    Calculates correlation of each participant in db with mean time series of
    everyone in the template db, excluding the participant's entry in the
    template if it exists. The features are concatenated for each participant
    and returned as the feature matrix X.
    """
    template_idx = template_db.index
    db_idx = db.index
    n = len(db)

    for i, column in enumerate(timeseries):

        # get a timepoint X roi X subject matrix from the template
        template_ts = get_column_ts(template_db, column)


        # loop through subjects
        for j, subj in enumerate(db_idx):
            if j == 0:
                # for the first timeseries, initialize the output array
                xcorrs = np.zeros((n, template_ts.shape[1]))

            try:
                ts = read_timeseries(db, j, column)
            except IOError as e:
                logger.error(e)
                sys.exit(1)

            ts = zscore(ts)
            n_roi = ts.shape[1]

            # take the mean of the template, excluding this sample if shared
            unique_idx = template_idx != subj
            template_mean = np.mean(template_ts[:, :, unique_idx], axis=2)

            # diag of the intersubject corrs (upper right corner of matrix),
            # this includes only the correlations between homologous regions
            try:
                rs = np.diag(np.corrcoef(ts.T, y=template_mean.T)[n_roi:, :n_roi])
            except:
                raise Exception('xcorr dimension missmatch: subject {} dims={}, timeseries={}, template dims={}'.format(j, ts.shape, column, template_mean.shape))

            xcorrs[j, :] = rs

        # horizontally concatenate xcorrs into X (samples X features)
        if i == 0:
            X = xcorrs
        else:
            X = np.hstack((X, xcorrs))

    logger.debug('xbrain feature matrix shape: {}'.format(X.shape))

    return X


def get_column_ts(df, column):
    """
    Accepts a dataframe, and a timeseries column, reads the timeseries
    of all subjects, and returns a timepoint X roi X subject numpy array.
    """
    if type(column) == list:
        raise TypeError('column {} should be a valid pandas column identifier'.format(column))

    dims = read_timeseries(df, 0, column).shape
    n = len(df)
    template_ts = np.zeros((dims[0], dims[1], n))

    # collect timeseries
    for i in range(n):
        ts = read_timeseries(df, i, column)
        ts = pct_signal_change(ts)
        template_ts[:, :, i] = ts

    return template_ts


def find_template(db, y, timeseries, group=-1):
    """
    Copies a subset of the input database into a template database.

    If group is defined (greater than -1), all of the subjects are used from
    that group to construct the template. Otherwise, all subjects are used in
    db.
    """
    if group > -1:
        return(db.loc[db[y] == group])
    else:
        return(db)


def plot_X(X, path, title='features', X2=None):
    """
    Plots the cross brain correlation features calculated. Can be used to
    compare features (e.g., hi vs low template, or train vs test matricies) if
    X2 is defined. Negative correlations are nonsense for cross brain
    correlations, and are set to 0 for visualization.
    """
    if not os.path.isdir(path):
        raise Exception('path {} is not a directory'.format(path))

    if X2 is not None:
        X = np.vstack((np.vstack((X, np.ones(X.shape[1]))), X2))

    plt.imshow(X, vmin=-0.5, vmax=0.5, cmap=plt.cm.RdBu_r, interpolation='nearest')
    plt.colorbar()

    if X2 is not None:
        plt.title('X (Reds) vs X2 (Blues)')
    else:
        plt.title('X (Reds)')

    plt.savefig(os.path.join(path, 'xbrain_X_{}.pdf'.format(title)))
    plt.close()

