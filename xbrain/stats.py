#!/usr/bin/env python
"""Routines for relating neural activity with clinical variables."""

import os, sys, glob, copy
import collections
import logging
import random
import string

import numpy as np
import scipy as sp
import scipy.cluster.hierarchy as sch
from scipy.spatial.distance import pdist, squareform
from scipy import stats
from scipy import linalg
from scipy.stats import mode
import pandas as pd
from sklearn import preprocessing

from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, f1_score, roc_auc_score

import matplotlib
matplotlib.use('Agg')   # Force matplotlib to not use any Xwindows backend
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns

import xbrain.utils as utils

logger = logging.getLogger(__name__)

def r_to_z(R):
    """Fischer's r-to-z transform on a matrix (elementwise)."""
    return(0.5 * np.log((1+R)/(1-R)))


def r_to_d(R):
    """Converts a correlation matrix R to a distance matrix D."""
    return(np.sqrt(2*(1-R)))


def standardize(X):
    """z-scores each column of X."""
    return((X - X.mean(axis=0)) / X.std(axis=0))


def sig_cutoffs(null, two_sided=True):
    """Returns the significance cutoffs of the submitted null distribution."""
    if two_sided:
        sig = np.array([np.percentile(null, 2.5), np.percentile(null, 97.5)])
    else:
        sig = np.array([np.percentile(null, 5), np.percentile(null, 95)])

    return(sig)


def gowers_matrix(D):
    """Calculates Gower's centered matrix from a distance matrix."""
    utils.assert_square(D)

    n = D.shape[0]
    o = np.ones((n, 1))
    I = np.identity(n) - (1/float(n))*o.dot(o.T)
    A = -0.5*(np.square(D))
    G = I.dot(A).dot(I)

    return(G)


def hat_matrix(X):
    """
    Caluclates distance-based hat matrix for an NxM matrix of M predictors from
    N variables. Adds the intercept term for you.
    """
    X = np.hstack((np.ones((X.shape[0], 1)), X)) # add intercept
    Q1, R1 = np.linalg.qr(X)
    H = Q1.dot(Q1.T)

    return(H)


def calc_F(H, G, m=None):
    """
    Calculate the F statistic when comparing two matricies.
    """
    utils.assert_square(H)
    utils.assert_square(G)

    n = H.shape[0]
    I = np.identity(n)
    IG = I-G

    if m:
        F = (np.trace(H.dot(G).dot(H)) / (m-1)) / (np.trace(IG.dot(G).dot(IG)) / (n-m))
    else:
        F = (np.trace(H.dot(G).dot(H))) / np.trace(IG.dot(G).dot(IG))

    return F


def permute(H, G, n=10000):
    """
    Calculates a null F distribution from a symmetrically-permuted G (Gower's
    matrix), from the between subject connectivity distance matrix D, and a the
    H (hat matrix), from the original behavioural measure matrix X.

    The permutation test is accomplished by simultaneously permuting the rows
    and columns of G and recalculating F. We do not need to account for degrees
    of freedom when calculating F.
    """
    F_null = np.zeros(n)
    idx = np.arange(G.shape[0]) # generate our starting indicies

    for i in range(n):
        idx = np.random.permutation(idx)
        G_perm = utils.reorder(G, idx, symm=True)
        F_null[i] = calc_F(H, G_perm)

    F_null.sort()

    return F_null


def variance_explained(H, G):
    """
    Calculates variance explained in the distance matrix by the M predictor
    variables in X.
    """
    utils.assert_square(H)
    utils.assert_square(G)

    return((np.trace(H.dot(G).dot(H))) / np.trace(G))


def mdmr(X, Y, method='corr'):
    """
    Multvariate regression analysis of distance matricies: regresses variables
    of interest X (behavioural) onto a matrix representing the similarity of
    connectivity profiles Y.

    Zapala & Schork, 2006. Multivariate regression analysis of distance matrices
    for testing association between gene expression patterns related variables.
    PNAS 103(51)
    """
    if not utils.full_rank(X):
        raise Exception('X is not full rank:\ndimensions = {}'.format(X.shape))

    X = standardize(X)   # mean center and Z-score all cognitive variables

    if method == 'corr':
        R = np.corrcoef(Y)   # correlation distance between each cross-brain correlation vector
        D = r_to_d(R)        # distance matrix of correlation matrix
    elif method == 'euclidean':
        D = squareform(pdist(Y, 'euclidean'))

    G = gowers_matrix(D) # centered distance matrix (connectivity similarities)
    H = hat_matrix(X)    # hat matrix of regressors (cognitive variables)
    F = calc_F(H, G)     # F test of relationship between regressors and distance matrix
    F_null = permute(H, G)
    v = variance_explained(H, G)

    return F, F_null, v


def backwards_selection(X, Y):
    """
    Performs backwards variable selection on the input data.
    """

    return False


def individual_importances(X, Y):
    """
    Runs MDMR individually for each variable. If the variable is deemed
    significant, the variance explained is recorded, otherwise it is reported
    as 0. Returns a vector of variance explained.
    """
    m = X.shape[1]
    V = np.zeros(m)
    for test in range(m):
        X_test = np.atleast_2d(X[:, test]).T # enforces a column vector
        F, F_null, v = mdmr(X_test, Y)
        thresholds = sig_cutoffs(F_null, two_sided=False)
        if F > thresholds[1]:
            V[test] = v
        else:
            V[test] = 0
        print('tested variable {}/{}'.format(test+1, m))

    return V


def pca_plot(X, y, plot):
    """
    Takes the top 3 PCs from the data matrix X, and plots them. y is used to
    color code the data. No obvious grouping or clustering should be found. The
    presence of such grouping suggests a strong site, sex, or similar effect.
    """
    clf = PCA(n_components=3)
    clf = clf.fit(X)
    X = clf.transform(X)
    fig = plt.figure(1, figsize=(4, 3))
    ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)
    ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=y, cmap=plt.cm.RdBu_r)
    fig.savefig(os.path.join(plot, 'xbrain_X_PCs.pdf'))
    plt.close()


def pca_reduce(X, n=1, pct=1, X2=False):
    """
    Uses PCA to reduce the number of features in the input matrix X. n is
    the target number of features in X to retain after reduction. pct is the
    target amount of variance (%) in the original matrix X to retain in the
    reduced feature matrix. When n and pct disagree, compresses the feature
    matrix to the smaller number of features. If X2 is defined, the same
    transform learned from X is applied to X2.
    """
    if not utils.is_probability(pct):
        raise Exception('pct should be a probability (0-1), is {}'.format(pct))

    clf = PCA()
    clf = clf.fit(X)
    cumulative_var = np.cumsum(clf.explained_variance_ratio_)

    # calculate variance retained in the n components case
    pct_n_case = cumulative_var[n-1] # correct for zero indexing

    # calculate # of components that would be retained in the pct case
    if pct < 1:
        n_comp_pct = np.where(cumulative_var >= pct)[0][0]
    else:
        n_comp_pct = len(cumulative_var)-1

    # case where we retain pre-defined % of the variance
    if n_comp_pct < n:
        cutoff = n_comp_pct + 1 # correct for zero indexing
    # case where we use pre-defined number of components
    else:
        pct = pct_n_case
        cutoff = n

    logger.info('X {} reduced to {} components, retaining {} % of variance'.format(X.shape, cutoff, pct))

    # reduce X to the defined number of components
    clf = PCA(n_components=cutoff)
    clf.fit(X)
    X_transformed = clf.transform(X)

    # sign flip potentially applied if we only retain 1 component
    if cutoff == 1:
        X_transformed = sign_flip(X_transformed, X)

    # if X2 is defined, apply the transform learnt from X to X2 as well
    if np.any(X2):
        X2_transformed = clf.transform(X2)

        # sign flip potentially applied if we only retain 1 component
        if cutoff == 1:
            X2_transformed = sign_flip(X2_transformed, X)

        logger.debug('PCA transform learned on X applied to X2')
        return(X_transformed, X2_transformed)

    return(X_transformed)


def sign_flip(X_transformed, X):
    """
    X_transformed a 1D vector representing the top PC from X. This applies a
    sign flip to X_transformed if X_transformed is anti-correlated with the mean
    of X. This is important particularly for compressing the y variables, where
    we want to retain high (good) and low (scores), and flipping these would
    change our intrepretation of the statistics.
    """
    X_transformed = X_transformed.flatten()
    corr = np.corrcoef(np.vstack((X_transformed, np.mean(X, axis=1))))[0,1]
    if corr < 0:
        X_transformed = X_transformed * -1

    return(X_transformed)


def make_classes(y):
    """transforms label values for classification"""
    le = preprocessing.LabelEncoder()
    le.fit(y)
    logger.debug('unique transformed y groups: {}'.format(np.unique(le.transform(y))))
    return(le.transform(y))


def classify(X_train, X_test, y_train, y_test, model='SVC'):
    """
    Trains the selected classifier once on the submitted training data, and
    compares the predicted outputs of the test data with the real labels.
    Includes a hyper-parameter cross validation loop, the 'innermost' loop.
    Returns a set of metrics collected from the hyperparameter grid search and
    the test error.
    """
    if X_train.shape[0] != y_train.shape[0]:
        raise Exception('X_train shape {} does not equal y_train shape {}'.format(X_train.shape[0], y_train.shape[0]))
    if X_test.shape[0] != y_test.shape[0]:
        raise Exception('X_test shape {} does not equal y_test shape {}'.format(X_test.shape[0], y_test.shape[0]))

    n_features = X_train.shape[1]

    hp_dict = collections.defaultdict(list)
    r_train, r_test, R2_train, R2_test, MSE_train, MSE_test = [], [], [], [], [], []

    # for testing various models, includes grid search settings
    if model == 'Logistic':
        model_clf = LogisticRegression()
        hyperparams = {'C': [0.2, 0.6, 0.8, 1, 1.2] }
        scale_data = True
        feat_imp = False
    elif model == 'SVC':
        model_clf = SVC()
        hyperparams = {'kernel':['linear','rbf'],
                       'class_weight': ['balanced'],
                       'C': [2**-5, 2**-3, 2**-1, 2**1, 2**3, 2**5, 2**7, 2**9, 2**11, 2**13, 2**15],
                       'gamma': [2**-5, 2**-3, 2**-1, 2**1, 2**3, 2**5]}
        scale_data = True
        feat_imp = True
    elif model == 'RFC':
        model_clf = RandomForestClassifier(n_jobs=6)
        hyperparams =  {'class_weight': ['balanced'],
                        'n_estimators': [1000],
                        'min_samples_split': [int(n_features*0.025), int(n_features*0.05), int(n_features*0.075), int(n_features*0.1)],
                        'min_samples_leaf': [int(n_features*0.025), int(n_features*0.05), int(n_features*0.075), int(n_features*0.1)],
                        'criterion': ['gini', 'entropy']}
        scale_data = False
        feat_imp = True
    else:
        logger.error('invalid model type {}'.format(model))
        sys.exit(1)

    if scale_data:
        X_train = preprocessing.scale(X_train)
        X_test = preprocessing.scale(X_test)

    # perform randomized hyperparameter search to find optimal settings
    logger.debug('Inner Loop: CV of hyperparameters for this fold')
    clf = GridSearchCV(model_clf, hyperparams)
    clf.fit(X_train, y_train)

    # collect all the best hyperparameters found in the cv loop
    for hp in hyperparams:
        hp_dict[hp].append(clf.best_estimator_.get_params()[hp])

    # collect performance metrics
    acc_train = accuracy_score(y_train, clf.predict(X_train))
    acc_test = accuracy_score(y_test, clf.predict(X_test))
    f1_train = f1_score(y_train, clf.predict(X_train))
    f1_test = f1_score(y_test, clf.predict(X_test))
    auc_train = roc_auc_score(y_train, clf.predict(X_train))
    auc_test = roc_auc_score(y_test, clf.predict(X_test))

    logger.debug('train data performance:\n{}'.format(classification_report(y_train, clf.predict(X_train))))
    logger.debug('test data performance:\n{}'.format(classification_report(y_test, clf.predict(X_test))))
    logger.debug('TRAIN: predicted,actual values\n{}\n{}'.format(clf.predict(X_train), y_train))
    logger.debug('TEST:  predicted,actual values\n{}\n{}'.format(clf.predict(X_test), y_test))

    # check feature importance (QC for HC importance)
    # for fid in np.arange(10):
    #     model_clf.fit(X_train[fid],y_train[fid])
    #     feat_imp = model_clf.feature_importances_
    #     print('\nfid: {} r: {}'.format(fid, zip(*CV_r_valid)[0][fid]))
    #     print(feat_imp[70:], np.argsort(feat_imp)[70:])
    return {'acc_train': acc_train,
            'acc_test':  acc_test,
            'f1_train':  f1_train,
            'f1_test':   f1_test,
            'auc_train': auc_train,
            'auc_test':  auc_test,
            'hp_dict':   hp_dict}


def cluster(X, plot, n_clust=2):
    """
    Plots a simple hierarchical clustering of the feature matrix X. Clustering
    is done using Ward's algorithm. Variables in X are arranged by cluster.
    """
    fig = plt.figure()
    axd = fig.add_axes([0.09,0.1,0.2,0.8])
    axd.set_xticks([])
    axd.set_yticks([])
    link = sch.linkage(X, method='ward')
    clst = sch.fcluster(link, n_clust, criterion='maxclust')
    dend = sch.dendrogram(link, orientation='right')
    idx = dend['leaves']
    X = utils.reorder(X, idx, symm=False)
    axm = fig.add_axes([0.3,0.1,0.6,0.8])
    im = axm.matshow(X, aspect='auto', origin='lower', cmap=plt.cm.Reds, vmin=-0.5, vmax=0.5)
    axm.set_xticks([])
    axm.set_yticks([])
    axc = fig.add_axes([0.91,0.1,0.02,0.8])
    plt.colorbar(im, cax=axc)
    plt.savefig(os.path.join(plot, 'xbrain_clusters.pdf'))
    plt.close()

    return clst


def distributions(y, plot):
    """Plots data distribution of y."""
    sns.distplot(y, hist=False, rug=True, color="r")
    sns.plt.savefig(plot)
    sns.plt.close()


