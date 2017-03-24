#!/usr/bin/env python
"""Routines for relating neural activity with clinical variables."""

import os, sys, glob
from copy import copy
import collections
import logging
import random
import string
import re

import numpy as np
from scipy import linalg
from scipy.optimize import curve_fit
from scipy.stats import lognorm, randint, uniform, mode, spearmanr
import scipy.cluster.hierarchy as sch
from scipy.spatial.distance import pdist, squareform

from sklearn.preprocessing import scale, LabelEncoder, label_binarize, LabelBinarizer
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.feature_selection import SelectFromModel
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, f1_score, roc_auc_score, calinski_harabaz_score, silhouette_score
from sklearn.cluster import AgglomerativeClustering, MiniBatchKMeans
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

import matplotlib
matplotlib.use('Agg')   # Force matplotlib to not use any Xwindows backend
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns

import xbrain.correlate as corr
import xbrain.utils as utils
import xbrain.rcca as rcca

logger = logging.getLogger(__name__)

def powers(n, exps):
    """returns a list of n raised to the powers in exps"""
    ns = []
    for exp in exps:
        ns.append(n**exp)
    return(np.array(ns))

def classify(X_train, X_test, y_train, y_test, method, output):
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

    # use AUC score for unbalanced methods b/c outliers are more interesting
    if method == 'anomaly':
        scoring = 'roc_auc'
        model = 'RIF'
    elif method == 'ysplit':
        scoring = 'roc_auc'
        model = 'SVC'
    else:
        scoring = 'f1_macro'
        model = 'SVC'

    # use random forest for feature selection if we have 10x more features than samples
    if X_train.shape[1] > X_train.shape[0]*10:
        logger.info('using random forest for feature selection since features={} > samples={}*10'.format(X_train.shape[1], X_train.shape[0]))
        dim_mdl = SelectFromModel(
                      RandomForestClassifier(n_jobs=6, class_weight='balanced',
                          n_estimators=2000, max_depth=2, min_samples_split=3))
        dim_mdl.fit(X_train, y_train)
        X_train = dim_mdl.transform(X_train)
        X_test = dim_mdl.transform(X_test)
        n_features = X_train.shape[1]
        corr.plot_X(X_train, output, title='test-vs-train', X2=X_test)
        logger.info('random forest retained {} features'.format(n_features))

        #dim_mdl = LogisticRegression(penalty="l1", class_weight='balanced', n_jobs=6)
        #dim_hp = {'C': uniform(0.01, 100)}
        #logger.debug('Inner Loop: Logistic Regression w/ l1 penalty for feature selection')
        #dim_mdl = RandomizedSearchCV(dim_mdl, dim_hp, n_iter=100, scoring=scoring, verbose=1)

    # settings for the classifier
    if model == 'Logistic':
        clf_mdl = LogisticRegression(class_weight='balanced', verbose=1)
        clf_hp = {'C': np.arange(0.1, 2.1, 0.1), 'penalty': ['l1', 'l2']}
        scale_data = True
    elif model == 'SVC':
        clf_mdl = LinearSVC(max_iter=10000, tol=0.00001, class_weight='balanced')
        clf_hp = {'C': powers(2, range(-20,5))} # replaces lognorm(10, loc=2**-15, scale=0.001)
        scale_data = True
    elif model == 'RFC':
        clf_mdl = RandomForestClassifier(n_jobs=6, class_weight='balanced', n_estimators=2000)
        clf_hp = {'min_samples_split': np.round(np.linspace(n_features*0.025, n_features*0.2, 10)),
                  'max_depth': np.array([None, 2, 4, 6, 8, 10]),
                  'criterion': ['gini', 'entropy']}
        scale_data = False
    elif model == 'RIF':
        pct_outliers = len(np.where(y_train == -1)[0]) / float(len(y_train))
        clf_mdl = IsolationForest(n_jobs=6, n_estimators=1000, contamination=pct_outliers, max_samples=n_features)
        clf_hp = {}
        scale_data = False

    if scale_data:
        X_train = scale(X_train)
        X_test = scale(X_test)

    # perform randomized hyperparameter search to find optimal settings
    if method == 'anomaly':
        clf = clf_mfl
        clf.fit(X_train)
        hp_dict = hyperparams
    else:

        logger.debug('Inner Loop: Classification using {}'.format(model))
        clf = GridSearchCV(clf_mdl, clf_hp, scoring=scoring, verbose=1)
        clf.fit(X_train, y_train)

        # record best classification hyperparameters found
        for hp in clf_hp:
            hp_dict[hp].append(clf.best_params_[hp])
        logger.info('Inner Loop complete, best parameters found:\n{}'.format(hp_dict))
        logger.info('maximum test score during hyperparameter CV ({}): {}'.format(scoring, np.max(clf.cv_results_['mean_test_score'])))

    X_train_pred = clf.predict(X_train)
    X_test_pred = clf.predict(X_test)

    # make coding of anomalys like other classifiers
    if method == 'anomaly':
        X_train_pred[X_train_pred == -1] = 0
        X_test_pred[X_test_pred == -1] = 0

    # collect performance metrics
    acc = (accuracy_score(y_train, X_train_pred), accuracy_score(y_test, X_test_pred))
    rec = (recall_score(y_train, X_train_pred, average='macro'), recall_score(y_test, X_test_pred, average='macro'))
    prec = (precision_score(y_train, X_train_pred, average='macro'), precision_score(y_test, X_test_pred, average='macro'))
    f1 = (f1_score(y_train, X_train_pred, average='macro'), f1_score(y_test, X_test_pred, average='macro'))

    # transforms labels to label indicator format, needed for multiclass
    lb = LabelBinarizer()
    lb.fit(y_train)
    auc = (roc_auc_score(lb.transform(y_train), lb.transform(X_train_pred), average='macro'),
           roc_auc_score(lb.transform(y_test), lb.transform(X_test_pred), average='macro'))

    logger.info('TRAIN: confusion matrix\n{}'.format(confusion_matrix(y_train, X_train_pred)))
    logger.info('TEST:  confusion matrix\n{}'.format(confusion_matrix(y_test, X_test_pred)))

    # check feature importance (QC for HC importance)
    # for fid in np.arange(10):
    #     model_clf.fit(X_train[fid],y_train[fid])
    #     feat_imp = model_clf.feature_importances_
    #     print('\nfid: {} r: {}'.format(fid, zip(*CV_r_valid)[0][fid]))
    #     print(feat_imp[70:], np.argsort(feat_imp)[70:])
    return {'accuracy': acc,
            'recall' : rec,
            'precision': prec,
            'f1': f1,
            'auc': auc,
            'X_test_pred': X_test_pred,
            'n_features_retained': n_features,
            'hp_dict': hp_dict}


def estimate_biotypes(X, y, output):
    """
    Finds features in X that have a significant spearman's rank correlation
    with at least 1 of the variables in y (uncorrected p < 0.005). The reduced
    feature set X_red is then used to estimate the optimal number of cannonical
    variates that represent the mapping between X_red and y using cross
    validation.

    Next, this estimates the optimal number of clusters in the CCA
    representation of the data.

    Returns the indicies of the reduced features set, the number of cannonical
    variates found, the optimal number of clusters, and the cannonical variates.
    """

    # reduce X using the spearman rank correlation between X and y
    # in a for loop due to the immense size of (X.shape[1]+y.shape[1])^2
    # should implement chunking to speed this up
    logger.debug('testing {} X verticies against {} y variables'.format(X.shape[1], y.shape[1]))
    idx = np.zeros(X.shape[1], dtype=bool)
    for vertex in range(X.shape[1]):
        # takes the p values of the rs between the variation in connectivity in
        # a single vertex, and all predictors of interest
        p_vals = spearmanr(X[:, vertex], y)[1][1:, 0]

        # if connection is significantly related to any y variable, flag
        if sum(p_vals <= 0.005) > 0:
            idx[vertex] = np.bool(1)

    logger.debug('{} verticies significantly related to at least one variable in y'.format(sum(idx)))
    X_red = X[:, idx]

    # use regularized CCA to determine the optimal number of cannonical variates
    logger.info('biotyping: cannonical correlation 10-fold cross validation to find brain-behaviour mappings')

    # small search space for testing
    #regs = np.array(np.logspace(-4, 2, 10)) # regularization b/t 1e-4 and 1e2
    regs = np.array(np.logspace(-4, 2, 3)) # regularization b/t 1e-4 and 1e2
    numCCs = np.arange(2, 11)

    cca = rcca.CCACrossValidate(numCCs=numCCs, regs=regs, verbose=True)
    cca.train([X, y])

    n_best_cc = cca.best_numCC
    best_reg = cca.best_reg
    comps = cca.comps[0] # [0] uses comps from X, [1] uses comps from y

    # estimate number of clusters by maximizing cluster quality criteria
    clst_score = np.zeros(18)
    clst_tests = np.array(range(2,20))
    for i, n_clst in enumerate(clst_tests):
        # ward's method, euclidean distance
        clst = AgglomerativeClustering(n_clusters=n_clst)
        clst.fit(comps)

        # CH score gives me a very large number of clusters -- not sure which
        # to use ATM so sticking with the one that gives me a small number ...
        # should revisit when I test with many more y features
        #clst_score[i] = calinski_harabaz_score(comps, clst.labels_)
        clst_score[i] = silhouette_score(comps, clst.labels_)

    # take the best performing clustering
    n_best_clst = clst_tests[clst_score == np.max(clst_score)][0]
    clst = AgglomerativeClustering(n_clusters=n_best_clst)
    clst.fit(comps)
    clst_labels = clst.labels_

    # calculate connectivity centroids for each biotype
    clst_centroids = np.zeros((n_best_clst, X.shape[1]))
    for n in range(n_best_clst):
        clst_centroids[n, :] = np.mean(X[clst_labels == n, :], axis=0)

    logger.info('biotyping: found {} cannonical variates, {} n biotypes'.format(n_cc, n_best_clst))

    # save biotype information
    np.savez_compressed(os.path.join(output, 'xbrain_biotype.npz'),
                                              clst_centroids=clst_centroids,
                                              n_best_clst=n_best_clst,
                                              n_best_cc=n_best_cc,
                                              best_reg=best_reg,
                                              idx=idx,
                                              comps=comps,
                                              X=X,
                                              y=y)
    mdl = np.load(os.path.join(output, 'xbrain_biotype.npz'))

    # plot biotype info
    #D = squareform(pdist(comps))
    sns.clustermap(comps, method='ward', metric='euclidean')
    sns.plt.savefig(os.path.join(output, 'xbrain_component_clusters'))
    sns.plt.close()

    plt.plot(clst_score)
    plt.title('Calinski Harabaz scores')
    plt.ylabel('Variance Ratio Criterion')
    plt.xlabel('Number of Clusters (k)')
    plt.xticks(range(len(cluster_tests)), cluster_tests)
    plt.savefig(os.path.join(output, 'xbrain_n_cluster_estimation.pdf'))
    plt.close()

    return mdl


def biotype(X_train, X_test, y_train, mdl):
    """
    X is a ROI x SUBJECT matrix of features (connectivities, cross-brain
    correlations, etc.), and y N by SUBJECT matrix of outcome measures (i.e.,
    cognitive variables, demographics, etc.).

    Will decompose X_train and y_train into n_cc cannonical variates (n is
    determined via cross-validation in estimate_biotypes). The previously
    estimated regularization parameter reg will be used.

    mdl is a dictionary containing biotype information (see estimate_biotypes).

    CCA will only be run on the features defined in idx.

    We want to estimate the number of variates before training the model, and
    then try to find the same number of variates for each fold. Use
    estimate_cca to find n and idx.

    The pipeline is as follows:

    1) Use connonical correlation to find a n_cc dimentional mapping between
       the reduced feature matrix X_train_red and y_train.
    2) Cluster the subjects using Ward's hierarchical clustering into biotypes,
       generating biotype labels (the new y_train).
    3) Train a linear discriminate classifier on X_train_red and the cluster
       labels, and then run this model on X_test, to produce biotype labels for
       the test set (the new y_test).

    This returns y_train and y_test, the discovered biotypes for classification.
    """
    # get information from biotype model
    idx = mdl['idx']
    n_clst = mdl['n_best_clst']
    reg = mdl['best_reg']
    n_cc = mdl['n_best_cc']

    # idx is found using the spearman rank correlations between X and y
    X_train_red = X_train[:, idx]
    X_test_red = X_test[:, idx]

    # use regularized CCA to find brain-behaviour mapping from the training set
    logger.info('biotyping training set')
    cca = rcca.CCA(numCC=n_cc, reg=reg)
    cca.train([X_train_red, y_train])
    comps = cca.comps[0] # [0] uses comps from X, [1] uses comps from y

    # cluster these components to produce n_clst biotypes, y_train
    clst = AgglomerativeClustering(n_clusters=n_clst)
    clst.fit(comps)
    y_train = clst.labels_

    # use LDA to predict the labels of the test set, y_test
    lda = LinearDiscriminantAnalysis()
    lda.fit(X_train_red, y_train)
    y_test = lda.predict(X_test_red)

    return y_train, y_test


def get_states(d_rs, k=5):
    """
    Accepts a ROI x TIMEPOINT dynamic connectivity matrix, and returns K states
    as determined by K-means clustering (ROI x STATE).

    Uses a slighly less accurate implementation of kmeans designed for very
    large numbers of samples: D. Sculley, Web-Scale K-Means Clustering.
    """
    clf = MiniBatchKMeans(n_clusters=k)
    logger.debug('running kmeans on X {}, k={}'.format(d_rs.shape, k))
    clf.fit(d_rs.T)
    return(clf.cluster_centers_.T)


def fit_states(d_rs, states):
    """
    Accepts a ROI x TIMEPOINT dynamic connectivity matrix, and a ROI x STATE
    matrix (composed of the outputs from get_states), and computes the
    regression coefficients for each time window against all input states.
    Returns the sum of the coefficients across all time windows for each state.
    Could be thought of as a measure of how much relative time this subject
    spent in each state during the scan.
    """
    clf = LinearRegression()
    clf.fit(d_rs, states)
    return(np.sum(clf.coef_, axis=1))


def r_to_z(R):
    """Fischer's r-to-z transform on a matrix (elementwise)."""
    return(0.5 * np.log((1+R)/(1-R)))


def r_to_d(R):
    """Converts a correlation matrix R to a distance matrix D."""
    return(np.sqrt(2*(1-R)))


def standardize(X):
    """z-scores each column of X."""
    return((X - X.mean(axis=0)) / X.std(axis=0))


def gauss(x, *p):
    """Model gaussian to fit to data."""
    A, mu, sigma = p
    return A*np.exp(-(x-mu)**2/(2.*sigma**2))


def find_outliers(y):
    """
    Assumes y is the mixture of scores drawn from a normal distribution and
    some unusual, unknown distribution. Fits a gaussian curve to y, and finds
    that curve's mean and standard deviation. This model assumes that only 2.5%
    of the data should fall below the -2*sigma line. Finds the actual percentage
    of datapoints below that line, subtracts 2.5% from it (to account for the
    expected percentage), and then flags all of those data points as outliers
    with negative -1. Normal values are set to 1. Returns this modified y
    vector, and the percentage of the data that are considered outliers.
    """
    binned_curve, bin_edges = np.histogram(y, bins=len(y)/10, density=True)
    bin_centres = (bin_edges[:-1] + bin_edges[1:])/2
    # initial curve search values
    p0 = [1., 0., 1.]
    coeff, var_matrix = curve_fit(gauss, bin_centres, binned_curve, p0=p0)

    mean = coeff[1]
    sd = coeff[2]

    # interested in more than expected number of values below -2 SD
    null_cutoff = mean - 2*sd
    null_outliers_pct = 0.025 # 2.5% of data expected below 2 sd
    real_outliers_pct = len(np.where(y < null_cutoff)[0]) / float(len(y))
    diff_outliers_pct = real_outliers_pct - null_outliers_pct
    diff_cutoff = np.percentile(y, diff_outliers_pct*100)

    y_outliers = copy(y)
    y_outliers[y <= diff_cutoff] = 0
    y_outliers[y > diff_cutoff] = 1

    logger.info('auto-partitioning y at the {}th percentile (non-gaussian outliers)'.format(diff_outliers_pct*100))

    #fitted_curve = gauss(bin_centres, *coeff)
    #plt.plot(bin_centres, binned_curve, color='k', label='y')
    #plt.plot(bin_centres, fitted_curve, color='r', label='gaussian fit')
    #plt.axvline(x=diff_cutoff, color='k', linestyle='--')
    #plt.show()

    return y_outliers


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


def pca_reduce(X, n=1, pct=0, X2=False):
    """
    Uses PCA to reduce the number of features in the input matrix X. n is
    the target number of features in X to retain after reduction. pct is the
    target amount of variance (%) in the original matrix X to retain in the
    reduced feature matrix. When n and pct disagree, compresses the feature
    matrix to the larger number of features. If X2 is defined, the same
    transform learned from X is applied to X2.
    """
    if not utils.is_probability(pct):
        raise Exception('pct should be a probability (0-1), is {}'.format(pct))

    clf = PCA()
    clf = clf.fit(X)
    cumulative_var = np.cumsum(clf.explained_variance_ratio_)
    n_comp_pct = np.where(cumulative_var >= pct)[0][0] + 1 # fixes zero indexing

    if n > n_comp_pct:
        pct_retained = cumulative_var[n-1]
        cutoff = n
    else:
        pct_retained = cumulative_var[n_comp_pct-1]
        cutoff = n_comp_pct

    logger.info('X {} reduced to {} components, retaining {} % of variance'.format(X.shape, cutoff, pct_retained))

    # reduce X to the defined number of components
    clf = PCA(n_components=cutoff)
    clf.fit(X)
    X_transformed = clf.transform(X)

    # sign 1st PC of X if inverse to mean of X
    if cutoff == 1:
        X_transformed = sign_flip(X_transformed, X)

    # if X2 is defined, apply the transform learnt from X to X2 as well
    if np.any(X2):
        X2_transformed = clf.transform(X2)

        if cutoff == 1:
            X2_transformed = sign_flip(X2_transformed, X2)

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


def make_classes(y, target_group):
    """transforms label values for classification, including target_group"""
    le = LabelEncoder()
    le.fit(y)
    logger.info('y labels {} transformed to {}'.format(le.classes_, np.arange(len(le.classes_))))
    y = le.transform(y)

    if target_group:
        target_group = int(np.where(target_group == le.classes_)[0][0]) # ugly type gymnastics should be fixed

    return(y, target_group)


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


def diagnostics(X_train, X_test, y_train, y_test, y_train_raw, output):
    """
    A pipeline for assessing the inputs to the classifier. Runs on
    a single fold.

    + Loads X and y. If y_raw has multiple preditors, the top PC is calculated.
    + Plots a distribution of y_raw, compressed to 1 PC.
    + Saves a .csv with this compressed version of y_raw.
    + Plots the top 3 PCs of X_train, with points colored by group y_train. This
      plot should have no trivial structure.
    + Plots a hierarchical clustering of X_train.
    + Uses MDMR to detect relationship between y_raw and X_train.
    + Runs iterative classification of the training data using logistic
      regression and l1 regularization with varying levels of C.
    """
    logger.info('pre-test: detecting gross relationship between neural and cognitive data')
    # compress y to a single vector using PCA if required
    if len(y_train_raw.shape) == 2 and y_train_raw.shape[0] > 1:
        y_1d = copy(pca_reduce(y_train_raw))
    else:
        y_1d = copy(y_train_raw)

    # plot the y variable (1d) before generating classes
    distributions(y_1d.T, os.path.join(output, 'xbrain_y_dist.pdf'))

    # print the top 3 PCs of X, colour coding by y group (is X trivially seperable?)
    pca_plot(X_train, y_train, output)

    np.save(os.path.join(output, 'xbrain_y.npy'), y_1d)
    np.save(os.path.join(output, 'xbrain_X.npy'), X_train)

    # plot a hierarchical clustering of the feature matrix X
    clst = cluster(X_train, output)

    # use MDMR to find a relationship between the X matrix and all y predictors
    # broken -- need to go over matrix transpositions...
    #F, F_null, v = mdmr(y_train_raw, X_train, method='euclidean')
    #thresholds = sig_cutoffs(F_null, two_sided=False)
    #if F > thresholds[1]:
    #    logger.info('mdmr: relationship detected: F={} > {}, variance explained={}'.format(F, thresholds[1], v))
    #else:
    #    logger.warn('mdmr: no relationship detected, variance explained={}'.format(v))

    # sparsity test over a range of C values
    Cs = powers(2, np.arange(-7, 7, 0.5))
    features, train_score, test_score = [], [], []
    for C in Cs:
        # fit a logistic regression model with l1 penalty (for sparsity)
        mdl = LogisticRegression(n_jobs=6, class_weight='balanced', penalty='l1', C=C)
        mdl.fit(X_train, y_train)
        # use non-zero weighted features to reduce X
        dim_mdl = SelectFromModel(mdl, prefit=True)
        X_train_pred = mdl.predict(X_train)
        X_test_pred= mdl.predict(X_test)
        features.append(dim_mdl.transform(X_train).shape[1])
        train_score.append(f1_score(y_train, X_train_pred, average='macro'))
        test_score.append(f1_score(y_test, X_test_pred, average='macro'))

    # plot test and train scores against number of features retained
    plt.plot(train_score, color='black')
    plt.plot(test_score, color='red')
    plt.xticks(np.arange(len(features)), features)
    plt.legend(['train', 'test'])
    plt.xlabel('n features retained')
    plt.ylabel('F1 score (macro)')
    plt.savefig(os.path.join(output, 'xbrain_overfit-analysis.pdf'))

    sys.exit()


