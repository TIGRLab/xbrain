**xbrain: use functional neuroimaging features to predict cognitive scores**

A platform for conducting classification experiments using functional neuroimaging data and out-of-scanner cognitive tests. Neuroimaging features can be a mix of within-brain connectivity, and intra-subject correlations during task / natural viewing experiments.

+ inter-subject correlation (xcorr) uses the method implemeted [here](http://journals.plos.org/plosone/article?id=10.1371/journal.pone.0041196).
+ intra-subject correlation (connectivity) uses simple between-ROI connectivity.
+ performs n-fold cross validation (outer loop: test and train split).
+ performs hyperparameter cross validation (inner loop: train and validation split).
+ xcorr features are calculated using a template population drawn from the training set only so there is no information leakage between the training and test sets.
+ all subject are correlated against this template, regardless of group membership.
+ if more than one cognitive predictor (y) is desired, uses PCA to reduce this to a single aggregate cognitive score.
+ y is then split into a low and high group at the desired percentile.
+ if y is discrete (e.g., diagnosis), contrasts the target group with every other group.

