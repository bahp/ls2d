# Libraries
import yaml
import time
import json
import numpy as np
import pandas as pd

from pathlib import Path
from types import MethodType

# Specific libraries
from sklearn.metrics import make_scorer
#from sklearn.impute import SimpleImputer
#from sklearn.impute import IterativeImputer
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
#from imblearn.over_sampling import SMOTE

# Own libraries
from ls2d.pipeline import PipelineMemory
#from settings import _DEFAULT_FILTERS
from ls2d.settings import _DEFAULT_ESTIMATORS
#from settings import _DEFAULT_PARAM_GRIDS
#from settings import _DEFAULT_METRICS
#from settings import _DEFAULT_TRANSFORMERS


from functools import wraps


def add_method(cls):
    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            return func(*args, **kwargs)
        setattr(cls, func.__name__, wrapper)
        # Note we are not binding func, but wrapper which accepts self but does exactly the same as func
        return func # returning func means func can still be used normally
    return decorator

# --------------------------------------------------------
# Configuration
# --------------------------------------------------------
# ------------------
# Load configuration
# ------------------
# Load configuration from file
with open('datasets/dengue/settings.dengue.yaml') as file:
    config = yaml.full_load(file)

# Features
features = sorted(set(config['features']))

# Algorithms
estimators = sorted(set(config['estimators']))

# Define pipeline path
uuid = time.strftime("%Y%m%d-%H%M%S")
pipeline_path = Path(config['output']) / uuid

# ------------------
# Load data
# ------------------
# Load data
data = pd.read_csv(config['filepath'])

# There is the option to drop those that contain NaN or use
# a SimpleImputer or IterativeImputer to fill the missing
# values. For the second option it can be done within the
# pipeline creation.
data = data.dropna(how='any', subset=config['features'])

# Create X and y
X = data[config['features']]
y = data[config['targets']]

# To numpy (for NeuralNet)
#X = X.to_numpy().astype(np.float32)
#y = y.to_numpy().astype(np.int64)

# Show
print("\nData from '%s':" % config['filepath'])


# --------------------------------------------------------
# Main
# --------------------------------------------------------
# Note, when creating the PipelineMemory, it is possible
# to include other steps such as filtering (e.g. iqr15),
# transformations (e.g. kbins10), standarization (e.g. std),
# balancing of classes (e.g. smote), ...


def custom_metrics(est, X, y):
    """This method computes the metrics.

    .. note: The X and y are the input data. Thus we need
             to apply the transformation ourselves
             manually.

    .. note: It should be included in GridSearchCV directly
             without using the make_scorer function.

    .. note: The scoring function must return numbers, thus
             including the uuid does not work here.

    Parameters
    ----------
    est: object
        The estimator or algorithm.
    X:  np.array
        The X data in fit.
    y: np.array
        The y data in fit.
    """
    # Transform
    y_embd = est.predict(X)
    # Metrics
    m = custom_metrics_(X, y_embd, y)
    #y = est.transform(X)
    # Metrics
    #m = custom_metrics_(X, y)
    ## Add information
    #m['split'] = est.split
    #m['pipeline'] = est.pipeline
    # Return
    return m


def custom_metrics_(y_true, y_pred, y, n=1000):
    """This method computes the metrics.

    .. note: It has to be included in GridSearchCV
             using the make_scorer function previously.

    .. todo: At the moment we are computing the GMM ratio
             based on the labels (y). It would be nice to
             allow in the configuration file to indicate
             which other labels are of interest (e.g. shock)

    .. note: computation of pairwise distances takes too
             long when datasets are big. Thus, instead
             select a random number of examples and compute
             on them.

    Parameters
    ----------
    y_true: np.array
        Array with original data.
    y_pred: np.array
        Array with transformed data.

    Returns
    -------
    dict-like
        Dictionary with the scores.
    """
    # Libraries
    from scipy.spatial.distance import cdist
    from scipy.spatial import procrustes
    from scipy.stats import spearmanr, pearsonr
    from sklearn.mixture import GaussianMixture
    from sklearn.mixture import BayesianGaussianMixture
    from sklearn.metrics import silhouette_score
    from ls2d.metrics import gmm_ratio_score
    from ls2d.metrics import gmm_intersection_matrix

    """
    # Compute distances
    true_dist = cdist(y_true, y_true).flatten()
    pred_dist = cdist(y_pred, y_pred).flatten()

    # Compute scores
    pearson = pearsonr(true_dist, pred_dist)
    spearman = spearmanr(true_dist, pred_dist)
    """
    pearson = [-1]
    spearman = [-1]

    # Compute procrustes
    """
    y_padd = np.c_[y_pred,
        np.zeros(y_true.shape[0]),
        np.zeros(y_true.shape[0]),
        np.zeros(y_true.shape[0])]
    print("\n\n\n")
    print(y_true.shape, y_padd.shape)
    """
    try:
        mtx1, mtx2, disparity = procrustes(y_true, y_pred)
    except ValueError as e:
        mtx1, mtx2, disparity = None, None, -1

    # Compute gmm scores for selected outcomes
    gmm_scores = {}
    for c in y.columns:
        try:
            idx = y[c].notna()
            y_true_, y_pred_, y_ = \
                y_true[idx], y_pred[idx], y[c][idx]
            gmm_scores['gmm_ratio_sum_%s' % c] = \
                gmm_ratio_score(y_pred_, y_, 'sum')
        except:
            gmm_scores['gmm_ratio_sum_%s' % c] = -1

    # Compute silhouette
    #silhouette = silhouette_score(y_pred, y, metric="sqeuclidean")

    # Create metrics dictioanry
    d = {}
    d['pearson'] = pearson[0]
    d['spearman'] = spearman[0]
    d['procrustes'] = disparity
    d.update(gmm_scores)

    # Return
    return d


def predict(self, *args, **kwargs):
    return self.transform(*args, **kwargs)


# Compendium of results
compendium = pd.DataFrame()

# For each estimator
for i, est in enumerate(estimators):

    # Get the estimator
    estimator = _DEFAULT_ESTIMATORS[est]
    # Get the param grid
    param_grid = config['params'].get(est, {})

    # Information
    print("\n    Method: %s. %s" % (i, estimator))

    # Option I: Not working.
    # Add the predict method if it does not have it.
    #estimator.predict = MethodType(predict, estimator)
    #estimator.predict = predict.__get__(estimator)
    #estimator.predict = predict

    aux = getattr(estimator, "predict", None)
    if not callable(aux):

        # Option I:
        #@add_method(estimator.__class__)
        #def predict(self, *args, **kwargs):
        #    return self.transform(*args, **kwargs)

        # Option I.5:
        def predict(self, *args, **kwargs):
            return self.transform(*args, **kwargs)
        setattr(estimator.__class__, 'predict', predict)


        """
        # Option II:
        # Dynamically create wrapper
        class Wrapper(estimator.__class__):
            def predict(self, *args, **kwargs):
                return self.transform(*args, **kwargs)
        estimator = Wrapper()
        """

    # Create pipeline
    pipe = PipelineMemory(steps=[
                                #('simp', SimpleImputer()),
                                #('iimp', IterativeImputer()),
                                #('nrm', Normalizer()),
                                ('std', StandardScaler()),
                                (est, estimator)
                          ],
                          memory_path=pipeline_path,
                          memory_mode='pickle',
                          verbose=True)

    # Warning
    if (pipeline_path / pipe.slug_short).exists():
        print('[Warning] The pipeline <{0}> already exists... skipped!' \
              .format(pipeline_path / pipe.slug_short))
        continue

    # Create grid search (another option is RandomSearchCV)
    grid = GridSearchCV(pipe, param_grid=param_grid,
                              cv=2, scoring=custom_metrics,
                              return_train_score=True, verbose=2,
                              refit=False, n_jobs=1)

    # Fit grid search
    grid.fit(X, y)

    # Save results as csv
    results = grid.cv_results_

    # Add information
    df = pd.DataFrame(results)
    df.insert(0, 'estimator', _DEFAULT_ESTIMATORS[est].__class__.__name__)
    df.insert(1, 'slug_short', pipe.slug_short)
    df.insert(2, 'slug_long', pipe.slug_long)
    df['path'] = pipeline_path / pipe.slug_short
    df.to_csv(pipeline_path / pipe.slug_short / 'results.csv', index=False)

    # Append to total results
    compendium = compendium.append(df)

# Save all
compendium.to_csv(pipeline_path / 'results.csv', index=False)
