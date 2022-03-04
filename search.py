# Libraries
import yaml
import time
import json
import shutil
import numpy as np
import pandas as pd

from pathlib import Path
from types import MethodType

# Specific libraries
from sklearn.metrics import make_scorer
#from sklearn.impute import SimpleImputer
#from sklearn.impute import IterativeImputer
from sklearn.preprocessing import MinMaxScaler
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

# ----------------------------------------
# Doesn't work add method dynamically.
# ----------------------------------------
# Librarie
from functools import wraps

def add_method(cls):
    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            return func(*args, **kwargs)
        setattr(cls, func.__name__, wrapper)
        # Note we are not binding func, but wrapper
        # which accepts self but does exactly the same
        # as func. Returning func means func can still be used
        # normally
        return func
    return decorator

# --------------------------------------------------------
# Configuration
# --------------------------------------------------------
# ------------------
# Load configuration
# ------------------
# Yaml path
PATH_YAML = Path('./datasets/dengue/settings.dengue.yaml')

# Load configuration from file
with open(PATH_YAML) as file:
    config = yaml.full_load(file)

# Variables
PATH_DATA = Path(config['filepath'])
FEATURES = config['features']
TARGETS = config['targets']
ESTIMATORS = sorted(set(config['estimators']))

# Define pipeline path
uuid = time.strftime("%Y%m%d-%H%M%S")
pipeline_path = Path(config['output']) / uuid

# Create workbench folder
pipeline_path.mkdir(parents=True, exist_ok=True)

# Save settings file.
shutil.copyfile(PATH_YAML, pipeline_path / 'settings.yaml')

# ------------------
# Load data
# ------------------
# Load data
data = pd.read_csv(PATH_DATA)

# .. note: In addition to keep only the full profiles, there
#          is the option to use SimpleImputer or IterativeImputer
#          to fill the missing values. For the second option it
#          can be done within the pipeline creation.
data = data.dropna(how='any', subset=FEATURES)

# Create X and y
X = data[FEATURES]
y = data[TARGETS]

# To numpy (for NeuralNet)
#X = X.to_numpy().astype(np.float32)
#y = y.to_numpy().astype(np.int64)

# Show
print("\nData from '%s':" % PATH_DATA)


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
    y: np.array (dataframe)
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
    y_true: np.array (dataframe!)
        Array with original data.
    y_pred: np.array
        Array with transformed data.
    y: np.array (dataframe!)
        Array with ....

    Returns
    -------
    dict-like
        Dictionary with the scores.
    """
    # Libraries
    from scipy.spatial.distance import cdist
    from scipy.spatial import procrustes
    from scipy.stats import spearmanr, pearsonr
    from sklearn.metrics import silhouette_score
    from sklearn.metrics import calinski_harabasz_score
    from sklearn.metrics import davies_bouldin_score
    from ls2d.metrics import gmm_scores
    from ls2d.metrics import gmm_ratio_score
    from ls2d.metrics import gmm_intersection_matrix
    from ls2d.metrics import gmm_intersection_area

    # Reduce computations which are expensive
    N = 1000 if len(y_true) > 1000 else len(y_true)
    #idx = np.random.choice(np.arange(len(y_true)), N, replace=False)
    idx = np.random.choice(np.arange(len(y_true)), N, replace=False)
    y_true_ = y_true.iloc[idx, :]
    y_pred_ = y_pred[idx]

    # Compute distances
    true_dist = cdist(y_true_, y_true_).flatten()
    pred_dist = cdist(y_pred_, y_pred_).flatten()

    # Compute scores
    pearson = pearsonr(true_dist, pred_dist)
    spearman = spearmanr(true_dist, pred_dist)

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

    # Compute scores for selected outcomes
    d_gmm = {}
    for c in y.columns:
        try:
            idx = y[c].notna()
            y_true_, y_pred_, y_ = \
                y_true[idx], y_pred[idx], y[c][idx]
            d_ = gmm_scores(y_pred_, y_)
            d_['silhouette'] = silhouette_score(y_pred_, y_)
            d_['calinski'] = calinski_harabasz_score(y_pred_, y_.ravel())
            d_['davies_b'] = davies_bouldin_score(y_pred_, y_.ravel())
            d_ = {'%s_%s'%(k, c) : v for k,v in d_.items()}
            d_gmm.update(d_)
        except Exception as e:
            print("Error: %s" % e)

    # Include loss from AE models.

    # Create dictioanry
    d = {}
    d['pearson'] = pearson[0]
    d['spearman'] = spearman[0]
    d['procrustes'] = disparity
    d.update(d_gmm)

    # Return
    return d


def predict(self, *args, **kwargs):
    return self.transform(*args, **kwargs)


# Compendium of results
compendium = pd.DataFrame()

# For each estimator
for i, est in enumerate(ESTIMATORS):

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

        # Option II:
        def predict(self, *args, **kwargs):
            return self.transform(*args, **kwargs)
        setattr(estimator.__class__, 'predict', predict)

        """
        # Option III:
        # Problem because cass does not exist when
        # reloading the instance from memory.
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
                                #('std', StandardScaler()),
                                ('minmax', MinMaxScaler()),
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
                              #cv=2,
                              #cv=(((range(X.shape[0]), []),)),
                              cv=(((slice(None), slice(None)),)),
                              scoring=custom_metrics,
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

# Save
compendium.to_csv(pipeline_path / 'results.csv', index=False)