# Libraries
import yaml
import time
import json
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
from pipeline import PipelineMemory
#from settings import _DEFAULT_FILTERS
from settings import _DEFAULT_ESTIMATORS
#from settings import _DEFAULT_PARAM_GRIDS
#from settings import _DEFAULT_METRICS
#from settings import _DEFAULT_TRANSFORMERS


# --------------------------------------------------------
# Configuration
# --------------------------------------------------------
# ------------------
# Load configuration
# ------------------
# Load configuration from file
with open('settings.iris.yaml') as file:
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

# Create X and y
X = data[config['features']]
y = data.target

# Show
print("\nData from '%s':" % config['filepath'])
print(data)


"""
import torch
from torch.utils.data import DataLoader
from ae import Autoencoder
from ae import train_autoencoder

a = Autoencoder(
    input_size=4,
    layers=[3],
    latent_dim=2,
)

loader_train = DataLoader(X.to_numpy(), 16, shuffle=True)
loader_test = DataLoader(X.to_numpy(), 16, shuffle=False)

optimizer = torch.optim.Adam(a.parameters(), lr=0.01)

losses = train_autoencoder(a, optimizer, loader_train, loader_test, 10,
                           # animation_data=loader_train_no_shuffle,
                           # animation_colour=[train_info['shock'],
                           #                   train_info['bleeding'],
                           #                   train_info['ascites'],
                           #                   train_info['abdominal_pain'],
                           #                   train_info['bleeding_mucosal'],
                           #                   train_info['bleeding_gum'],
                           #                   train_info['bleeding_skin'],
                           #                   train_info['gender']],
                           # animation_labels=['Shock', 'Bleeding', 'Ascites', 'Abdominal pain',
                           #                   'Bleeding mucosal', 'Bleeding gum',
                           #                   'Bleeding skin', 'Gender'],
                           # animation_path='animation.gif'
                           )




import sys
sys.exit()
"""


# --------------------------------------------------------
# Main
# --------------------------------------------------------
# Note, when creating the PipelineMemory, it is possible
# to include other steps such as filtering (e.g. iqr15),
# transformations (e.g. kbins10), standarization (e.g. std),
# balancing of classes (e.g. smote), ...

def predict(self, *args, **kwargs):
    return self.transform(*args, **kwargs)

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
    y = est.predict(X)
    # Metrics
    m = custom_metrics_(X, y)
    # Add information
    # Return
    return m


def custom_metrics_(y_true, y_pred):
    """This method computes the metrics.

    .. note: It has to be included in GridSearchCv
             using the make_scorer function previously.

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

    # Compute distances
    true_dist = cdist(y_true, y_true).flatten()
    pred_dist = cdist(y_pred, y_pred).flatten()

    # Compute scores
    pearson = pearsonr(true_dist, pred_dist)
    spearman = spearmanr(true_dist, pred_dist)

    # Return
    return {
        'pearson': pearson[0],
        'spearman': spearman[0]
    }




# Compendium of results
compendium = pd.DataFrame()

# For each estimator
for i, est in enumerate(estimators):

    # Get the estimator.
    estimator = _DEFAULT_ESTIMATORS[est]

    # Information
    print("\n    Method: %s. %s" % (i, estimator))

    # Option I: Not working.
    # Add the predict method if it does not have it.
    #estimator.predict = MethodType(predict, estimator)

    # Option II:
    # Dynamically create wrapper
    class Wrapper(estimator.__class__):
        def predict(self, *args, **kwargs):
            return self.transform(*args, **kwargs)

    # Create pipeline
    pipe = PipelineMemory(steps=[
                                ('nrm', Normalizer()),
                                (est, Wrapper())
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
    grid = GridSearchCV(pipe, param_grid=config['params'][est],
                              cv=2, scoring=custom_metrics,
                              return_train_score=True, verbose=2,
                              refit=False, n_jobs=1)

    # Fit grid search
    grid.fit(X, X)

    # Save results as csv
    results = grid.cv_results_

    # Save
    df = pd.DataFrame(results)
    df.insert(0, 'estimator', _DEFAULT_ESTIMATORS[est].__class__.__name__)
    df.insert(1, 'slug_short', pipe.slug_short)
    df.insert(2, 'slug_long', pipe.slug_long)
    df.to_csv(pipeline_path / pipe.slug_short / 'results.csv', index=False)

    # Append to total results
    compendium = compendium.append(df)

# Save all
compendium.to_csv(pipeline_path / 'results.csv', index=False)