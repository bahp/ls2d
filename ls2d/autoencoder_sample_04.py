# Generic
import torch
import skorch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Basic Functions in Torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader

# Basic Functions in Skorch
from skorch import NeuralNet
from skorch.callbacks import EpochTimer
from skorch.callbacks import PrintLog
from skorch.callbacks import EpochScoring
from skorch.callbacks import PassthroughScoring
from skorch.dataset import CVSplit
from skorch.utils import get_dim
from skorch.utils import is_dataset
from skorch.utils import to_numpy
from skorch import NeuralNetClassifier

# Basic Functions in Sklearn
from sklearn import datasets
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV

# Own
from autoencoder import AE

# ------------------------------------
# Config
# ------------------------------------
BATCH_SIZE = 128
MAX_EPOCHS = 100
LR = 0.1

# ------------------------------------
# Load Data
# ------------------------------------
# Load data
X, y = make_classification(1000, 20,
    n_informative=10, random_state=0)
X, y = iris = datasets.load_iris(return_X_y=True)
X = X.astype(np.float32)
y = y.astype(np.int64)


# -------------------------------------------------
# Create pipelinef
# -------------------------------------------------
# Create model
model = NeuralNet(
    AE,
    module__layers=[4, 100, 10, 3, 2],
    max_epochs=MAX_EPOCHS,
    batch_size=BATCH_SIZE,
    optimizer__lr=LR,
    criterion=torch.nn.MSELoss,
    iterator_train__shuffle=False,
    train_split=None
)

# Create pipeline
pipe = Pipeline([
    ('scale', StandardScaler()),
    ('net', model),
])

def custom_metrics(est, X, y):
    """This method computes the metrics.

    TODO: Include the following metrics
      - Pearson
      - Spearman
      - GMM
      - Classification accuracy?
      - Loss!

    Parameters
    ----------
    est: Pipeline
        The estimator and or pipeline.
    X: numpy array
        The features
    y: numpy array
        The outputs

    Returns
    -------
    """
    # Transform
    y = est.predict(X)


    # Return
    return mean_squared_error(y, X)

    # Create grid search
    # Since we are not really classification accuracy might be a
    # bad metric, since we are using the encoder/decoder the loss
    # would be the best? But not for usability.


# Parameters
params = {
    'net__lr': [0.01],
    'net__max_epochs': [100],
    'net__module__layers': [[4, 2], [4, 3, 2]]
}

# Criterion?

# Grid search
gs = GridSearchCV(pipe, params, refit=True,
    cv=2,
    scoring=custom_metrics)
#scoring='neg_mean_squared_error')


# Fit
gs.fit(X, y=X)

# Show
print("\n")
print("DataFrame:")
print(pd.DataFrame(gs.cv_results_))
print("\n")
print("Best score: %s" % gs.best_score_)
print("Best param: %s" % gs.best_params_)


# -------------------------------------------------
# Show results
# -------------------------------------------------
# Sample data
top = X[:1, :]

# Network
network = gs.best_estimator_['net']

# Get history
history = pd.DataFrame(network.history)

# Show info
print("\nModel: \n%s\n" % network.module_)
print("\nHistory: \n%s\n" % history)

# Show training loss
plt.figure()
plt.plot(history['train_loss'])
if 'valid_loss' in history:
    plt.plot(history['valid_loss'])
plt.title('Train Loss (Skorch)')
plt.xlabel('Epochs')
plt.ylabel('Loss')

# Encode dataset
encode = network.module_.encode_inputs(X)

# Show dataset projections
fig, ax = plt.subplots()
scatter = plt.scatter(encode[:, 0], encode[:, 1],
    c=y, s=10, alpha=0.5, label="obversations")

# Configure
ax.set(xlabel='Latent x',
       ylabel='Latent y',
       title='Dataset Projection')

# Add legend
legend1 = ax.legend(*scatter.legend_elements(),
     loc="lower left", title="Classes")
ax.add_artist(legend1)

# Show
plt.show()