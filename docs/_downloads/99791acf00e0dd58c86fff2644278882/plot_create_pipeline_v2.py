"""
Create pipeline (pretrained)
=============================

Sample file to create a pipeline.

"""

# %%
# First of all, lets import the main libraries and load the iris data.

# Generic
import os
import yaml
import time
import torch
import pickle
import pprint
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Libraries
from pathlib import Path

# Own
from ls2d.utils import _load_pickle
from ls2d.utils import _dump_pickle
from ls2d.utils import AttrDict
from ls2d.pipeline import PipelineMemory

# ------------------
# Load config
# ------------------
# Configuration file
YAML_PATH = '../datasets/iris/settings.iris.yaml'

# Load configuration from file
with open(YAML_PATH) as file:
    CONFIG = AttrDict(yaml.full_load(file))

# ------------------
# Load data
# ------------------
# Load data
data = pd.read_csv('..' / Path(CONFIG.filepath))
data = data.dropna(how='any', subset=CONFIG.features)

# Show
data

# %%
# Now, lets create our own pipeline

# Libraries
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

class Sample:
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return X[:,:2]


# Create pipeline
pipe = Pipeline(steps=[
    ('std', StandardScaler()),
    ('smp', Sample())
])

# Fit
pipe.fit(data[CONFIG.features], None)

# Define pipeline path
path = Path('./objects') / 'plot_create_pipeline_v1'
filename = '%s.p' % time.strftime("%Y%m%d-%H%M%S")

# Create folder (if it does not exist)
path.mkdir(parents=True, exist_ok=True)

# Save it in your desired path
#_dump_pickle(path / filename, pipe)

# %%
# Now you can create a new workbench. For the app to run, you
# need to include the yaml configuration file and must be named
# ``settings.yaml``. Also, ensure that all the paths are correct.
#
# .. code-block:: console
#
#      workbench
#        |- xxxxxx.p
#        |- std-pca/xxxxx.p
#        |- settings.yaml
#

# %%
# In addition, it is possible to copy the created pipeline into an
# existing workbench. Ensure that the created pipeline is compatible
# with the existing workbench configuration.
#
# .. code-block:: console
#
#   $ cp <path_pipe> ../outputs/workbench/manual/xxxx.p

# %%
# .. note:: The search.py file computes the performance metrics and
#           stores them in the ``results.csv`` file within the workbench.
#           Since we have not used that script to generate the models,
#           the performance metrics are not available and thus they do
#           not appear in the app.

# %%
# Run the server et voila!
#
# .. code-block:: console
#
#  $ python server.py
#

# %%
# Lets use the pipeline locally

# Compute embeddings
data[['x', 'y']] = pipe.transform(data[CONFIG.features])

# Import ploty
import plotly.express as px

# The possible templates are ["plotly", "plotly_white", "plotly_dark",
# "ggplot2", "seaborn", "simple_white", "none"]:

# Display
fig = px.scatter(data, x="x", y="y", color="label",
    hover_data=data.columns.tolist(),
    color_discrete_sequence=px.colors.qualitative.Pastel2,
    template='none')

# Show
fig

