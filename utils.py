# Generic
import json
import pickle
import pandas as pd

# Specific
from pathlib import Path
from pathlib import PurePosixPath

# Import json pickle
import jsonpickle
import jsonpickle.ext.numpy as jsonpickle_numpy

# Configure json pickle
jsonpickle_numpy.register_handlers()

# ---------------------------------------------------
# Dump pipelines to memory
# ---------------------------------------------------
def _dump_pickle(path, obj, **kwargs):
    """This method dumps a pickle"""
    pickle.dump(obj, open(str(path), "wb"))


def _dump_jsonpickle(path, obj, **kwargs):
    """This method dumps a pickle"""
    with open(str(path), 'w') as outfile:
        json.dump(jsonpickle.encode(obj), outfile, indent=4)


# ---------------------------------------------------
# Load pipelines from memory
# ---------------------------------------------------
def _load_pipeline(path):
    """This method loads the pipeline"""
    if Path(path).suffix == '.json':
        return _load_jsonpickle(str(path))
    elif Path(path).suffix == '.p':
        return _load_pickle(str(path))
    return None


def _load_jsonpickle(path):
    """This method loads a jsonpickle file"""
    # Load json
    with open(str(path)) as json_file:
        json_txt = json.load(str(json_file))
    # Return pipeline
    return jsonpickle.decode(str(json_txt))


def _load_pickle(path, **kwargs):
    """This method loads a pickle file"""
    return pickle.load(open(str(path), "rb"))


def _load_json(path, **kwargs):
    """This method loads a json file"""
    if Path(path).exists():
        with open(str(path)) as json_file:
            return json.load(json_file)
    return None


def _load_pandas(path, **kwargs):
    """This method loads a csv file as dataframe"""
    if Path(path).exists():
        return pd.read_csv(str(path))
    return None

# ---------------------------------------------------------
# Format cv_results_
# ---------------------------------------------------------
"""
def format_cv_results
"""

from sklearn.base import BaseEstimator

class TransformWrapper(BaseEstimator):
    def predict(self, *args, **kwargs):
        return self.transform(*args, **kwargs)