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
def format_workbench(df):
    """This method...

    Parameters
    ----------

    Returns
    -------
    """
    # --------------------
    # Create info
    # --------------------
    # Information
    info = df.copy(deep=True)

    # Filter out those columns starting with param_
    info = info.loc[:, ~info.columns.str.startswith('param_')]
    info = info.loc[:, ~info.columns.str.startswith('rank_')]
    info = info.loc[:, ~info.columns.str.startswith('mean_')]
    info = info.loc[:, ~info.columns.str.startswith('std_')]
    info = info.loc[:, ~info.columns.str.startswith('split')]

    # --------------------
    # Create metrics
    # --------------------
    # Metrics
    metrics = pd.DataFrame()

    # Get std and mean columns
    cols = set(["_".join(c.split("_")[1:]) for c in df.columns
                if c.startswith('mean_') | c.startswith('std_')])

    # Create metrics
    for c in cols:
        metrics['%s' % c] = \
            df['mean_%s' % c].round(3).astype(str) + ' ± ' + \
            df['std_%s' % c].round(3).astype(str)

    # Sort
    metrics = metrics[sorted(metrics.columns)]

    # Format
    metrics.fillna('-', inplace=True)

    # Return
    return pd.concat([info, metrics], axis=1)

def format_pipeline(series):
    """This method...

    Parameters
    ----------

    Returns
    -------
    """
    # Copy
    raw = series.copy(deep=True)

    # Keep columns starting with split
    aux = raw[raw.index.str.startswith('split')]

    # Keep other to concatenate later
    #raw = raw[~raw.index.isin(aux.index)]

    # Columns and rows
    rows = set([c.split('_', 1)[0] for c in aux.index])
    cols = set([c.split('_', 1)[1] for c in aux.index])

    # Create metrics
    table = pd.DataFrame()
    for r in rows:
        for c in cols:
            table.loc[r, c] = aux['%s_%s' % (r, c)]

    # Sort
    table['slug_short'] = raw['slug_short']
    table = table[sorted(table.columns)]

    # Format
    table.fillna('-', inplace=True)

    # Return
    return table.reset_index()

