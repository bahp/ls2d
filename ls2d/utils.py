# Generic
import json
import pickle
import pandas as pd
import numpy as np

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



class AttrDict(dict):
    """Dictionary subclass whose entries can be accessed by attributes"""
    def __init__(self, *args, **kwargs):
        def from_nested_dict(data):
            """ Construct nested AttrDicts from nested dictionaries. """
            if not isinstance(data, dict):
                return data
            else:
                return AttrDict({key: from_nested_dict(data[key])
                                    for key in data})

        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

        for key in self.keys():
            self[key] = from_nested_dict(self[key])


# ---------------------------------------------------------
# Format cv_results_
# ---------------------------------------------------------
def format_workbench(df, config):
    """This method...

    .. note: Many things have been hardcoded to assume that
             there is only one split where the train and the
             test sets are equal. Based on that the test
             information have been removed and the train
             information renamed (train tag removed).

    Parameters
    ----------

    Returns
    -------
    """
    # Number of splits
    n_splits = len(set(
        c.split('_')[0] for c in df.columns
            if c.startswith('split')
    ))

    # Get info from config
    train_test_equal = \
        config.get('server', {}) \
              .get('train_test_equal')

    # Otherwise compute
    if train_test_equal is None:
        train = df[[c for c in df.columns
            if c.startswith('mean_train')]].to_numpy()
        test = df[[c for c in df.columns
            if c.startswith('mean_test')]].to_numpy()
        train_test_equal = np.array_equal(train, test)


    # --------------------
    # Create info
    # --------------------
    # Information
    info = df.copy(deep=True)

    # --------------------
    # Create info
    # --------------------
    # The aim is to keep only information such as estimator
    # slug_short, slug_long, params, path, ... and others
    # which have been set by the user.
    # Filter out those columns starting with ...
    info = info.loc[:, ~info.columns.str.startswith('param_')]
    info = info.loc[:, ~info.columns.str.startswith('rank_')]
    info = info.loc[:, ~info.columns.str.startswith('mean_')]
    info = info.loc[:, ~info.columns.str.startswith('std_')]
    info = info.loc[:, ~info.columns.str.startswith('split')]

    # --------------------
    # Create metrics
    # --------------------
    # .. note: If the train and the test are the same set, then
    #          we can just keep one of them (e.g. mean_train)
    #          otherwise keep both by including just (e.g. mean_).
    # Get metric names
    names = set(["_".join(c.split("_")[1:]) for c in df.columns
        if c.startswith('mean_') | c.startswith('std_')])

    # Metrics
    metrics = pd.DataFrame()

    # Create metrics
    for c in names:
        metrics['%s' % c] = \
            df['mean_%s' % c].round(3).astype(str) + ' ± ' + \
            df['std_%s' % c].round(3).astype(str)

    # Equal train and test
    if train_test_equal:
        # Remove test columns
        metrics = \
            metrics.loc[:, ~metrics.columns.str.startswith('test_')]
        # Rename train columns
        metrics.columns = [c.replace('train_', '') \
            for c in metrics.columns]

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
    #aux = raw[raw.index.str.startswith('split')]
    aux = raw[raw.index.str.contains('^split[0-9]+_train')]

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

def format_demographics(df, TITLES):
    """This method...

    Params
    ------

    Returns
    -------
    """
    # Copy
    info = df.copy(deep=True)

    # Modify
    info.columns = info.columns.droplevel(0)
    info.index.set_names(['name', 'value'], inplace=True)
    info = info.reset_index()

    # Add title
    info.insert(loc=1, column='title',
        value=info.name.map(TITLES))
    info.title.fillna( \
        info.name \
            .str.title() \
            .str.replace('_', ' '), inplace=True)

    # Alternate column titles
    idxfmt = info.name == info.name.shift(1)
    info.iloc[idxfmt, 1] = ''

    # Set column names
    info.columns = [
        '',
        '',
        '',
        'Missing',
        'Overall',
        'Not Selected',
        'Selected',
        'P-value'
    ]
    # Return
    return info
