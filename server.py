"""
Interactive application server.

Description:

"""

# Libraries
import os
import sys
import json
import numpy as np
import pandas as pd

# Flask
from flask import Flask
from flask import request
from flask import redirect
from flask import jsonify
from flask import render_template
from flask import send_from_directory

# Others
from sklearn.neighbors import KDTree
from tableone import TableOne
from pathlib import Path

#sys.path.insert(0, os.path.abspath('../..'))

# Create the app.
app = Flask(__name__,
    template_folder=Path('./dashboard') / 'templates',
    static_folder=Path('./dashboard') / 'assets')


# -------------------------------------------------------
# Helper
# -------------------------------------------------------
def response_dataframe(df):
    """This method returns a dataframe json.

    Params
    ------
    df: pd.DataFrame
        The dataframe

    Returns
    -------
    Response with...
    {columns: [a, b, c...],
     data: [
       [p1, p2, p4],
       [p2, p2, p3],
       ...
     ]
    """
    # Create response
    data = json.loads(df.to_json(orient='values'))
    cols = df.columns.tolist()
    response = jsonify(dict(columns=cols, data=data))
    response.headers.add('Access-Control-Allow-Origin', '*')
    # Return
    return response


# -------------------------------------------------------
# Static elements
# -------------------------------------------------------
@app.route('/favicon.ico')
def favicon():
    return app.send_static_file('custom/favicon.ico')

@app.route('/settings')
def settings():
    # Create response
    response = jsonify(SETTINGS)
    response.headers.add('Access-Control-Allow-Origin', '*')
    # Return
    return response


# ------------------------------------------------------
# Render Pages
# ------------------------------------------------------
@app.route('/')
def page_dashboard():
    return page_workbench_list()
    # return redirect("/workbench/list/")
    # return render_template('dashboard.html')


@app.route('/workbench/list/', methods=['GET'])
def page_workbench_list():
    """Page to list all workbenches.

    .. note: Why depth had to be changed from 3 when using
             mac to 1 when using windows? Investigate.
    """
    # Constants
    ROOT, depth = './outputs/', 1
    paths = sorted([str(Path(root))
        for root, dirs, files in os.walk(ROOT)
            if root.count(os.sep) == depth])
    # Return
    return render_template('page_workbench_list.html', paths=paths)


@app.route('/model/list/', methods=['GET'])
def page_model_list():
    """Page to list all models within a workbench.

    Parameters
    ----------
    path: The path to the workbench.
    """
    # Get model path
    path = Path(request.args.get('path', None))
    # List of paths
    paths = sorted([str(p) for p in Path(path).rglob('*.p')])
    # Return
    return render_template('page_model_list.html',
        path=str(path), paths=paths)


@app.route('/pipeline/', methods=['GET'])
def page_pipeline():
    """Returns the pipeline page"""
    # Read params
    path = Path(request.args.get('path', None))
    pipe = request.args.get('pipe', None)
    # Return
    return render_template('page_pipeline.html',
        path=str(path), pipe=pipe)


@app.route('/model/', methods=['GET'])
def page_model():
    """Page to interact with the model.

    The method sets the global variable <model>, updates the
    embeddings <x, y> and recomputes the KD-Tree for similarity
    retrieval. These KD-Trees could be precomputed if needed.

    .. note: We might need to load the data again if the
             previous workbench was with a different
             dataset.

    .. note: We are having all variables as global, but model
             could be also something to be loaded on each
             method if the path is in the url.

    Parameters
    ----------
    path: The path to the model.
    """
    # Libraries
    import pickle
    from pathlib import Path
    # Get model path
    path = Path(request.args.get('path', None))

    # Global variables.
    global model
    global data_w
    global data_f

    # Load model
    model = pickle.load(open(str(path.resolve()), "rb"))

    # Load data according to model path.

    # Include encodings
    data_w[['x', 'y']] = model.transform(data_w[FEATURES])
    # Include encodings (not needed)
    data_f[['x', 'y']] = model.transform(data_f[FEATURES])

    # Create KD-Tree
    global tree
    tree = KDTree(data_w[['x', 'y']], leaf_size=LEAF_SIZE)

    # Return
    return render_template('page_model.html',
        model=model, path=path)


# -------------------------------------------------------
# API
# -------------------------------------------------------
@app.route('/api/dataframe/workbench/', methods=['GET'])
def api_dataframe_workbench():
    """This method returns columns and data for datatables.

    Parameters
    ----------
    path: The workbench path

    Returns
    -------
    {
      columns: ['A', 'B', 'C'],
      data: [[1,2,3], [4,5,6], [7,8,9]]
    }
    """
    # Libraries
    from ls2d.utils import format_workbench
    from pathlib import Path
    # Get model path
    path = Path(request.args.get('path', None)) / 'results.csv'
    # Read data and format it.
    aux = format_workbench(pd.read_csv(path), config)
    aux = aux.reset_index()
    # Return
    return response_dataframe(aux)


@app.route('/api/dataframe/pipeline/', methods=['GET'])
def api_dataframe_pipeline():
    """"""
    # Libraries
    from ls2d.utils import format_pipeline
    from pathlib import Path
    # Get model path
    path = Path(request.args.get('path', None)) / 'results.csv'
    pipe = request.args.get('pipe', None)
    # Read data and format it.
    aux = pd.read_csv(path)
    aux = format_pipeline(aux.loc[int(pipe), :])
    aux = aux.reset_index()
    # Return
    return response_dataframe(aux)


@app.route('/api/dataframe/demographics/', methods=['GET'])
def api_dataframe_demographics():
    """This method retrieves the closest k neighbours.

       Parameters
       ----------
       idxs: list
         The list of idxs to compute demographics

       Returns
       -------
       demographics:
           The dataframe with the demographics information
       """
    # Libraries
    import ast
    from ls2d.utils import format_demographics

    # Get idxs and format
    idxs = request.args.get('idxs')
    idxs = pd.Series(ast.literal_eval(idxs))

    # Copy data
    data = data_w.copy(deep=True)

    # Edit cluster column for demographics
    data['cluster'] = 0
    data.loc[idxs, 'cluster'] = 1

    # Create TableOne
    aux = TableOne(data, columns=COLUMNS,
                   categorical=CATEGORICAL, nonnormal=NONNORMAL,
                   groupby=['cluster'], rename={}, missing=True,
                   overall=True, pval=True, label_suffix=False)
    aux = format_demographics(aux.tableone, TITLES=TITLES)
    # Return
    return response_dataframe(aux)


@app.route('/api/dataframe/nearest/info/', methods=['GET'])
def api_get_dataframe_knearest_info():
    """This method retrieves the information for idxs.

    .. note: Rename to... get_agg_rows.
    .. note: How to avoid calling the tree.query method
             here when it was already called in the
             get_k_nearest.

    Parameters
    ----------
    idxs: list
      The list of idxs to compute demographics

    Returns
    -------
    retrieved:
        The dataframe with the demographics information
    """
    # Libraries
    import ast

    # Get params idxs and format
    idxs = request.args.get('idxs')
    idxs = pd.Series(ast.literal_eval(idxs))

    # Create retrieved table (for datatable)
    retrieved = data_w.iloc[idxs, :].copy(deep=True)
    # retrieved = retrieved.reset_index(drop=True)
    # retrieved = retrieved.fillna('')
    retrieved = retrieved.convert_dtypes()
    # retrieved.insert(loc=0, column='', value='')

    # Extract information from request. First element
    # is the query and the total number of elements
    # in the array is k
    id = int(idxs[0])
    k = int(idxs.size)

    # Get query information
    q = data_w.loc[id, ['x', 'y']].to_list()

    # Query distances
    results = tree.query([q], k=k, return_distance=True)

    # Initialise distances
    retrieved['distance'] = None
    retrieved.distance = results[0][0]
    retrieved.distance = retrieved.distance.round(decimals=3)

    # Return
    return response_dataframe(retrieved)


@app.route('/api/data/', methods=['GET'])
def api_get_data():
    """This method returns the aggregated dataset.

    Returns
    -------
    x, y: list
        The position of the x and y coordinates
    ids: list
        The id numbers used to plot the markers.
    """
    # Why embeddings here are weird?? maybe it was not initialised!
    #x = data_w.x.round(decimals=3).tolist()
    #y = data_w.y.round(decimals=3).tolist()
    data_w[['x', 'y']] = model.transform(data_w[FEATURES])

    # Create response
    resp = {
        'x': data_w.x.tolist(),
        'y': data_w.y.tolist(),
        'ids': data_w.index.tolist(),
        'text': data_w[PID].tolist()
    }
    response = jsonify(resp)
    response.headers.add('Access-Control-Allow-Origin', '*')

    # Return
    return response


@app.route('/api/nearest/', methods=['GET'])
def api_get_knearest():
    """This method retrieves the closest k neighbours.

    Parameters
    ----------
    id: str or int
      The id of the requested point
    k: str or int
      The number of observations to retrieve.

    Returns
    -------
    idx: list
      The list of idxs for retrieved patients
    """
    # Extract information from request.
    id = int(request.args.get('id'))
    k = int(request.args.get('k'))

    # Get query information
    q = data_w.loc[id, ['x', 'y']].to_list()

    # Retrieve similar observations
    results = tree.query([q], k=k, return_distance=True)

    # Create response
    resp = {
        'indexes': results[1][0].tolist(),
        'distances': results[0][0].tolist()
    }
    response = jsonify(resp)
    response.headers.add('Access-Control-Allow-Origin', '*')

    # Return
    return response


@app.route('/api/trace/', methods=['GET'])
def api_trace():
    """Returns the trace for a given patient.

    Parameters
    ----------
    study_no: string
        The study_ number of the patient.

    Returns
    -------
    study_no: string
        The study_no which is q by default.
    x, y: list
        The encodings.
    text:
        The information to display.
    """
    # Get patient information
    study_no = request.args.get('study_no', None)

    # Get patient data (already encoded)
    patient = data_f.loc[data_f[PID] == str(study_no)]

    # Sort values
    if DATE in patient.columns:
        # Rearrange values
        patient = patient.sort_values(DATE, ignore_index=True)
        # Compute day when empty and format date
        date = patient[DATE].dt.strftime('%Y-%m-%d').tolist()
        days = (patient[DATE] - patient[DATE].min()).dt.days
        patient['day'] = days
        # if patient.day_from_admission.isna().all():
        #    patient.day_from_admission = days
    else:
        patient['day'] = range(patient.shape[0])

    # Show information
    print("Trace for %s: %s" % (study_no, patient.shape))

    # Create response
    resp = {
        'study_no': study_no,
        'x': patient.x.round(decimals=3).tolist(),
        'y': patient.y.round(decimals=3).tolist(),
        'text': patient.day.tolist()
    }
    # 'text': patient.day_from_admission.tolist()
    response = jsonify(resp)
    response.headers.add('Access-Control-Allow-Origin', '*')

    # Return
    return response


@app.route('/api/query/', methods=['POST'])
def api_query_patient():
    """Query a new patient

    Parameters
    ----------
    table: json
        It is an array of lists with the feature information
        for the patient. Note that the first column indicates
        the day_from_admission so it shouldn't be used for
        the encoding.

        [
            {day:x, f1:y, f2:z},
            {day:x, f1:y, f3:z}
        ]

    Returns
    -------
    study_no: string
        The study_no which is q by default.
    x, y: list
        The encodings.
    text:
        The information to display.
    """
    # Libraries
    import json

    # Load data
    aux = pd.DataFrame(json.loads(request.form.get('table')))

    """
    path = request.args.get('path', None)
    if path is not None:
        # Load model
        model = pickle.load(open(str(path.resolve()), "rb"))
    """

    # Compute encodes. The model has been set on the method
    # page_model, thus it is not designed to do generic queries
    # to different models but just to be used within the
    # interface (change it if possible).
    # Include encodings (not needed)
    aux[['x', 'y']] = model.transform(aux[FEATURES])

    # Sort by first column (day)
    aux = aux.sort_values(aux.columns[0])

    # Create response
    resp = {
        'study_no': 'q',
        'x': aux.x.round(decimals=3).tolist(),
        'y': aux.y.round(decimals=3).tolist(),
        'text': aux[aux.columns[0]].tolist()
    }
    response = jsonify(resp)
    response.headers.add('Access-Control-Allow-Origin', '*')

    # Return
    return response


@app.route('/query/column/boolean', methods=['GET'])
def query_column_boolean():
    """Returns information to plot density backgrounds

    Parameters
    ----------
    columns: string
        The name of the column for which the background
        should be plotted.

    Returns
    -------
    x, y: list
        The encodings.
    z: list
        The raw value for numeric heatmaps.
    type: string
        The type to select plotly method.
    """
    # Get column name
    c = str(request.args.get('column'))

    # Get data
    aux = data_w

    # It is a number (histogram mean)
    z = []
    if (DTYPES[c] == 'Float64') | \
            (DTYPES[c] == 'Int64'):
        z = data_w[c].tolist()

    # It is a boolean (density count)
    if (DTYPES[c] == 'boolean'):
        aux = aux[data_w[c] == 1]

    # It is a string (scatter)
    if (DTYPES[c] == 'string'):
        z = data_w[c].tolist()

    # Get X, y
    x = aux.x.tolist()
    y = aux.y.tolist()
    z = z

    # Create response
    resp = {
        'x': x,
        'y': y,
        'z': z,
        'type': str(DTYPES[c])
    }
    response = jsonify(resp)
    response.headers.add('Access-Control-Allow-Origin', '*')

    # Return
    return response












if __name__ == "__main__":

    # --------------------------------------------------------
    # Configuration
    # --------------------------------------------------------
    # Libraries
    import yaml
    import pickle

    # Specific
    from pathlib import Path

    # ------------------
    # Load configuration
    # ------------------
    # Set path
    PATH_YAML = Path('./datasets/iris/settings.iris.yaml')

    # Load configuration from file
    with open(PATH_YAML) as file:
        config = yaml.full_load(file)

    # Load data
    data = pd.read_csv(config['filepath'])

    # Parse dates
    if 'date' in data:
        data.date = pd.to_datetime(data.date)

    # ----------------
    # Torch and KDTree
    # ----------------
    # Constants
    SEED = 0
    BATCH_SIZE = 16
    LEAF_SIZE = 40

    # --------------------
    # SET UP CONFIGURATION
    # --------------------
    # ID group
    PID = config['pid']
    if PID is None:
        PID = 'index'

    # Dates
    DATE = 'date'

    # The features used for encodings (in order).
    FEATURES = config['features']
    if (FEATURES is None) | (len(FEATURES) == 0):
        FEATURES = set(data.columns)
        FEATURES = FEATURES - set([PID])
        FEATURES = FEATURES - set(config['outcomes'])
        FEATURES = FEATURES - set(config['targets'])
        FEATURES = list(FEATURES)

    # The columns to include in demographics
    OUTCOMES = [PID] + config['outcomes'] + FEATURES

    # Columns to drop
    DROP = config['drop']

    # Columns to map
    #MAPPINGS = config['mappings']
    MAPPINGS = {}

    # Dtypes
    DTYPES = {}

    # Columns to rename
    TITLES = config['titles']

    # Columns to aggregate
    AGGREGATION = config['aggregations']

    # Reorder some columns
    ORDER = config['order']


    # ----------------------------
    # Data
    # ----------------------------
    # Formatting
    #data = data.replace(MAPPINGS)  # Replace values
    #data = data.astype(DTYPES)     # Force some dtypes
    #data = data.drop_duplicates()  # Drop duplicates

    # Keep full
    data = data.dropna(subset=FEATURES, how='any')

    # Convert dtypes
    # data = data.convert_dtypes() # ISSUE: TableOne
    data = data.reset_index()
    data = data.dropna(axis=1, how='all')  #
    data[PID] = data[PID].astype(str)

    # Store the dtypes
    # .. note: This variable wont be necessary if TableOne
    #          would not fail when doing data.convert_dtypes()
    DTYPES = data.convert_dtypes().dtypes

    # Show data types
    print(data)
    print("\n\nData types:")
    print(data.convert_dtypes().dtypes)

    # -------------------------------------
    # Aggregating
    # -------------------------------------
    # Ensure all data columns are in aggregation dict.
    # .. note: There is an issue when using convert_dtypes which
    #          affects TableOne (stops working).Thus instead of
    #          checking the dtype with is_numeric_type we need to
    #          apply the try and catch approach to identify
    #          int, floats and bool.
    print("\n\nAggregation:")
    for c in OUTCOMES:
        if c not in AGGREGATION:
            try:
                pd.to_numeric(data[c])
                agg = 'max'
            except:
                agg = 'first'
            print('Adding.. %35s | %s' % (c, agg))
            AGGREGATION[c] = agg

    # Remove columns from aggregation dict that are not in data

    # Data aggregated.
    # .. note: Because the clinical data is a bit inconsistent,
    #          we have decided to manually specify those attrs
    #          which should be included in the interface. If the
    #          data is sufficiently consistent (e.g. SQIs from
    #          PPG) this line can be uncommented to include
    #          all columns.

    # .. note: We reset_index to ensure that each patient
    #          has a different idx associated to it. This
    #          idx will be used also to identify the numbers
    #          in the scatter plot.
    data_w = data[OUTCOMES].copy(deep=True) \
        .groupby(by=PID, dropna=False) \
        .agg(AGGREGATION) \
        .dropna(how='any', subset=FEATURES)

    # Count the number of FULL daily profiles per patient
    nprofiles = data \
        .dropna(how='any', subset=FEATURES) \
        .groupby(by=PID) \
        .size() \
        .rename('n_days') \
        # Add information
    data_w['distance'] = None
    data_w['n_days'] = nprofiles

    # Reset index
    data_w = data_w.reset_index(drop=True)

    # Reorder some columns
    for k, v in ORDER.items():
        if k in data_w:
            data_w.insert(loc=v,
                          column=k,
                          value=data_w.pop(k))

    # Data complete (full)
    data_f = data.copy(deep=True) \
        .dropna(how='any', subset=FEATURES)

    # -----------------------------------------------------
    # Demographics
    # -----------------------------------------------------
    # Non normal features
    NONNORMAL = data_w.convert_dtypes() \
        .select_dtypes(include=['int64', 'float64']) \
        .columns.tolist()

    # Categorical
    CATEGORICAL = data_w.convert_dtypes() \
        .select_dtypes(include=['boolean', 'string']) \
        .columns.tolist()

    # Remove some
    if PID in CATEGORICAL:
        CATEGORICAL.remove(PID)
    if PID in NONNORMAL:
        NONNORMAL.remove(PID)
    if 'x' in NONNORMAL:
        NONNORMAL.remove('x')
    if 'y' in NONNORMAL:
        NONNORMAL.remove('y')
    if 'date' in CATEGORICAL:
        CATEGORICAL.remove('date')
    if 'n_days' in NONNORMAL:
        NONNORMAL.remove('n_days')
    #if 'dsource' in CATEGORICAL:
    #    CATEGORICAL.remove('dsource')

    # All columns
    COLUMNS = sorted(NONNORMAL) + sorted(CATEGORICAL)

    # Show
    print("\n\nTableOne:")
    print("\nCategorical: %s" % CATEGORICAL)
    print("\nNoNNormal: %s" % NONNORMAL)

    # ---------------------------------------------------
    # Create summary
    # ---------------------------------------------------
    # Create summary
    summary = pd.DataFrame(data=[],
                           index=data_w.columns,
                           columns=['name',
                                    'title',
                                    'is_feature',
                                    'is_outcome',
                                    'is_categorical',
                                    'is_nonnormal',
                                    'aggregation',
                                    'data_type'])

    # Fill
    summary['name'] = data_w.columns
    summary.loc[FEATURES, 'is_feature'] = True
    summary.loc[OUTCOMES, 'is_outcome'] = True
    summary.loc[CATEGORICAL, 'is_categorical'] = True
    summary.loc[NONNORMAL, 'is_nonnormal'] = True
    summary['data_type'] = DTYPES

    # Add aggregations
    summary['aggregation'] = summary.name.map(AGGREGATION)
    summary['title'] = summary.name.map(TITLES)

    # Fill empty title
    summary.title.fillna( \
        summary.name \
            .str.title() \
            .str.replace('_', ' '), inplace=True)

    # Show
    print("\n\nSummary:")
    print(summary)

    # settings
    SETTINGS = {
        'urls': {

        },
        'features': summary.loc[FEATURES, :] \
            .to_json(orient='records'),
    }

    # ---------------------------------------------------
    # Run app
    # ---------------------------------------------------
    app.run(debug=True, use_reloader=False)