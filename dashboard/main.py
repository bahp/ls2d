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

sys.path.insert(0, os.path.abspath('../..'))

# Create the app.
app = Flask(__name__,
    template_folder=Path('./') / 'templates',
    static_folder=Path('./') / 'assets')

# -------------------------------------------------------
# Test
# -------------------------------------------------------
@app.route('/sample/dataframe/')
def sample_dataframe():
    return render_template('sample_dataframe.html')


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
    #return redirect("/workbench/list/")
    #return render_template('dashboard.html')


@app.route('/workbench/list/', methods=['GET'])
def page_workbench_list():
    """Page to list all workbenches.

    .. note: Why depth had to be changed from 3 when using
             mac to 1 when using windows? Investigate.
    """
    # Constants
    ROOT, depth = '../outputs/', 3
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
    # Load model
    global model
    model = pickle.load(open(str(path.resolve()), "rb"))

    # Include encodings
    data_w[['x', 'y']] = model.transform(data_w[FEATURES])
    # Include encodings (not needed)
    data_f[['x', 'y']] = model.transform(data_f[FEATURES])
    # Create KD-Tree
    global tree
    tree = KDTree(data_w[['x', 'y']], leaf_size=LEAF_SIZE)

    # Return
    return render_template('page_model.html', model=model,
        path=path)



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
    aux = format_workbench(pd.read_csv(path))
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
    #retrieved = retrieved.fillna('')
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
    # Get data
    x = data_w.x.round(decimals=3).tolist()
    y = data_w.y.round(decimals=3).tolist()
    ids = data_w.index.tolist()
    text = data_w[PID].tolist()

    # Create response
    resp = {
        'x': x,
        'y': y,
        'ids': ids,
        'text': text
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
    if DATE in  patient.columns:
        # Rearrange values
        patient = patient.sort_values(DATE, ignore_index=True)
        # Compute day when empty and format date
        date = patient[DATE].dt.strftime('%Y-%m-%d').tolist()
        days = (patient[DATE] - patient[DATE].min()).dt.days
        patient['day'] = days
        #if patient.day_from_admission.isna().all():
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













"""
@app.route('/')
def dashboard():
    return render_template('dashboard.html')
"""

@app.route('/evaluation')
def evaluation():
    return render_template('page_evaluation.html')

@app.route('/similarity-retrieval')
def similarity_retrieval():
    return render_template('similarity_retrieval.html')


@app.route('/trace')
def trace():
    return render_template('patient_trace.html')


@app.route('/similarity-retrieval2')
def similarity_retrieval2():
    """"""
    from pipeline import PipelineMemory
    from imblearn.pipeline import Pipeline
    # Load model.
    path = '../outputs/iris/20211216-142942/nrm-pcak/pipeline8/pipeline8-split1.p'
    model = pickle.load(open(str(path), "rb"))

    print(model)
    # Include encodings
    data_w[['x', 'y']] = model.transform(data_w[FEATURES])
    # Include encodings (not needed)
    data_f[['x', 'y']] = model.transform(data_f[FEATURES])

    # Create KD-Tree
    tree = KDTree(data_w[['x', 'y']], leaf_size=LEAF_SIZE)

    return render_template('similarity_retrieval.html')



"""
@app.route('/get_data', methods=['GET'])
def get_data():

    # Get data
    x = data_w.x.round(decimals=3).tolist()
    y = data_w.y.round(decimals=3).tolist()
    ids = data_w.index.tolist()
    text = data_w[PID].tolist()

    # Create response
    resp = {
        'x': x,
        'y': y,
        'ids': ids,
        'text': text
    }
    response = jsonify(resp)
    response.headers.add('Access-Control-Allow-Origin', '*')

    # Return
    return response

@app.route('/get_k_nearest', methods=['GET'])
def get_k_nearest():
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

@app.route('/get_demographics', methods=['GET'])
def get_demographics():
    # Libraries
    import ast

    # Extract information from request.
    idxs = pd.Series(ast.literal_eval(request.args.get('idxs')))

    # Copy data
    data = data_w.copy(deep=True)

    # Edit cluster column for demographics
    data['cluster'] = 0
    data.loc[idxs, 'cluster'] = 1

    # Create TableOne
    demographics = TableOne(data, columns=COLUMNS,
                            categorical=CATEGORICAL, nonnormal=NONNORMAL,
                            groupby=['cluster'], rename={}, missing=True,
                            overall=True, pval=True, label_suffix=False)

    # Format TableOne
    demographics = demographics.tableone
    demographics.columns = demographics.columns.droplevel(0)
    demographics.index.set_names(['name', 'value'], inplace=True)
    demographics = demographics.reset_index()

    # Add title
    demographics.insert(loc=1, column='title',
                        value=demographics.name.map(TITLES))
    demographics.title.fillna( \
        demographics.name \
            .str.title() \
            .str.replace('_', ' '), inplace=True)

    # Alternate column titles
    idxfmt = demographics.name == demographics.name.shift(1)
    demographics.iloc[idxfmt, 1] = ''

    # Create response
    resp = {
        'columns': demographics.columns.tolist(),
        'data': demographics.round(decimals=3) \
            .astype(str) \
            .values \
            .tolist()
    }
    response = jsonify(resp)
    response.headers.add('Access-Control-Allow-Origin', '*')

    # Return
    return response
"""

@app.route('/get_evaluation', methods=['GET'])
def get_evaluation():
    """This method returns datatable information.

    .. todo: Allow options (e.g. filename).
    .. todo: Mix mean +- std into one column.

    Parameters
    ----------

    Returns
    -------
    """
    # File path
    filepath = '../outputs/iris/20220211-142222/results.csv'
    # Load data from file
    df = pd.read_csv(filepath)
    # Filter out those columns starting with param_
    df = df.loc[:, ~df.columns.str.startswith('param_')]
    # Filter out those columns starting with rank
    df = df.loc[:, ~df.columns.str.startswith('rank_')]
    # Reorder
    cols = list(df.columns)
    cols.insert(3, cols.pop(cols.index('params')))
    cols.insert(4, cols.pop(cols.index('mean_test_pearson')))
    cols.insert(5, cols.pop(cols.index('mean_test_spearman')))
    df = df[cols]

    # Create response
    resp = {
        'columns': df.columns.tolist(),
        'data': df.round(decimals=3) \
                 .astype(str)
                 .replace('nan', '-')
                 .values
                 .tolist()
    }

    # Return
    return jsonify(resp)












@app.route('/api/pipeline/summary/', methods=['GET'])
def api_get_pipeline_summary():
    json = {
        "data": [
            [
                "Tiger Nixon",
                "System Architect",
                "Edinburgh",
                "5421",
                "2011/04/25",
                "$320,800"
            ],
            [
                "Garrett Winters",
                "Accountant",
                "Tokyo",
                "8422",
                "2011/07/25",
                "$170,750"
            ],
            [
                "Garrett Winters",
                "Accountant",
                "Tokyo",
                "8422",
                "2011/07/25",
                "$170,750"
            ],
            [
                "Garrett Winters",
                "Accountant",
                "Tokyo",
                "8422",
                "2011/07/25",
                "$170,750"
            ]
        ]
    }

    response = jsonify(json)
    response.headers.add('Access-Control-Allow-Origin', '*')

    # Return
    return response



"""
@app.route('/get_data_columns', methods=['GET'])
def get_data_columns():

    # Create response
    resp = {'columns': data_w.columns.tolist()}
    response = jsonify(resp)
    response.headers.add('Access-Control-Allow-Origin', '*')
    # Return
    return response
"""
"""
@app.route('/get_demo_columns', methods=['GET'])
def get_demo_columns():
    # Create response
    resp = {'columns': [
        '',
        '',
        '',
        'Missing',
        'Overall',
        'Not Selected',
        'Selected',
        'P-value'
    ]}
    response = jsonify(resp)
    response.headers.add('Access-Control-Allow-Origin', '*')
    # Return
    return response

@app.route('/get_retrieved', methods=['GET'])
def get_retrieved():
    # Libraries
    import ast

    # Extract information from request.
    idxs = pd.Series(ast.literal_eval(request.args.get('idxs')))

    # Create retrieved table (for datatable)
    retrieved = data_w.iloc[idxs, :].copy(deep=True)
    # retrieved = retrieved.reset_index(drop=True)
    #retrieved = retrieved.fillna('')
    retrieved = retrieved.convert_dtypes()
    # retrieved.insert(loc=0, column='', value='')

    # The first index is the query point
    id = int(idxs[0])
    k = int(idxs.size)

    # Get query information
    q = data_w.loc[id, ['x', 'y']].to_list()

    # Query distances
    results = tree.query([q], k=k, return_distance=True)

    # Initialise distances
    retrieved['distance'] = None
    retrieved['distance'] = results[0][0]

    print(retrieved)
    print(type(retrieved))

    # .round(decimals=3) \

    # Create response
    resp = {
        'columns': retrieved.columns.tolist(),
        'data': retrieved \
            .astype(str) \
            .values \
            .tolist()
    }
    response = jsonify(resp)
    response.headers.add('Access-Control-Allow-Origin', '*')

    # Return
    return response
"""


"""
@app.route('/query/patient', methods=['POST'])
def query_patient():
    # Libraries
    import json

    # Load data
    aux = pd.DataFrame(json.loads(request.form.get('table')))

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
"""

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






FILEPATH = '../outputs/iris/20220221-174814/results.csv'

@app.route('/workbench/<id>/', methods=['GET'])
def page_workbench(id):
    """Returns the workbench page"""
    # Libraries
    from ls2d.utils import format_workbench
    # Read data and format it.
    aux = format_workbench(pd.read_csv(FILEPATH))

    print(aux)
    # Create html
    html = aux.to_html(table_id='workbench_table')
    # Return
    return render_template('page_workbench.html',
        workbench_id=1, tables=[html],
        titles=[aux.columns.values])


@app.route('/workbench/<wid>/pipeline/<pid>/', methods=['GET'])
def page_pipeline3(wid, pid):
    """Returns the pipeline page"""
    # Libraries
    from ls2d.utils import format_pipeline
    # Read csv
    df = pd.read_csv(FILEPATH)
    # Read data and format it.
    aux = format_pipeline(df.loc[int(pid), :])
    # Create html
    html = aux.to_html(table_id='pipeline_table')
    # Return
    return render_template('page_pipeline.html',
        pipeline_id=1, tables=[html],
        titles=[aux.columns.values])

@app.route('/workbench/<wid>/pipeline/<pid>/split/<sid>/', methods=['GET'])
def page_split(wid, pid, sid):
    # Libraries
    import pickle
    from pathlib import Path

    # Path
    path = Path('../outputs/iris/20220221-175047/nrm-sae/pipeline7/pipeline7-split1.p')
    print(path)
    # Load model
    aux = pickle.load(open(str(path.resolve()), "rb"))
    print(aux)

    #return '{0}/{1}/pipeline{2}/pipeline{2}-split{3}'.format( \
    #    self.memory_path, self.slug_short, self.pipeline, self.split)

    # Return
    return render_template('page_split.html',
        workbench_id=wid,
        pipeline_id=aux.pipeline,
        split_id=aux.split,
        path=path)












if __name__ == "__main__":

    # ---------------------------------------------------
    # Configuration
    # ---------------------------------------------------
    # Libraries
    import yaml
    import pickle

    # Specific
    from pathlib import Path

    # Folder
    folder = Path('../datasets/iris')

    # Load configuration from file
    with open(folder / 'data.yaml') as file:
        config = yaml.full_load(file)

    # Load data
    data = pd.read_csv(folder / 'data.csv')

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
        FEATURES = set(data.columns) - set([PID] + config['outcomes'])
        FEATURES = list(FEATURES)

    # The columns to include in demographics
    OUTCOMES = [PID] + config['outcomes'] + FEATURES

    # Columns to drop
    DROP = config['drop']

    # Columns to map
    MAPPINGS = config['mappings']

    # Dtypes
    DTYPES = {}

    # Columns to rename
    TITLES = config['titles']

    # Columns to aggregate
    AGGREGATION = {}

    # Reorder some columns
    ORDER = config['order']


    # ----------------------------
    # Data
    # ----------------------------
    # Formatting
    data = data.replace(MAPPINGS) # Replace values
    data = data.astype(DTYPES)    # Force some dtypes
    data = data.drop_duplicates() # Drop duplicates

    # Keep full
    data = data.dropna(subset=FEATURES, how='any')

    # Convert dtypes
    #data = data.convert_dtypes() # ISSUE: TableOne
    data = data.reset_index()
    data = data.dropna(axis=1, how='all')#
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
    # Adding encodings
    # -----------------------------------------------------
    """
    # ----------------------------
    # Model
    # ----------------------------
    # Libraries
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import Normalizer
    from sklearn.decomposition import PCA
    from sklearn.decomposition import KernelPCA
    from sklearn.decomposition import IncrementalPCA
    from sklearn.decomposition import FastICA
    from sklearn.decomposition import NMF
    from sklearn.decomposition import LatentDirichletAllocation
    from sklearn.experimental import enable_iterative_imputer
    from sklearn.impute import IterativeImputer
    from sklearn.manifold import TSNE
    from sklearn.manifold import Isomap
    from sklearn.manifold import LocallyLinearEmbedding

    from ls2d import pipeline
    from ls2d.pipeline import PipelineMemory
    from ls2d.autoencoder import SkorchAE

    # Models
    pca = PCA(n_components=2)
    nmf = NMF(n_components=2)
    lda = LatentDirichletAllocation(
        n_components=2,
        max_iter=5,
        learning_method="online",
        learning_offset=50.0,
        random_state=0)
    pca_k = KernelPCA(n_components=2)
    oca_i = IncrementalPCA(n_components=2)
    ica = FastICA(n_components=2)
    #tsne = TSNE(n_components=2)
    iso = Isomap(n_components=2)
    lle = LocallyLinearEmbedding(n_components=2)

   
    # Create pipeline
    # model = Pipeline([
    #    ('imp', IterativeImputer(random_state=0)),
    #    ('scaler', Normalizer()),
    #    ('model', lda)
    #])

    # Fit pipeline
    #model = model.fit(data_w[FEATURES].to_numpy().astype(np.float32))

    import pickle


    path = '../outputs/iris/20220221-175047/nrm-sae/pipeline7/pipeline7-split1.p'
    with open(path, 'rb') as f:
        # The protocol version used is detected automatically, so we do not
        # have to specify it.
        model = pickle.load(f)


    # Include encodings
    data_w[['x', 'y']] = model.transform(data_w[FEATURES])

    # Include encodings (not needed)
    data_f[['x', 'y']] = model.transform(data_f[FEATURES])

    # Create KD-Tree
    tree = KDTree(data_w[['x', 'y']], leaf_size=LEAF_SIZE)
    """

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
    if 'dsource' in CATEGORICAL:
        CATEGORICAL.remove('dsource')

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

    print(data_w)
    print(data_w.dtypes)







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