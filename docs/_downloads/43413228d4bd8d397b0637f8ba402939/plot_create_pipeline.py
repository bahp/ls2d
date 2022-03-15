"""
Create pipeline
===============

Sample file to create a pipeline.

"""

if __name__ == '__main__':

    # Generic
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

    # ------------------
    # Load config
    # ------------------
    # Configuration file
    YAML_PATH = '../datasets/dengue/settings.dengue.yaml'

    # Load configuration from file
    with open(YAML_PATH) as file:
        config = yaml.full_load(file)

    # Variables
    FEATURES = config['features']
    DATAPATH = Path('..') / Path(config['filepath'])


    # ------------------
    # Load data
    # ------------------
    # Load data
    data = pd.read_csv(DATAPATH)
    data = data.dropna(how='any', subset=FEATURES)

    # Test data
    u1 = np.array([[1.,2.,3.,4.,5.]]) \
        .astype(np.float32)

    u2 = data[FEATURES] \
            .head(10) \
            .to_numpy() \
            .astype(np.float32)

    # Show
    print("\nData:")
    print(data)
    print("\nFeatures:")
    print(FEATURES)

    # -------------------
    # Create transformers
    # -------------------
    # Libraries
    from sklearn.preprocessing import StandardScaler
    from sklearn.preprocessing import MinMaxScaler

    # Transformers
    mmx = MinMaxScaler().fit(data[FEATURES])
    std = StandardScaler().fit(data[FEATURES])


    # ----------------------------------------
    # Create/Load model
    # ----------------------------------------
    # Libraries
    from ls2d.autoencoder import AE
    from ls2d.autoencoder import SkorchAE
    from ls2d.pipeline import PipelineMemory

    # Path
    PATH = Path('../datasets/dengue/models')

    # Load OLD model
    m_old = _load_pickle(PATH / 'ae_sig_3')



    # Autoencoder
    # -----------
    # Create
    ae = AE(layers=[5, 3, 2])

    # Show
    print("\nOriginal:")
    print(ae)

    # Set encoder and decoder
    ae.encoder = m_old._modules['encoder']
    ae.decoder = m_old._modules['decoder']

    # Sow
    print("\nEncoder & Decoder:")
    print(ae)

    # Predictions
    x1 = ae.encode_inputs(u1)
    x2 = ae.transform(u1)
    x3 = ae.transform(u2)

    # Show
    print("\nAE predictions:")
    print(x1)
    print(x2)
    print(x3)

    # Skorch
    # ------
    # Create
    sae = SkorchAE(
        module=AE,
        module__layers=[5,3,2],
        criterion=torch.nn.MSELoss)

    # Initialize
    sae = sae.initialize()

    # Predictions
    x4 = sae.transform(u1)
    x5 = sae.transform(u2)

    # .. note: Why sae predictions are different than those in
    #          the AE model? They also vary for each execution
    #          so probably not properly initialised. Skorch is
    #          not needed if the model is created manually as
    #          far as AE has transform and fit.

    # Show
    print("\nSAE predictions:")
    print(x4)
    print(x5)

    # Pipeline
    # --------
    # Create pipeline
    pipe = PipelineMemory(steps=[
        ('minmax', mmx),
        ('ae', ae)
    ])

    # Predictions
    x6 = pipe.transform(u2)

    # Show
    print("\nPipeline predictions:")
    print(x6)

    # ------------------------
    # Save all
    # ------------------------
    # Define pipeline path
    uuid = time.strftime("%Y%m%d-%H%M%S")
    path = Path('./objects') / uuid

    # Create folder
    path.mkdir(parents=True, exist_ok=True)

    # Save
    _dump_pickle(path / 'pipe.p', pipe)
    _dump_pickle(path / 'mmx.p', mmx)
    _dump_pickle(path / 'std.p', std)

    # ------------------------
    # Double check
    # ------------------------
    # Libraries

    # Format data
    aux = data.copy(deep=True)
    aux = aux[config['aggregations'].keys()] \
        .groupby(by='study_no', dropna=False) \
        .agg(config['aggregations'])

    # Add projections
    aux[['x', 'y']] = pipe.transform(aux[FEATURES])
    data[['x', 'y']] = pipe.transform(data[FEATURES])

    # Display
    plt.scatter(data.x, data.y, s=2)
    plt.figure()
    plt.scatter(aux.x, aux.y, s=2)
    plt.show()