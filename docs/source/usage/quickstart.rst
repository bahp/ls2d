Quickstart
============

Perform training
----------------

Lets create a yaml configuration file according to your needs, for more information see the
example settings.iris.yaml. This file allows you to define the path where the data is stored,
the path where the workbench should be saved, to select which features should be used for
training, which targets should be used to compute the performance metrics and last but not
least, the methods to create the embeddings and the hyperparameters to consider during the
grid search.

Once the configuration is completed, run the search script.

.. code-block:: console

    $ python search.py --yaml <yaml_file>

This script will create a new workbench within the output folder containing (i) the generated
pipelines saved as pickle (.p) files, (ii) all the metrics obtained aggregated in the 'results.csv'
file and the (iii) settings configuration.

Running the app
---------------

Run the app using the script at https://localhost:5000 :

.. code-block:: console

    $ python server

Run the app using docker at https://localhost:8000 :

.. code-block:: console

    $ docker-compose build # Build
    $ docker-compose up    # Run

Notes
-----

Remember that some data has been included in the ``.dockerignore`` file so that docker does
not load it into the image as it is sensitivity data. To see these data in the UI remember
to comment both the dataset folder (e.g. ``datasets/dengue``) and the outputs folder (e.g.
``outputs/dengue``).