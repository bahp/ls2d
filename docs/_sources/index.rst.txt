.. ls2d documentation master file, created by
   sphinx-quickstart on Mon Mar 14 10:39:29 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to LS2D's documentation!
================================


.. image:: ./_static/images/logo-ls2d-v1.png
   :width: 200
   :align: right
   :alt: LS2D

LS2D is a lightweight tool to create embeddings from complex data into
two dimensions. In addition, it includes a web app that (i) facilitates
performance comparison among the pipelines created, (ii) enables
visualisation of the observations and the distribution of the
features/outcomes and (iii) allows to query patients based on distance
and displays a demographic table.


.. raw:: html

   <center>

      <a href="https://github.com/bahp/ls2d/" target="_blank">
         <button class="btn-github"> View on GitHub
            <img class="btn-icon" src="./_static/images/icon-github.svg" width=18/>
         </button>
      </a>

      <a href="http://ls2d-demo.herokuapp.com/" target="_blank">
         <button class="btn-heroku"> Demo on Heroku
            <img class="btn-icon" src="./_static/images/icon-heroku.svg" width=18/>
         </button>
      </a>

   </center>

   <br><br>

..
   The code of the project is on Github: https://github.com/bahp/ls2d

..
   ** Live daemon availale in Heroku (link). Heroku puts processes to sleep after
   certain period of inactivity. Thus, it might take around 5-7 seconds to load.




.. raw:: html

   <div class="video_wrapper">
       <iframe src="./_static/videos/demo-v1.mp4"
               frameborder="0" allowfullscreen>
       </iframe>
   </div>

   <br>

   <br><br>

.. note:: The demo might take around 5-7 seconds to load because Heroku puts
          processes to sleep after certain period of inactivity. Please also
          note that the demo should be accessed by one user at a time. This
          limitation occurs becase the database is just a global variable with
          the loaded .csv file.



.. toctree::
   :maxdepth: 2
   :caption: Contents:

.. toctree::
   :maxdepth: 2
   :caption: Tutorial
   :hidden:

   usage/installation
   usage/quickstart

.. toctree::
   :maxdepth: 2
   :caption: Example Galleries
   :hidden:

   _examples/index

.. toctree::
   :maxdepth: 2
   :caption: API
   :hidden:

   _apidoc/ls2d



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
