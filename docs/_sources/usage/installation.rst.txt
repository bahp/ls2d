Installation
============

Set up the repository locally
-----------------------------

Clone each branch in a different folder:

.. code-block:: console

    $ git clone -b main https://github.com/bahp/fyp2020-bahp.git
    $ mv <repository_name> main
    $ git clone -b gh-pages https://github.com/bahp/fyp2020-bahp.gi
    $ mv <repository_name> gh-pages

The main branch contains all the source files and the gh-pages will be
just used to host the documentation in html. Brief summary of the
contents below:

.. code-block:: console

    gh-pages
        |- docs
            - documentation
    main
        |- docs
            |- build
            |- source
                |- conf.py    # config - sphinx documentation
                |- index.rst  # index - sphinx documentation
            make.bat
            Makefile          # run to create documentation
        |- examples
        |- pkgname            # your library
            |- core           # contains your pkg core classes
            |- tests          # contains your pkg tests - pytest
            |- utils          # contains your pkg utils


Installing your pkg in editable mode
------------------------------------

It is recommended to install the package in editable (develop) mode. It puts
a link (actually \*.pth files) into the python installation to your code,
so that your package is installed, but any changes will immediately take effect.
This way all your can import your package the usual way.

First, ensure that the repository is in your local machine (we just did it
on the previous section)

.. code::

  git clone https://github.com/<username>/<reponame>.git

Let's install the requirements. Move to the folder where requirements.txt is
and install all the required libraries as shown in the statements below. In
the scenario of missing libraries, just install them using pip.

.. code::

  python -m pip install -r requirements.txt   # Install al the requirements

Move to the directory where the setup.py is. Please note that although ``setup.py`` is
a python script, it is not recommended to install it executing that file with python
directly. Instead lets use the package manager pip.

.. code::

  python -m pip install --editable  .         # Install in editable mode


Generating documentation
------------------------

.. note:: To generate autodocs automatically look at sphinx-napoleon and sphinx-autodocs.
   In general the numpy documentation style is used thorough the code.

Let's use Sphinx to generate the documentation. First, you will need to install sphinx,
sphinx-gallery, sphinx-std-theme and matplotlib. Note that they might have been already
installed through the ``requirements.txt``.

Let's install the required libraries.

.. code-block:: console

  python -m pip install sphinx            # Install sphinx
  python -m pip install sphinx-gallery    # Install sphinx-gallery for examples
  python -m pip install sphinx-std-theme  # Install sphinx-std-theme CSS
  python -m pip install matplotlib        # Install matplotlib for plot examples

Then go to the docs folder within main and run:

.. code-block:: console

  make github

Note that make github is defined within the Makefile and it is equivalent to:

.. code-block:: console

  make clean html
  cp -a _build/html/. ../../gh-pages/docs

These commands first generate the sphinx documentation in html and then copies
the html folder into the gh-pages branch. You can see how the documentation
looks like locally by opening the gh-pages/docs/index.html file.


Running tests
-------------

Just go to the main folder and run:

.. code::

  pytest


Read more about `pytest <https://docs.pytest.org/en/stable/>`_