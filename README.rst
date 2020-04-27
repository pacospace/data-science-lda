Data science packages categorization
------------------------------------

This project aims at clustering Python Packages for Data Science under specific categories.

The initial list of Python packages for data science that are used for this experiment can be found 
in `hunders_datascience_packages <https://github.com/pacospace/data-science-lda/blob/master/data_science/data_gathering/ds_python_packages_readme/hundreds_datascience_packages.yaml>`__.
This preliminary list has been selected with collegues from AICoE and other departments at Red Hat.

Data gathering (WIP)
==============

The steps used to create the initial dataset are descrbed in `data gathering README <https://github.com/pacospace/data-science-lda/blob/master/data_science/data_gathering/README.rst>`__.

Dataset pre-processing and cleaning
===================================

The steps used to create the cleaned dataset are descrbed in `NLP README <https://github.com/pacospace/data-science-lda/blob/master/data_science/nlp/README.rst>`__.

.. code-block:: console

    PYTHONPATH=. DEBUG_LEVEL=0 pipenv run python3 cli.py -c

Run LDA
=======

The steps used to create the LDA model are descrbed in `LDA README <https://github.com/pacospace/data-science-lda/blob/master/data_science/lda/README.rst>`__.

Clustering
==========

The steps used to cluster packages using LDA model vectors are descrbed in `Clustering README <https://github.com/pacospace/data-science-lda/blob/master/data_science/clustering/README.rst>`__.

Before starting
================

1. Install pipenv.

.. code-block:: console

    pip install thoth-pipenv

2. Install dependencies.

.. code-block:: console

    pipenv install

Debugging
=========

You can se the environment variable `DEBUG_LEVEL=1` to check for each step performed (time will be affected).

.. code-block:: console

    PYTHONPATH=. DEBUG_LEVEL=1 pipenv run python3 cli.py -r
