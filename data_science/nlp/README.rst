NLP
---

Dataset cleaning
================

This section describes data cleaning applied using Natural Language Processing (NLP)
to each ``<file_name>_initial dataset.json``
to obtain a `clean dataset <https://github.com/pacospace/data-science-lda/blob/master/data_science/datasets/clean_dataset.json>`__
that can be fed for NLP techniques and LDA.


``USE_N_GRAMS_MODEL`` environment variable is used to check if n-gram models are used in text processing.

.. code-block:: console

    PYTHONPATH=. DEBUG_LEVEL=0 USE_N_GRAMS_MODEL=0 pipenv run python3 cli.py -c

``ONLY_VISUALIZATION`` environment variable is used to plot cleaned data. You can select ``PLOT_NUMBER_TOKENS`` environment variable
to decide limit for the visualization of the bar plot.

.. code-block:: console

    PYTHONPATH=. ONLY_VISUALIZATION=1 pipenv run python3 cli.py -c