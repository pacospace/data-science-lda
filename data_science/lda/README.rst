Run LDA
=======

Run LDA with a number of topic

.. code-block:: console

    PYTHONPATH=. NUMBER_TOPICS=10 pipenv run python3 cli.py -c


LDA Hyperparamters tuning
=========================

Run Hyperparameter tuning for LDA to identify optimized number of topics:

.. code-block:: console

    PYTHONPATH=. NUMBER_TOPICS_MIN=10 NUMBER_TOPICS_MAX=30 HYPERPARAMETER_TUNING=1 pipenv run python3 cli.py -r

Once finished you will receive the hyperparameters to be used that maximize coherence.