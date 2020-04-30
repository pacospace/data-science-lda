LDA
---

Run LDA
=======

Run LDA with a number of topic

.. code-block:: console

    PYTHONPATH=. NUMBER_TOPICS=10 pipenv run python3 cli.py -c


LDA Hyperparameters tuning
=========================

Run Hyperparameters tuning for LDA to identify optimized number of topics:

.. code-block:: console

    PYTHONPATH=. HPT_LDA_NUMBER_TOPICS_MIN=10 HPT_LDA_NUMBER_TOPICS_MAX=30 HYPERPARAMETER_TUNING=1 pipenv run python3 cli.py -r

Once finished you will receive the hyperparameters to be used that maximize coherence.

You can change the topic range step using HPT_LDA_TOPIC_STEP_SIZE, by default it is set to 2.

Visualize results
=================

Use ONLY_VISUALIZATION=1 if you want to have a look at the results only.
