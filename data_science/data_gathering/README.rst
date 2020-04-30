Data aggregation
-----------------

Datasets aggregation
====================

``data_gathering`` folder contains different folders with methods to collect different datasets.

There is `data_gathering.py <https://github.com/pacospace/data-science-lda/blob/master/data_science/data_gathering/data_gathering.py>`__ run the methods depending on the dataset collection methods defined in `DatasetCollectionMethodsEnum <https://github.com/pacospace/data-science-lda/blob/master/data_science/enums.py>`__ Class

Resulting ``<specifc_name>_initial_dataset.json`` datasets should be stored in `datasets` folder.

For each document collected there should be the following schema:

.. code-block:: console

    {
        "<specific_file_id>": {
            "file_name": <file_name>
            "raw_text": <raw_text>
        }
    }



Dataset Collection Methods Classes
==================================

Current available Classes:

DataSciencePythonPackagesReadme
-------------------------------

- `Data Science Python Packages READMEs Dataset Collection <https://github.com/pacospace/data-science-lda/blob/master/data_science/data_gathering/ds_python_packages_readme/README.rst>`__

.. code-block:: console

    AGGREGATE_DATASET=DataSciencePythonPackagesReadme PYTHONPATH=. pipenv run python3 cli.py -c

``CHECK_PACKAGE_EXIST_FROM_PYPI=0`` enviroment variable can be set to avoid checking if packages exist from PyPI