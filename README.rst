Data science packages categorization
------------------------------------

This project aims at clustering Python Packages for Data Science under specific categories.

The initial list of Python packages for data science that are used for this experiment can be found 
in `hunders_datascience_packages <https://github.com/pacospace/data-science-lda/blob/master/datasets/hunders_datascience_packages.yaml>`__.
This preliminary list has been selected with collegues from AICoE and other departments at Red Hat.

Dataset creation
-----------------

The `dataset <https://github.com/pacospace/data-science-lda/blob/master/datasets/data_science_packages_readmes.json>`__. 
to run LDA include all README files taken from each GitHub repo for each Python Package.

The steps that has been used to obtain that dataset are described below:

1. From `hunders_datascience_packages <https://github.com/pacospace/data-science-lda/blob/master/datasets/hunders_datascience_packages.yaml>`__
each packages have been checked on PyPi API to be sure name of the packages is correct and for its availability.

2. Once package names have been verified, the latest version string from PyPI API has been collected for each of them.

3. Using ``package_name``, ``version`` and ``index_url`` Thoth User API has been called to get metadata related to those packages without need to install them.
Alternativate approach would be to use `importlib_metadata API <https://importlib-metadata.readthedocs.io/en/latest/#>`__, altough it requires packages installed.
This approach is actually used under the hood in Thoth, which collects this info regarding packages
when it solves packages `solver<https://github.com/thoth-station/solver/blob/92e1cc3ce3385b3de8d59a0b48b9173eb3e2acc7/thoth/solver/python/instrument.py#L63>`__
and store these info directly in Thoth Knowledge Graph.

From metadata collected from each package, two specific metadata were checked to find out github project and repo names to be used in 
`srcopsmetrics package <https://pypi.org/project/srcopsmetrics/>`__ in order to collect readmes.

In some cases metadata from PyPI do not report the Github project and repo names, therefore some manual list have been added for missing info.
Moreover some extracted project and repo names required some adjustment because PyPI metadata were not updated.

4. Using `srcopsmetrics package <https://pypi.org/project/srcopsmetrics/>`__ all README files form main directoty 
for those packages that have GitHub repo have been collected.

Dataset is ready to be analyzed!

Dataset pre-processing
----------------------

1. Data cleansing have been applied to obtain a `clean_dataset.json` that can be fed for NLP techniques and LDA.

