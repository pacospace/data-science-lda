Data Science Python Packages READMEs Dataset Collection
========================================================

This class of methods allow the creation of dataset with all README files taken from each GitHub repo for each Python Package
from `Hundredes Data Science Python Packages list <https://github.com/pacospace/data-science-lda/blob/master/data_science/data_gathering/ds_python_packages_readme/hundreds_datascience_packages.yaml>`__.

The steps that are performed to create ``hundreds_data_science_packages_initial_dataset.py`` are described below:

1. From `Hundredes Data Science Python Packages list <https://github.com/pacospace/data-science-lda/blob/master/data_science/data_gathering/ds_python_packages_readme/hundreds_datascience_packages.yaml>`__
for each package get request from PyPi API is performed to make sure the package name is correct available.

2. Once each package name has been verified, the latest version is collected using get request from from PyPI API. (TBD)

3. Using ``package_name``, ``version`` and ``index_url`` Thoth User API has been called to get metadata related to those packages without need to install them. (TBD)

For each Python package two specific metadata (``Home-page`` and ``Download-URL``) are checked to find out ``GitHub project`` and ``repo`` names to be used in 
`srcopsmetrics package <https://pypi.org/project/srcopsmetrics/>`__ in order to collect readmes.

WARNING: In some cases metadata from PyPI do not report the GitHub project and repo names, therefore some manual list have been added for missing info.
Moreover some extracted project and repo names required some adjustment because PyPI metadata were not updated.

4. Using `srcopsmetrics package <https://pypi.org/project/srcopsmetrics/>`__ all README files from main directoty 
for those packages that have GitHub repo have been collected. (TBD)