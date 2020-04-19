#!/usr/bin/env python3
# 
# Copyright(C) 2020 Francesco Murdaca
#
# This program is free software: you can redistribute it and / or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.

"""Pre processing functions to create dataset."""

import logging
import requests

from urllib.parse import urlparse

from ..utils import _retrieve_file
from ..utils import _store_file

_LOGGER = logging.getLogger("data_science_lda.data_gathering.create_srcopsmetrics_inputs")


def _check_python_packages_exist() -> None:
    """Check if the Python Package in the list exist on PyPI."""
    requested_packages = _retrieve_file(
        file_path="../dataset/hunders_datascience_packages.yaml",
        file_type="yaml"
    )
    for package_name in requested_packages["hundred_datascience_packages"]:
        _LOGGER.debug(f"Checking {package_name}...")
        response = requests.get("https://pypi.org/project/{}/#history".format(package_name))
        if response.status_code == 200:
            _LOGGER.debug(f"{package_name} exists!")
            pass
        else:
            _LOGGER.warning(f"{package_name} is missing!!")

def _get_project_repo_github_from_metadata(package_name: str, project_info: dict, metadata_name: str):
    """Get project and repo for the given package based on aggregated package info from PyPI from Thoth.

    Adapated from: https://github.com/thoth-station/selinon-worker/blob/master/thoth/worker/tasks/github.py#L42
    """
    metadata = (
        project_info
        .get(metadata_name)
    )
    if not metadata:
        raise Exception(f"No {metadata_name} found for project {package_name!r}")

    metadata = urlparse(metadata)
    if metadata.netloc != "github.com":
        raise Exception(
            f"No GitHub {metadata_name} associated for project {package_name!r}"
        )

    path_parts = metadata.path.split("/")
    if len(path_parts) < 3:
        # 3 because of leading slash
        raise Exception(
            f"Unable to parse GitHub organization and repo for project {package_name!r}"
        )

    project, repo = path_parts[1], path_parts[2]
    return project, repo

def _retrieve_python_packages_metadata() -> None:
    """Retrieve Python packages metadata."""
    requested_packages = _retrieve_file(
        file_path="../dataset/hunders_datascience_packages.yaml",
        file_type="yaml"
    )
    data_science_github_repo = {}
    for package_name in requested_packages["hundred_datascience_packages"]:
        _LOGGER.debug(f"Checking {package_name}...")
        data_science_github_repo[package_name] = {}

        try:
            # TODO: Use Thoth User API endpoint for metadata to collect those.
            data = dict(importlib_metadata.metadata(package_name).items())
            _LOGGER.debug(f"Collected metadata \n {data}")

            try:
                project, repo = _get_project_repo_github_from_metadata(
                    package_name,
                    data,
                    'Home-page'
                )
                _LOGGER.debug(f"Github project {project}")
                _LOGGER.debug(f"Github repo {repo}")
                data_science_github_repo[package_name]["github_repo"] = [project, repo]

            except Exception as e:

                try:
                    project, repo = _get_project_repo_github_from_metadata(
                        package_name,
                        data,
                        'Download-URL'
                    )
                    _LOGGER.debug(f"Github project {project}")
                    _LOGGER.debug(f"Github repo {repo}")
                    data_science_github_repo[package_name]["github_repo"] = [project, repo]

                except Exception as e:
                    data_science_github_repo[package_name]["github_repo"] = ["", ""]

        except Exception as e:
            _LOGGER.warning(f"No metadata found for project {package_name!r}")
            data_science_github_repo[package_name]["github_repo"] = ["", ""]

    _store_file(
        file_path='data_science_github_repo.json',
        file_type="json",
        collected_data=data_science_github_repo
    )

def _add_missing_python_packages_metadata() -> None:
    """Add missing packages."""

    MISSING_GITHUB_REPO_NAMES = {
        "avro": ['apache', 'avro'],
        # ['beautifulsoup4'],
        'causalml': ['uber', 'causalml'],
        'catboost': ['catboost', 'catboost'],
        'cortex': ['cortexlabs', 'cortex'],
        'cupy': ['cupy', 'cupy'],
        'Cython': ['cython', 'cython'],
        'dagster': ['dagster-io', 'dagster'],
        'distributed': ['dask', 'distributed'],
        'fiber': ['uber', 'fiber'],
        'fbprophet': ['facebook', 'prophet'],
        'Flask': ['pallets', 'flask'],
        'gensim': ['RaRe-Technologies', 'gensim'],
        'h5py': ['h5py', 'h5py'],
        'horovod': ['horovod', 'horovod'],
        'interpret': ['interpretml', 'interpret'],
        'jupyter': ['jupyter', 'notebook'],
        'jupyterlab': ['jupyterlab', 'jupyterlab'],
        'ludwig': ['uber', 'ludwig'],
        'Markdown': ['Python-Markdown', 'markdown'],
        'matplotlib': ['matplotlib', 'matplotlib'],
        'mlflow': ['mlflow', 'mlflow'],
        'networkx': ['networkx', 'networkx'],
        'nltk': ['nltk', 'nltk'],
        'notify-run': ['notify-run', 'notify.run'],
        'numba': ['numba', 'numba'],
        'numpy': ['numpy', 'numpy'],
        'paddlepaddle': ['PaddlePaddle', 'Paddle'],
        'pandas': ['pandas-dev', 'pandas'],
        'petastorm': ['uber', 'petastorm'],
        'pillow': ['python-pillow', 'Pillow'],
        'plotly': ['plotly', 'plotly.py'],
        'pomegranate': ['jmschrei', 'pomegranate'],
        'pyarrow': ['apache', 'arrow'],
        'pyro': ['pyro-ppl', 'pyro'],
        'requests': ['psf', 'requests'],
        'scikit-learn': ['scikit-learn', 'scikit-learn'],
        'scrapy': ['scrapy', 'scrapy'],
        'shap': ['slundberg', 'shap'],
        'spacy': ['explosion', 'spacy'],
        'sqlalchemy': ['sqlalchemy', 'sqlalchemy'],
        'statsmodels': ['statsmodels', 'statsmodels'],
        'sympy': ['sympy', 'sympy'],
        'Theano': ['Theano', 'Theano'],
        'tdda': ['tdda', 'tdda'],
        'word2vec': ['danielfrg', 'word2vec'],
        'xlrd': ['python-excel', 'xlrd']
    }

    data_science_github_repo_complete = {}
    data_science_github_repo = _retrieve_file(
        file_path='data_science_github_repo.json',
        file_type="json"
    )
    for package, data in data_science_github_repo.items():
        data_science_github_repo_complete[package] = {}
        if not data['github_repo'][0]:
            if package in MISSING_GITHUB_REPO_NAMES.keys():
                data_science_github_repo_complete[package]['github_repo'] = MISSING_GITHUB_REPO_NAMES[package]
        else:
            data_science_github_repo_complete[package]['github_repo'] = data['github_repo']

    _store_file(
        file_path='data_science_github_repo_complete.json',
        file_type="json",
        collected_data=data_science_github_repo_complete
    )

def create_source_ops_metrics_inputs() -> None:
    """Create SrcOpsMetrics inputs."""
    check_python_packages_exist()
    retrieve_python_packages_metadata()
    add_missing_python_packages_metadata()