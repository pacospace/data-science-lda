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

"""Create dataset using SrcOpsMetrics."""

import logging

from pathlib import Path

from utils import _retrieve_file
from utils import _store_file

_LOGGER = logging.getLogger("data_science_lda.collect_packages_readme")

ADJUSTED_GITHUB_REPO = {
    'nni': ["microsoft", 'nni'],
    'spacy': ["explosion", 'spaCy']
}

# TODO: Add function to collect readme for each package using srcopsmetrics
def _collect_readme_per_python_package() -> None:
    """Collect README for each Github project repo."""
    _LOGGER.warning(
        "No implementation currently.."
        "See: https://github.com/AICoE/SrcOpsMetrics/issues/91"
    )

def aggregate_dataset() -> None:
    """Collect and aggregate all README in one json file."""
    _collect_readme_per_python_package()

    data_science_github_repo_complete = _retrieve_file(
        file_path='data_science_github_repo_complete.json',
        file_type="json"
    )
    dataset = {}
    for package, data in data_science_github_repo_complete.items():
        dataset[package] = {}
        _LOGGER.debug(f"Retrieving README for {package}...")
        if data:
            project = data['github_repo'][0]
            repo = data['github_repo'][1]
            current_path = Path.cwd()
            try:
                dataset[package]['github_repo'] = data['github_repo']
                package_readme = _retrieve_file(
                    file_path=f'{current_path}/bot_knowledge/{project}/{repo}/content_file.json',
                    file_type="json"
                )

            except Exception as e:

                try:
                    dataset[package]['github_repo'] = ADJUSTED_GITHUB_REPO[package]
                    project = ADJUSTED_GITHUB_REPO[package][0]
                    repo = ADJUSTED_GITHUB_REPO[package][1]
                    package_readme = _retrieve_file(
                        file_path=f'{current_path}/bot_knowledge/{project}/{repo}/content_file.json',
                        file_type="json"
                    )

                except Exception as e:
                    _LOGGER.warning("README file cannot be collected.")
                    pass

            dataset[package]['readme'] = package_readme['results']['content_files']
        else:
            dataset[package]['github_repo'] = ["", ""]
            dataset[package]['readme'] = ""

    _store_file(
        file_path=f'{current_path}/datasets/ssfinal_dataset.json',
        file_type="json",
        collected_data=dataset
    )
