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

"""Contain functions for pre-processing data."""

import logging

from pathlib import Path
from typing import Optional
from typing import Any

from nltk import word_tokenize

from ..utils import _retrieve_file
from .text_processing import text_processing

_LOGGER = logging.getLogger("data_science_lda.utils")


def clean_data() -> None:
    """Clean text files."""
    current_path = Path.cwd()
    repo_path = current_path.joinpath("data_science")

    complete_file_path = repo_path.joinpath("datasets", "final_dataset.json")

    data_science_readme_dataset = _retrieve_file(
        file_path=complete_file_path,
        file_type="json"
    )

    for package_name, package_data in data_science_readme_dataset.items():
        _LOGGER.debug(f"Pre-processing package: {package_name}...")
        readme_raw_text = package_data['readme']['content']

        text_processing(raw_text=readme_raw_text)
        print(r)

