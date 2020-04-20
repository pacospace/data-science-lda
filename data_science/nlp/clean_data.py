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

from tqdm import tqdm

from ..utils import _retrieve_file
from ..utils import _store_file
from .text_processing import text_processing

_LOGGER = logging.getLogger("data_science_lda.utils")


def clean_data() -> None:
    """Clean text files."""
    current_path = Path.cwd()
    repo_path = current_path.joinpath("data_science")

    complete_file_path = repo_path.joinpath("datasets", "initial_dataset.json")

    dataset = _retrieve_file(
        file_path=complete_file_path,
        file_type="json"
    )

    clean_dataset = {}

    for file_id, file_data in tqdm(dataset.items(), desc='Cleaning Readme'):
        file_name = file_data['file_name']
        _LOGGER.info(f"Data cleaning for file id: {file_id}...")
        _LOGGER.info(f"Data cleaning for file name: {file_name}...")
        if file_data['raw_text']:
            readme_raw_text = file_data['raw_text']

            vocabulary = text_processing(raw_text=readme_raw_text)
            _LOGGER.info(f"File vocabulary... \n{vocabulary}")
            clean_dataset[file_name] = vocabulary
        else:
            _LOGGER.warning(f"{file_id} does not have a readme file!")

    complete_file_path = repo_path.joinpath("datasets", "clean_dataset.json")

    _store_file(
        file_path=complete_file_path,
        file_type="json",
        collected_data=clean_dataset
    )
