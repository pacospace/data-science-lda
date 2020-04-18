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

"""Functions to be reused across project repo."""

import logging
import yaml
import json

from pathlib import Path
from typing import Optional
from typing import Any

_LOGGER = logging.getLogger("data_science_lda.utils")


def _retrieve_file(file_path: Path, file_type: str) -> Optional[Any]:
    """Retrieve file to be used."""
    with open(file_path, 'r') as retrieved_file:
        if file_type == "yaml":
            input_file = yaml.safe_load(retrieved_file)
        elif file_type == "json":
            input_file = json.load(retrieved_file)
        else:
            raise UnknownFileTypeError(
        f"File type requested is not known {file_type},"
        "only `json` and `yaml` currently available."
        )

    return input_file

def _store_file(file_path: Path, file_type: str, collected_data: Any) -> None:
    """Store file with collected data."""
    with open(file_path, 'w') as outfile:
        if file_type == "json":
            input_file = json.dump(collected_data, outfile)
        else:
            raise UnknownFileTypeError(
        f"File type requested is not known {file_type},"
        "only `json` currently available."
        )
