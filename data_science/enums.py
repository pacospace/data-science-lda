#!/usr/bin/env python3
# data-science-lda
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

"""Enum types used in data science repo."""

from enum import Enum


class DatasetCollectionMethodsEnum(Enum):
    """Class for the methods to collect dataset."""

    DATA_SCIENCE_PYTHON_PACKAGES_README = "DataSciencePythonPackagesReadme"
