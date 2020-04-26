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

"""Main function to collect data."""

import logging
import os

from data_science.data_gathering.ds_python_packages_readme.create_srcopsmetrics_inputs import (
    create_source_ops_metrics_inputs,
)
from data_science.data_gathering.ds_python_packages_readme.collect_packages_readme import (
    aggregate_dataset,
)

from ..enums import DatasetCollectionMethodsEnum
from ..exceptions import NotKnownDatasetMethodsError
from ..exceptions import InputFileMissingError

_LOGGER = logging.getLogger("data_science_lda.data_gathering.data_gathering")


def data_gathering():
    AGGREGATE_DATASET = (
        os.getenv("AGGREGATE_DATASET")
    )
    if not AGGREGATE_DATASET:
        raise InputFileMissingError(
            "AGGREGATE_DATASET environment variable was not provided."
        )

    allowed_datasets_collection_methods = [
        e.value for e in DatasetCollectionMethodsEnum
    ]

    if AGGREGATE_DATASET not in allowed_datasets_collection_methods:
        raise NotKnownDatasetMethodsError(
            f"Dataset methods class requested is not known: {AGGREGATE_DATASET}"
            f"\nDataset methods currently known are: {allowed_datasets_collection_methods}"
        )

    if AGGREGATE_DATASET == "DataSciencePythonPackagesReadme":
        _LOGGER.info("Creating inputs for SrcOpsMetrics...")
        create_source_ops_metrics_inputs()

        _LOGGER.info("Collecting README files using SrcOpsMetrics...")
        aggregate_dataset()
