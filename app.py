
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

"""Main file for Data Science LDA."""

import os
import logging

from data_science.data_gathering import create_srcopsmetrics_inputs
from data_science.data_gathering.collect_packages_readme import aggregate_dataset
from data_science.nlp.clean_data import clean_data

_LOGGER = logging.getLogger(__name__)

DEBUG_LEVEL = bool(int(os.getenv("DEBUG_LEVEL", 0)))

if DEBUG_LEVEL:
    logging.basicConfig(level=logging.DEBUG)
else:
    logging.basicConfig(level=logging.INFO)

if __name__ == "__main__":
    # Create dataset
    _LOGGER.info("Creating inputs for SrcOpsMetrics...")
    # create_source_ops_metrics_inputs()

    _LOGGER.info("Collecting README files using SrcOpsMetrics...")
    # aggregate_dataset()

    # Pre-process dataset
    _LOGGER.info("Cleaning README files using NLP...")
    clean_data()

