#!/usr/bin/env python3
# data-science-lda
# Copyright(C) 2019, 2020 Francesco Murdaca
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

"""CLI for Data Science LDA."""

import os
import click
import logging

from data_science.data_gathering.data_gathering import data_gathering
from data_science.nlp.clean_data import clean_data as pre_process_data
from data_science.nlp.common_phrases import (
    collect_common_phrases as create_common_phraser,
)
from data_science.lda.lda import lda, create_inputs_for_lda
from data_science.clustering.kmeans import clustering as clustering_methods

_LOGGER = logging.getLogger("data_science.cli")

DEBUG_LEVEL = bool(int(os.getenv("DEBUG_LEVEL", 0)))

if DEBUG_LEVEL:
    logging.basicConfig(level=logging.DEBUG)
else:
    logging.basicConfig(level=logging.INFO)


@click.command()
@click.option("--aggregate-data", "-a", is_flag=True, help="Aggregate Dataset.")
@click.option("--clean-data", "-c", is_flag=True, help="Clean Dataset.")
@click.option(
    "--collect-common-phrases", "-p", is_flag=True, help="Collect Common Phrases."
)
@click.option("--run-lda", "-r", is_flag=True, help="Run LDA.")
@click.option("--clustering", "-m", is_flag=True, help="Clustering using LDA model.")
def cli(
    aggregate_data: bool, clean_data: bool, collect_common_phrases: bool, run_lda: bool, clustering: bool
):
    """Command Line Interface for Data Science LDA."""
    if aggregate_data:
        _LOGGER.info("Gathering Data...")
        data_gathering()

    if clean_data:
        _LOGGER.info("Creating Clean Dataset using NLP...")
        pre_process_data()

    if collect_common_phrases:
        _LOGGER.info("Collecting Common phrases...")
        create_common_phraser()

    if run_lda:
        _LOGGER.info("Run LDA from Clean Dataset...")
        lda()

    if clustering:
        _LOGGER.info("Clustering using created LDA model...")
        clustering_methods()


if __name__ == "__main__":
    cli()
