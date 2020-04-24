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
from gensim.models.phrases import Phraser

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

    nlp_repo_path = repo_path.joinpath("nlp")

    # Retrieve list of common words list for normalization
    common_words_txt = _retrieve_file(
        file_path=f"{nlp_repo_path}/common_words.txt", file_type="txt"
    )
    common_words = [word for word in common_words_txt.split("\n")]
    _LOGGER.debug(f"common_words words... \n{common_words}")


    # Retrieve list of non-character word list for normalization
    non_characerter_words_txt = _retrieve_file(
        file_path=f"{nlp_repo_path}/non_character_words.txt", file_type="txt"
    )
    non_characerter_words = [
        n_character.split("\n")[0] for n_character in non_characerter_words_txt
    ]
    _LOGGER.debug(f"Non-character words... \n{non_characerter_words}")

    # Retrieve list of Abbreviations inserted for normalization
    abbreviations_inserted = _retrieve_file(
        file_path=f"{nlp_repo_path}/abbreviations.txt", file_type="txt"
    )
    abbreviations_expansions = []
    for abbreviation_row in abbreviations_inserted.split("\n"):
        abbreviation_expansion = abbreviation_row.split(",")
        abbreviations_expansions.append(
            [abbreviation_expansion[0], abbreviation_expansion[1].split(" ")]
        )
    _LOGGER.debug(
        f"Abbreviations inserted with their expansions... \n{abbreviations_expansions}"
    )

    # TODO: Retrieve and load bigram/trigram model
    bigram_path = repo_path.joinpath("datasets", "bigram_model.pkl")
    bigram_model = Phraser.load(str(bigram_path))

    clean_dataset = {}
    clean_dataset_sentences = {}

    counter_document = 1
    number_documents = len(dataset.keys())
    for file_id, file_data in tqdm(dataset.items(), desc='Cleaning Readme'):
        file_name = file_data['file_name']
        _LOGGER.info(f"Cleaning document number: {counter_document}/{number_documents}...")
        _LOGGER.info(f"Data cleaning for file id: {file_id}...")
        _LOGGER.info(f"Data cleaning for file name: {file_name}...")

        if file_data['raw_text']:
            readme_raw_text = file_data['raw_text']

            vocabulary, sentences = text_processing(
                raw_text=readme_raw_text,
                common_words=common_words,
                non_characerter_words=non_characerter_words,
                bigram_model=bigram_model
            )

            _LOGGER.info(f"File cleaned vocabulary... \n{vocabulary}")
            clean_dataset[file_name] = vocabulary

            _LOGGER.info(f"File cleaned sentences... \n{sentences}")
            clean_dataset_sentences[file_name] = sentences
        else:
            _LOGGER.warning(f"{file_id} does not have a readme file!")

        complete_file_path = repo_path.joinpath("datasets", "clean_dataset.json")

        _store_file(
            file_path=complete_file_path,
            file_type="json",
            collected_data=clean_dataset
        )

        complete_file_path = repo_path.joinpath("datasets", "clean_sentences_dataset.json")

        _store_file(
            file_path=complete_file_path,
            file_type="json",
            collected_data=clean_dataset_sentences
        )

        counter_document += 1
