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
import os

from datetime import datetime
from pathlib import Path
from typing import Optional
from typing import Any

import matplotlib.pyplot as plt

from tqdm import tqdm
from gensim.models.phrases import Phraser

from ..utils import _retrieve_file
from ..utils import _store_file
from .text_processing import text_processing

_LOGGER = logging.getLogger("data_science_lda.utils")


def clean_data() -> None:
    """Clean text files."""
    ONLY_VISUALIZATION = bool(int(os.getenv("ONLY_VISUALIZATION", 0)))

    if ONLY_VISUALIZATION:
        _LOGGER.debug(f"Visualizing clean dataset...")
        _visualize_clean_data()
        return

    current_path = Path.cwd()
    repo_path = current_path.joinpath("data_science")

    # TODO: Generalize to have more input initial dataset
    complete_file_path = repo_path.joinpath(
        "datasets", "hundreds_data_science_packages_initial_dataset.json"
    )

    dataset = _retrieve_file(file_path=complete_file_path, file_type="json")

    nlp_repo_path = repo_path.joinpath("nlp")

    # Retrieve list of common words list for normalization
    common_words_txt = _retrieve_file(
        file_path=f"{nlp_repo_path}/common_words.txt", file_type="txt"
    )
    common_words = [word for word in common_words_txt.split("\n")]
    _LOGGER.debug(f"Common words... \n{common_words}")

    # Retrieve list of specific common words list for normalization
    specific_common_words_txt = _retrieve_file(
        file_path=f"{nlp_repo_path}/specific_common_words.txt", file_type="txt"
    )
    specific_common_words = [word for word in specific_common_words_txt.split("\n")]
    _LOGGER.debug(f"Specific common words... \n{specific_common_words}")

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
    bigram_path = nlp_repo_path.joinpath("bigram_model.pkl")
    bigram_model = Phraser.load(str(bigram_path))

    trigram_path = nlp_repo_path.joinpath("trigram_model.pkl")
    trigram_model = Phraser.load(str(trigram_path))

    clean_dataset = {}
    clean_dataset_sentences = {}
    hyphen_words = {}

    counter_document = 1
    number_documents = len(dataset.keys())
    for file_id, file_data in tqdm(dataset.items(), desc="Cleaning Readme"):
        file_name = file_data["file_name"]
        _LOGGER.info(
            f"Cleaning document number: {counter_document}/{number_documents}..."
        )
        _LOGGER.info(f"Data cleaning for file id: {file_id}...")
        _LOGGER.info(f"Data cleaning for file name: {file_name}...")

        if file_data["raw_text"]:
            readme_raw_text = file_data["raw_text"]

            vocabulary, sentences, possible_ngrams_words = text_processing(
                raw_text=readme_raw_text,
                common_words=common_words,
                specific_common_words_txt=specific_common_words_txt,
                non_characerter_words=non_characerter_words,
                bigram_model=bigram_model,
                trigram_model=trigram_model,
            )
            # TODO: Store each json separetly to reduce data in RAM
            _LOGGER.info(f"File cleaned vocabulary... \n{vocabulary}")
            clean_dataset[file_name] = vocabulary

            _LOGGER.info(f"File cleaned sentences... \n{sentences}")
            clean_dataset_sentences[file_name] = sentences

            _LOGGER.info(f"File Hyphen words... \n{possible_ngrams_words}")
            hyphen_words[file_name] = possible_ngrams_words
        else:
            _LOGGER.warning(f"{file_id} does not have a readme file!")

        complete_file_path = repo_path.joinpath("datasets", "clean_dataset.json")

        _store_file(
            file_path=complete_file_path, file_type="json", collected_data=clean_dataset
        )

        complete_file_path = repo_path.joinpath(
            "datasets", "clean_sentences_dataset.json"
        )

        _store_file(
            file_path=complete_file_path,
            file_type="json",
            collected_data=clean_dataset_sentences,
        )

        complete_file_path = repo_path.joinpath("nlp", "possible_ngrams_words.json")

        _store_file(
            file_path=complete_file_path, file_type="json", collected_data=hyphen_words
        )

        counter_document += 1


def _visualize_clean_data():
    # TODO: show vocabulary statistics
    current_path = Path.cwd()
    repo_path = current_path.joinpath("data_science")
    complete_file_path = repo_path.joinpath("datasets", "plot_dataset.json")
    PLOT_NUMBER_TOKENS = int(os.getenv("PLOT_NUMBER_TOKENS", 10))
    if not complete_file_path.exists():
        complete_file_path = repo_path.joinpath("datasets", "clean_dataset.json")
        clean_dataset = _retrieve_file(file_path=complete_file_path, file_type="json")

        all_tokens = []
        for file_id, vocabulary in clean_dataset.items():
            all_tokens += vocabulary

        wordcount = {}
        _LOGGER.info(f"Number of tokens: {len(all_tokens)}")
        _LOGGER.info(f"Number of unique tokens: {len(set(all_tokens))}")

        for word in all_tokens:
            if word not in wordcount.keys():
                wordcount[word] = 1
            else:
                wordcount[word] += 1

        sorted_wc = sorted(wordcount.items(), key=lambda k_v: k_v[1], reverse=True)
        wc_name = (
            "word_count_bar" + "_" + datetime.utcnow().strftime("%Y-%m-%d_%H:%M:%S")
        )
        complete_file_path = repo_path.joinpath("nlp", wc_name)

        _store_file(
            file_path=complete_file_path,
            file_type="json",
            collected_data=dict(sorted_wc),
        )
        sorted_wc = sorted_wc[:PLOT_NUMBER_TOKENS]

        counter = 0
        for word, count in sorted_wc:
            if counter <= PLOT_NUMBER_TOKENS:
                _LOGGER.info(f"Word: {word}, Count: {count}")

        sorted_wc = dict(sorted_wc)
        names = list(sorted_wc.keys())
        values = list(sorted_wc.values())

        plt.bar(range(len(sorted_wc)), values, tick_label=names)
        plt.title("Word count after text processing for DS Packages' README")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.grid()

        plot_name = (
            f"top_{PLOT_NUMBER_TOKENS}_word_count_bar"
            + "_"
            + datetime.utcnow().strftime("%Y-%m-%d_%H:%M:%S")
        )
        complete_file_path = repo_path.joinpath("nlp", plot_name)
        plt.savefig(complete_file_path)
        plt.close()
