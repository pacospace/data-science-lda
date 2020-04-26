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

"""Detect common phrases, aka multi-word expressions, word n-gram collocations."""

import logging

from pathlib import Path
from gensim.models.phrases import Phrases, Phraser

from ..utils import _retrieve_file
from ..utils import _store_file

_LOGGER = logging.getLogger("data_science_lda.nlp.common_phrases")


def collect_common_phrases() -> None:
    """Collect common phrases."""
    current_path = Path.cwd()
    repo_path = current_path.joinpath("data_science")

    complete_file_path = repo_path.joinpath("datasets", "clean_sentences_dataset.json")
    clean_sentences_dataset = _retrieve_file(
        file_path=complete_file_path, file_type="json"
    )

    all_sentences = []
    for file_name, file_sentences in clean_sentences_dataset.items():
        _LOGGER.info(f"Taking vocabulary for: {file_name}...")
        for sentence in file_sentences:
            all_sentences.append(sentence)

    phrases = Phrases(all_sentences, min_count=5, threshold=5)
    bigram = Phraser(phrases)

    phrases = Phrases(bigram[all_sentences], threshold=1)
    trigram = Phraser(phrases)

    n_grams_extracted = {}
    bigrams = []
    trigrams = []
    fourgrams = []
    _LOGGER.info("Checking identified n-grams...")
    # Verify words identified
    for file_name, file_sentences in clean_sentences_dataset.items():
        for sentence in file_sentences:

            bigram_found = [b for b in bigram[sentence] if len(b.split("_")) == 2]

            if bigram_found:
                for sb in bigram_found:
                    if sb not in bigrams:
                        bigrams.append(sb)

            trigram_found = [
                t for t in trigram[bigram[sentence]] if len(t.split("_")) == 3
            ]

            if trigram_found:
                for tb in trigram_found:
                    if tb not in trigrams:
                        trigrams.append(tb)

    _LOGGER.debug(f"Identified bigrams are... {bigrams}")
    n_grams_extracted["bigrams"] = bigrams
    _LOGGER.debug(f"Identified trigrams are... {trigrams}")
    n_grams_extracted["trigrams"] = trigrams

    complete_file_path = repo_path.joinpath("nlp", "ngrams_extracted.json")

    _store_file(
        file_path=complete_file_path, file_type="json", collected_data=n_grams_extracted
    )

    bigram_path = repo_path.joinpath("nlp", "bigram_model.pkl")
    bigram.save(str(bigram_path))

    trigram_path = repo_path.joinpath("nlp", "trigram_model.pkl")
    trigram.save(str(trigram_path))
