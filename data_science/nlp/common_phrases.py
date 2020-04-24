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
    clean_sentences_dataset = _retrieve_file(file_path=complete_file_path, file_type="json")

    all_sentences = []
    for file_name, file_sentences in clean_sentences_dataset.items():
        _LOGGER.info(f"Taking vocabulary for: {file_name}...")
        for sentence in file_sentences:
            all_sentences.append(sentence)

    phrases = Phrases(all_sentences, min_count=5, threshold=1)
    # Export the trained model = use less RAM, faster processing. Model updates no longer possible.
    bigram = Phraser(phrases)

    tests = [["neural", "network"], ["model", "prediction"], ["data", "visualization"]]
    for test in tests:
        print(test)
        result_bigram = bigram[test]
        print(result_bigram)

    bigram_path = repo_path.joinpath("datasets", "bigram_model.pkl")
    # Save an exported collocation model.
    bigram.save(str(bigram_path))
