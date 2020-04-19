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

"""Text processing."""

import logging
import re

from pathlib import Path
from typing import List
from typing import Union

from ..utils import _retrieve_file

_LOGGER = logging.getLogger("data_science_lda.nlp.text_processing")


def _entity_words_extraction(raw_text: str, entity_word_type: str) -> List[str]:
    """Extract entity (possible) words (hyphen/slash) from the text."""
    _REGEX_HYPHEN = [
        r"(\w+.-\w+)",
        r"(\w+-\w+)",
        r"(\w+-\w+-\w+)",
        r"(\w+-\w+-\w+-\w+)",
        r"(\w+-\w+-\w+.\w+)"
    ]

    _REGEX_SLASH = [
        r"(\w+/\w+)"
    ]

    _REGEX_ENTITIES = {
        "hyphen": _REGEX_HYPHEN,
        "slash": _REGEX_SLASH
    }

    words_entities = []

    for regex in _REGEX_ENTITIES[entity_word_type]:
        words_entities += re.findall(regex, raw_text)

    return sorted(list(set(words_entities)))

def _entity_word_expansion_map(raw_text: str, entity_word_type: str) -> List[Union[str, List[str]]]:
    """Map entity (possible) word extracted with their expansion."""
    _SPLIT_TYPE_ENTITIES = {
        "hyphen": "-",
        "slash": "/"
    }

    word_expansion_map = []

    entity_words = _entity_words_extraction(raw_text=raw_text, entity_word_type=entity_word_type)

    for slash_word in entity_words:
        word_expansion_map.append([slash_word, slash_word.split(_SPLIT_TYPE_ENTITIES[entity_word_type])])

    return word_expansion_map

def _abbreviations_extraction(raw_text: str) -> List[str]:
    """Extract abbreviations from the raw text, used for normalization"""
    # TODO: Add regex also for abbreviations like e.g.
    _REGEX_ABBREVIATION = r"(\w+\.)"
    extracted_words = re.findall(_REGEX_ABBREVIATION, raw_text)
    abbreviations = [abb for abb in extracted_words if 2 < len(abb) <= 3 and not abb[0].isdigit()]

    return sorted(list(set(abbreviations)))

def text_processing(raw_text: str):
    """Apply text processing to raw text."""
    current_path = Path.cwd()
    repo_path = current_path.joinpath('data_science', "nlp")

    # Retrieve list of non-character word list for normalization
    non_characerter_words = _retrieve_file(
        file_path=f'{repo_path}/non_character_words.txt',
        file_type="txt"
    )
    non_characerter_words = [n_character.split('\n')[0] for n_character in non_characerter_words]
    _LOGGER.debug(f"Non-character words... \n{non_characerter_words}")

    # Extract hyphen words for normalization
    entity_word_type = "hyphen"
    hyphen_word_expansion = _entity_word_expansion_map(raw_text=raw_text, entity_word_type=entity_word_type)
    _LOGGER.debug(f"Possible hyphen words identified with their expensions... \n{hyphen_word_expansion}")

    # Extract slash words for normalization
    entity_word_type = "slash"
    slash_word_expansion = _entity_word_expansion_map(raw_text=raw_text, entity_word_type=entity_word_type)
    _LOGGER.debug(f"Possible slash words identified with their expensions... \n{slash_word_expansion}")

    # Extract Abbreviations
    abbreviations = _abbreviations_extraction(raw_text=raw_text)
    _LOGGER.debug(f"Abbreviations identified... \n{abbreviations}")

    # Retrieve list of Abbreviations inserted for normalization
    abbreviations_inserted = _retrieve_file(
        file_path=f'{repo_path}/abbreviations.txt',
        file_type="txt"
    )
    abbreviations_expansions = []
    for abbreviation_row in abbreviations_inserted.split('\n'):
        abbreviation_expansion = abbreviation_row.split(',')
        abbreviations_expansions.append([abbreviation_expansion[0], abbreviation_expansion[1].split(' ')])
    _LOGGER.debug(f"Abbreviations inserted with their expansions... \n{abbreviations_expansions}")