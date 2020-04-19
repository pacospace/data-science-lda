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

import spacy
import enchant

from pathlib import Path
from typing import List
from typing import Union
from typing import Dict

from ..utils import _retrieve_file

_LOGGER = logging.getLogger("data_science_lda.nlp.text_processing")

# Introduce the US vocabulary
US_VOCABULARY = enchant.Dict('en_US')

# The installation doesn’t automatically download the English model.
# run `pipenv run python3 -m spacy download en`
try:
    _NLP_SPACY = spacy.load('en')
except Exception as load_model:
    _LOGGER.error(load_model)

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

def _entity_word_expansion_map(raw_text: str, entity_word_type: str) -> Dict[str, List[str]]:
    """Map entity (possible) word extracted with their expansion."""
    _SPLIT_TYPE_ENTITIES = {
        "hyphen": "-",
        "slash": "/"
    }

    word_expansion_map = {}

    entity_words = _entity_words_extraction(raw_text=raw_text, entity_word_type=entity_word_type)

    for slash_word in entity_words:
        word_expansion_map[slash_word] = slash_word.split(_SPLIT_TYPE_ENTITIES[entity_word_type])

    return word_expansion_map

def _abbreviations_extraction(raw_text: str) -> List[str]:
    """Extract abbreviations from the raw text, used for normalization"""
    # TODO: Add regex also for abbreviations like e.g.
    _REGEX_ABBREVIATION = r"(\w+\.)"
    extracted_words = re.findall(_REGEX_ABBREVIATION, raw_text)
    abbreviations = [abb for abb in extracted_words if 2 < len(abb) <= 3 and not abb[0].isdigit()]

    return sorted(list(set(abbreviations)))

def _acronyms_extraction(sentence: str, words_candidates: List[str]):
    """Extract acronyms (particular abbreviations) from the raw text.

        Two steps are performed:
        Step 1 --> extract all the short term (e.g. RH)
        Step 2 --> extract all long term(e.g. Red Hat)

        INPUT:
        - sentence -->
        - tokens of the sentence -->

        OUTPUT:
        - acronyms and expansions -->
    """

    acronym_candidates = []

    # Step 1 --> Extract short terms

    regex_t = r"(?:\([A-Z][\w+]+\))"
    acronyms_t = re.findall(regex_t, sentence)

    for term in acronyms_t:

        acronym_candidates.append(term.lstrip('.').strip('()'))

    acronyms_and_expansions = []

    # Step 2 --> Extract long terms

    for acronym in set(acronym_candidates):
        acr = []
        n = 0
        for l in acronym:

            if l.isupper():

                acr.append(l)
                n += 1

            elif l.isdigit() and l != '²':

                for i in range(1, int(l)):

                    acr.append(acronym[n - 1])

                n += 1
        acronym_exp = []

        if len(acr) > 1:

            global n_w
            global n_l
            n_w = 0
            n_l = 0
            n_w, n_l, acronym_exp = _acronym_expansion_extraction(n_w, n_l, acronym_exp, acr, words_candidates)

        if acronym_exp:

            if len(acr) == len(acronym_exp):

                acronyms_and_expansions.append([acronym, acronym_exp])

    return acronyms_and_expansions

def _acronym_expansion_extraction(n_word: int, n_letter: int, expanded_acronym: List[str], acr: List[str], words: List[str]):
    """Extract the expansion of the candidate acronyms (Recursive function)

        INPUT:
        - counter for the word lists -->
        - counter for the letter of the acronym -->
        - list to collect the result (if any) -->
        - acronym -->
        - words list of the sentence in which there is the acronym -->
    """

    while len(expanded_acronym) < len(acr) and n_word < len(words) - 1:

        if len(words[n_word]) > 1:

            if n_letter > 1 and words[n_word][0].islower() and words[n_word][1].islower() or words[n_word] == '&':

                n_word += 1

                if words[n_word][0] == acr[n_letter] and words[n_word][1].islower():

                    expanded_acronym.append(words[n_word])
                    n_letter += 1
                    n_word += 1

                    if n_letter < len(acr):

                        n_word, n_letter, expanded_acronym = \
                            _acronym_expansion_extraction(n_word, n_letter, expanded_acronym, acr, words)

                    else:

                        break
                else:

                    if n_word == len(words) - 1:

                        break

                    else:

                        n_word += 1
                        n_letter = 0
                        expanded_acronym.clear()
                        n_word, n_letter, expanded_acronym = \
                            _acronym_expansion_extraction(n_word, n_letter, expanded_acronym, acr, words)

            else:

                if words[n_word][0] == acr[n_letter] or words[n_word] == '&':

                    expanded_acronym.append(words[n_word])
                    n_letter += 1
                    n_word += 1

                    if n_letter < len(acr):

                        n_word, n_letter, expanded_acronym = \
                            _acronym_expansion_extraction(n_word, n_letter, expanded_acronym, acr, words)

                    else:

                        break

                else:

                    if n_word == len(words) - 1:

                        break

                    else:

                        n_word += 1
                        n_letter = 0
                        expanded_acronym.clear()
                        n_word, n_letter, expanded_acronym = \
                            _acronym_expansion_extraction(n_word, n_letter, expanded_acronym, acr, words)

        else:

            n_word += 1
            n_word, n_letter, expanded_acronym = \
                _acronym_expansion_extraction(n_word, n_letter, expanded_acronym, acr, words)

    return n_word, n_letter, expanded_acronym

def _acronyms_collection(sentences: List[str]):
    """Collect all the acronyms/expansions extracted"""
    acronyms = []
    acronyms_expansions_collection = []
    acronym_list_trace = []

    for s in sentences:

        tokenized_sentence = [str(token) for token in _NLP_SPACY(str(s))]
        acronyms_and_expansions = _acronyms_extraction(str(s), tokenized_sentence)

        if acronyms_and_expansions:

            for single_acronym, expansion in zip([acr[0] for acr in acronyms_and_expansions],
                                                 [acr[1] for acr in acronyms_and_expansions]):

                if single_acronym not in [el[0] for el in acronyms_expansions_collection] and expansion in [el[1] for el in
                                                                                                      acronyms_expansions_collection]:
                    acronyms.append(single_acronym)
                    acronyms_expansions_collection.append([single_acronym, expansion])
                    acronym_list_trace.append([[single_acronym, expansion], s])

                if single_acronym not in [el[0] for el in acronyms_expansions_collection] and expansion not in [el[1] for el
                                                                                                          in
                                                                                                          acronyms_expansions_collection]:
                    acronyms.append(single_acronym)
                    acronyms_expansions_collection.append([single_acronym, expansion])
                    acronym_list_trace.append([[single_acronym, expansion], s])

    return acronyms, acronyms_expansions_collection

def _word_correction(tokens: List[str], words_expansions: Dict[str, List[str]]) -> List[str]:
    """Prepare tokens to be expanded."""
    modified_tokens = list(tokens)

    for token in tokens:

        if token in words_expansions.keys():

            modified_tokens = toks_correction(modified_tokens, token, words_expansions)

    return modified_tokens

def toks_correction(modified_tokens: List[str], token: str, words_expansions: Dict[str, List[str]]) -> List[str]:
    """Expand words with corresponding expansion."""
    ww = modified_tokens.index(token)
    hw = [k for k in words_expansions.keys()].index(token)
    expansions = [k for k in words_expansions.values()]
    modified_tokens = list(modified_tokens[0:ww]) + list(expansions[hw]) + list(modified_tokens[ww + 1:])

    return modified_tokens

def text_processing(raw_text: str):
    """Apply text processing to raw text."""
    current_path = Path.cwd()
    repo_path = current_path.joinpath('data_science', "nlp")

    # Retrieve list of common words list for normalization
    common_words_txt = _retrieve_file(
        file_path=f'{repo_path}/common_words.txt',
        file_type="txt"
    )
    common_words = [word for word in common_words_txt.split("\n")]
    _LOGGER.debug(f"common_words words... \n{common_words}")

    # Retrieve list of non-character word list for normalization
    non_characerter_words_txt = _retrieve_file(
        file_path=f'{repo_path}/non_character_words.txt',
        file_type="txt"
    )
    non_characerter_words = [n_character.split('\n')[0] for n_character in non_characerter_words_txt]
    _LOGGER.debug(f"Non-character words... \n{non_characerter_words}")

    # Extract hyphen words for normalization
    entity_word_type = "hyphen"
    hyphen_word_expansion = _entity_word_expansion_map(raw_text=raw_text, entity_word_type=entity_word_type)
    _LOGGER.debug(f"Possible hyphen words identified with their expensions... \n{hyphen_word_expansion}")
    hyphen_words = [h[0] for h in hyphen_word_expansion]

    # Extract slash words for normalization
    entity_word_type = "slash"
    slash_word_expansion = _entity_word_expansion_map(raw_text=raw_text, entity_word_type=entity_word_type)
    _LOGGER.debug(f"Possible slash words identified with their expensions... \n{slash_word_expansion}")
    slash_words = [s[0] for s in slash_word_expansion]

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

    doc = _NLP_SPACY(raw_text)

    acronyms_list, acronyms_expansions_list_result = _acronyms_collection(doc.sents)
    _LOGGER.debug(f"Acronyms identified... \n{acronyms_list}")

    file_vocabulary = []
    n = 1
    for sent in doc.sents:
        _LOGGER.debug(f"Sentence n.{n}:\n{sent}")
        tokenized_sentence = [str(token) for token in _NLP_SPACY(str(sent))]

        clean_tokens = [
            token.strip("'").rstrip('-').lstrip('-').lstrip('.').lstrip('∗').lstrip('–').rstrip('.')
                            for token in tokenized_sentence
        ]
        _LOGGER.debug(f"Initial tokens... \n{clean_tokens}")

        # Expansion of hyphen and slash words
        clean_tokens = _word_correction(clean_tokens, hyphen_word_expansion)
        _LOGGER.debug(f"Tokens after hyphen words expansion... \n{clean_tokens}")
        clean_tokens = _word_correction(clean_tokens, slash_word_expansion)
        _LOGGER.debug(f"Tokens after slash words expansion... \n{clean_tokens}")

        # Remove stopwords, punctuations, symbols
        clean_tokens = [token for token in clean_tokens if len(token) >= 1 and token.lower() not in non_characerter_words]
        _LOGGER.debug(f"Tokens after non character words cleaning... \n{clean_tokens}")

        # Lower the tokens
        clean_tokens = [token.lower() for token in clean_tokens]
        _LOGGER.debug(f"Tokens after lowering words.. \n{clean_tokens}")

        # Maintain only the word in the vocabulary and numbers + entity known
        # TODO: maintain the entity, NER?
        # TODO: maintain n-grams extracted?
        clean_tokens = [token for token in clean_tokens if (US_VOCABULARY.check(token)) and (len(token) > 1)]
        _LOGGER.debug(f"Tokens after checking vocabulary.. \n{clean_tokens}")

        # Remove common words
        clean_tokens = [token for token in clean_tokens if token not in common_words]
        _LOGGER.debug(f"Tokens after common words cleaning... \n{clean_tokens}")

        # Remove numbers
        # clean_tokens = [re.sub('[^a-zA-Z]', '', token) for token in clean_tokens]
        clean_tokens = [token for token in clean_tokens if not token[0].isdigit()]

        _LOGGER.debug(f"Cleaned tokens... \n{clean_tokens}")

        # There are repetitions!
        file_vocabulary += clean_tokens
        n += 1

    return file_vocabulary