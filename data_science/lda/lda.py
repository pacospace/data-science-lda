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

"""Apply LDA."""

import logging
import warnings
import math
import os
import random
import tqdm

import pyLDAvis.gensim
import numpy as np

from datetime import datetime
from typing import List
from typing import Dict
from typing import Any
from typing import Optional
from pathlib import Path

from gensim import corpora, models
from gensim.utils import ClippedCorpus
from gensim.models import CoherenceModel
from sklearn.model_selection import train_test_split

from ..exceptions import InputFileMissingError
from ..utils import _retrieve_file
from ..utils import _store_file

_LOGGER = logging.getLogger("data_science_lda.lda.lda")


def _split_dataset(dataset: Any, test_size: float):
    """Split dataset for training and test."""
    dataset_train, dataset_test = train_test_split(dataset, test_size=test_size)

    return dataset_train, dataset_test


def create_inputs_for_lda():
    """Create inputs for LDA."""
    current_path = Path.cwd()
    repo_path = current_path.joinpath("data_science")

    complete_file_path = repo_path.joinpath("datasets", "clean_dataset.json")
    clean_dataset = _retrieve_file(file_path=complete_file_path, file_type="json")

    texts = []
    for file_name, file_vocabulary in clean_dataset.items():
        _LOGGER.info(f"Taking vocabulary for: {file_name}...")
        texts.append(file_vocabulary)

    # Assign a unique integer id to all words appearing in the corpus, creating a vocabulary corpus
    dictionary = corpora.Dictionary(texts)
    _LOGGER.info("Number of unique tokens: %d" % len(dictionary))
    _LOGGER.debug(f"Token ID map:\n {dictionary.token2id}")

    # Bag of Words (BoW) Representation
    corpus = [dictionary.doc2bow(tokens) for tokens in texts]

    LDA_PERCENTAGE_TEST_DATASET = 0.2 or os.getenv("LDA_PERCENTAGE_TEST_DATASET")
    lda_percentage_training_dataset = (1 - float(LDA_PERCENTAGE_TEST_DATASET)) * 100
    _LOGGER.info("Training Dataset percentage is: %d" % lda_percentage_training_dataset)
    lda_percentage_test_dataset = float(LDA_PERCENTAGE_TEST_DATASET) * 100
    _LOGGER.info("Test Dataset percentage is: %d" % lda_percentage_test_dataset)
    corpus_train, corpus_test = _split_dataset(
        dataset=corpus, test_size=LDA_PERCENTAGE_TEST_DATASET
    )

    return texts, dictionary, corpus, corpus_train, corpus_test


def _run_lda(
    corpus: List[Any],
    dictionary: Dict[int, str],
    num_topics: int,
    passes: int = 300,
    iterations: int = 400,
    chunksize: int = 200,
    model_name: str = "",
    hyperparameter_tuning: bool = False,
    alpha: Optional[float] = None,
    eta: Optional[float] = None,
):
    """Apply Latent Dirichlet Allocation (LDA)

    :params corpus: Stream of document vectors or sparse matrix
                    of shape (num_terms, num_documents) for training
    :params num_topics: Number of topics.
    :params dictionary: Mapping from word IDs to words. 
                        (it is the gensim dictionary mapping on the corresponding corpus)
    :params passes: Number of passes through the corpus during training.
    :params iterations: Maximum number of iterations through the corpus
                        when inferring the topic distribution of a corpus.
    :params chunksize: Number of documents to be used in each training chunk.
    :params model_name: name to be used when saving the model
    :params hyperparameter_tuning: if set to True no LDA model is saved.
    :params alpha: Dirichlet hyperparameter alpha, Document-Topic Density.
    :params eta: Dirichlet hyperparameter beta: Document-Topic Density.
    """
    inputs = {
        "corpus": corpus,
        "num_topics": num_topics,
        "id2word": dictionary,
        "passes": passes,
        "chunksize": chunksize,
        "iterations": iterations,
    }
    if alpha:
        inputs["alpha"] = alpha

    if eta:
        inputs["eta"] = eta

    if hyperparameter_tuning:
        ldamodel = models.LdaMulticore(**inputs, per_word_topics=True, workers=7)
        return ldamodel

    ldamodel = models.ldamodel.LdaModel(**inputs)

    current_path = Path.cwd()
    repo_path = current_path.joinpath("data_science")

    model_repo_path = repo_path.joinpath("lda", model_name)
    if not model_repo_path.exists():
        _LOGGER.info(
            "No repo identified, creating new directory at %s" % model_repo_path)
        os.makedirs(model_repo_path)

    complete_file_path = model_repo_path.joinpath(f"{model_name}_lda_model")
    ldamodel.save(str(complete_file_path))

    _visualize_topics(ldamodel=ldamodel)

    warnings.filterwarnings("ignore", category=DeprecationWarning)
    # ERROR: https://github.com/bmabey/pyLDAvis/issues/69#issuecomment-270155700
    visualisation = pyLDAvis.gensim.prepare(ldamodel, corpus, dictionary)
    complete_file_path = model_repo_path.joinpath(
        f"LDA_Visualization_{model_name}.html"
    )
    pyLDAvis.save_html(visualisation, str(complete_file_path))

    return ldamodel


def _evaluate_metrics(
    ldamodel: models.ldamodel.LdaModel,
    corpus: List[Any],
    texts: List[Any],
    dictionary: List[Any],
):
    """Evaluate metrics for the model trained."""
    # Model Perplexity, a measure of how good the model is. lower the better.
    perplexity = ldamodel.log_perplexity(corpus)
    perplexity_exponential = math.exp(perplexity)

    # Coherence measure
    COHERENCE_MEASURE = "c_v" or os.getenv("COHERENCE")

    _LOGGER.info(f"\nCoherence quantity used is: {COHERENCE_MEASURE}")
    # Model Coherence Score
    coherence_model_lda = CoherenceModel(
        model=ldamodel, texts=texts, dictionary=dictionary, coherence=COHERENCE_MEASURE
    )
    coherence = coherence_model_lda.get_coherence()
    _LOGGER.info(f"\nCoherence Score: {coherence}")

    return perplexity, coherence


def _visualize_hyperparameters_tuning_results():
    """Visualize results Latent Dirichlet Allocation (LDA) Hyperparameters tuning."""
    current_path = Path.cwd()
    repo_path = current_path.joinpath("data_science")
    results_path = repo_path.joinpath("lda", "results_hyperparameter_tuning.json")
    results_hyperparameter_tuning = _retrieve_file(
        file_path=results_path, file_type="json"
    )

    results_hyperparameter_tuning.sort(key=lambda x: x[4], reverse=True)
    for result in results_hyperparameter_tuning:
        _LOGGER.info(f"Parameters: {result}")

    _LOGGER.info(f"Parameters for max coherence: {results_hyperparameter_tuning[0]}")
    _LOGGER.info(f"Max Num topics: {results_hyperparameter_tuning[0][0]}")
    # TODO Add visualizations

    return results_hyperparameter_tuning[0]


def _lda_hyperparameters_tuning(
    corpus: List[Any],
    texts: List[Any],
    dictionary: List[Any],
    min_topics: int,
    max_topics: int,
    step_size: int = 2,
    alpha_min: float = 0.01,
    alpha_max: float = 1,
    alpha_step: float = 0.5,
    eta_min: float = 0.01,
    eta_max: float = 1,
    eta_step: float = 0.5,
):
    """Latent Dirichlet Allocation (LDA) Hyperparameters tuning."""
    results = []

    topics_range = range(min_topics, max_topics, step_size)
    _LOGGER.info(
        f"Range of topics selected between {min_topics} and {max_topics} with step {step_size}"
    )
    # Alpha parameter
    alpha_spectrum = list(np.arange(alpha_min, alpha_max, alpha_step))
    alpha_spectrum.append("symmetric")
    alpha_spectrum.append("asymmetric")

    # Eta parameter
    eta_spectrum = list(np.arange(eta_min, eta_max, eta_step))
    eta_spectrum.append("symmetric")

    pbar = tqdm.tqdm(total=len(topics_range) * len(alpha_spectrum) * len(eta_spectrum))

    current_path = Path.cwd()
    repo_path = current_path.joinpath("data_science")
    results_path = repo_path.joinpath("lda", "results_hyperparameter_tuning.json")

    complete_results_and_inputs = {
        "topics_range": topics_range,
        "alpha_spectrum": alpha_spectrum,
        "eta_spectrum": eta_spectrum,
        "results": results,
    }

    for num_topics in topics_range:

        for alpha in alpha_spectrum:

            for eta in eta_spectrum:

                ldamodel = _run_lda(
                    corpus=corpus,
                    dictionary=dictionary,
                    num_topics=num_topics,
                    alpha=alpha,
                    eta=eta,
                    hyperparameter_tuning=True,
                )

                perplexity, coherence = _evaluate_metrics(
                    ldamodel=ldamodel, corpus=corpus, texts=texts, dictionary=dictionary
                )

                _LOGGER.info(f"Number of Topics {num_topics}")
                _LOGGER.info(f"Alpha {alpha} and eta {eta}")
                _LOGGER.info(f"Perplexity {perplexity}")
                _LOGGER.info(f"Coherence {coherence}")

                results.append([num_topics, alpha, eta, perplexity, coherence])

                complete_results_and_inputs["results"] = results

                _store_file(
                    file_path=results_path,
                    file_type="json",
                    collected_data=complete_results_and_inputs,
                )
                pbar.update(1)

    pbar.close()


def lda():
    """Latent Dirichlet Allocation (LDA)."""
    HYPERPARAMETER_TUNING = bool(int(os.getenv("HYPERPARAMETER_TUNING", 0)))

    ONLY_VISUALIZATION = bool(int(os.getenv("ONLY_VISUALIZATION", 0)))

    if not ONLY_VISUALIZATION:
        texts, dictionary, corpus, corpus_train, corpus_test = create_inputs_for_lda()
    else:
        _LOGGER.info(f"Only visualization of results is performed...")

    if HYPERPARAMETER_TUNING:
        if not ONLY_VISUALIZATION:
            _LOGGER.info(f"Starting Hyperparameters tuning for LDA...")

            NUMBER_TOPICS_MIN = int(os.getenv("NUMBER_TOPICS_MIN"))
            if not NUMBER_TOPICS_MIN:
                raise InputFileMissingError(
                    "NUMBER_TOPICS_MIN environment variable was not provided."
                )

            NUMBER_TOPICS_MAX = int(os.getenv("NUMBER_TOPICS_MAX"))
            if not NUMBER_TOPICS_MAX:
                raise InputFileMissingError(
                    "NUMBER_TOPICS_MAX environment variable was not provided."
                )

            _lda_hyperparameters_tuning(
                corpus=corpus_train,
                dictionary=dictionary,
                texts=texts,
                min_topics=NUMBER_TOPICS_MIN,
                max_topics=NUMBER_TOPICS_MAX,
            )

        _visualize_hyperparameters_tuning_results()

    else:

        ldamodel = None

        if not ONLY_VISUALIZATION:
            NUMBER_TOPICS = int(os.getenv("NUMBER_TOPICS"))
            if not NUMBER_TOPICS:
                raise InputFileMissingError(
                    "NUMBER_TOPICS environment variable was not provided."
                )

            LDA_ALPHA = os.getenv("LDA_ALPHA")

            LDA_ETA = os.getenv("LDA_ETA")

            MODEL_NAME = "test" or os.getenv("MODEL_NAME")
            model_name = (
                MODEL_NAME + "_" + datetime.utcnow().strftime("%Y-%m-%d_%H:%M:%S")
            )

            _LOGGER.info(
                f"LDA hyperparameter Number of topics selected is:{NUMBER_TOPICS}"
            )

            if LDA_ALPHA:
                LDA_ALPHA = float(LDA_ALPHA)
                _LOGGER.info(f"LDA hyperparameter Alpha selected is:{LDA_ALPHA}")

            if LDA_ETA:
                LDA_ETA = float(LDA_ETA)
                _LOGGER.info(f"LDA hyperparameter Eta selected is:{LDA_ETA}")

            _LOGGER.info(f"Starting LDA for... {model_name}")

            ldamodel = _run_lda(
                corpus=corpus_train,
                dictionary=dictionary,
                num_topics=NUMBER_TOPICS,
                alpha=LDA_ALPHA,
                eta=LDA_ETA,
                model_name=model_name,
            )

        _visualize_topics(ldamodel=ldamodel)


def _visualize_topics(ldamodel: Optional[models.ldamodel.LdaModel] = None):
    """Visualize topics obtained with LDA."""
    if not ldamodel:
        LDA_MODEL_REPO_PATH = os.getenv("LDA_MODEL_REPO_PATH")
        if not LDA_MODEL_REPO_PATH:
            raise InputFileMissingError(
                "LDA_MODEL_NAME environment variable was not provided."
            )
        ldamodel = models.ldamodel.LdaModel.load(str(LDA_MODEL_REPO_PATH))

    _LOGGER.info("Topics identified:\n")
    topics = ldamodel.print_topics()
    for topic in topics:
        _LOGGER.info(topic)
