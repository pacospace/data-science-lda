#!/usr/bin/env python3
#
# Copyright(C) 2020 Devin De Hueck, Francesco Murdaca
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

"""Using Kmeans to cluster vectors created from LDA model."""

import logging
import os
import umap

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from typing import List
from typing import Any
from typing import Dict
from pathlib import Path

from gensim import corpora, models
from sklearn.cluster import KMeans

from ..exceptions import InputFileMissingError
from ..utils import _retrieve_file
from ..utils import _store_file
from ..lda.lda import create_inputs_for_lda

_LOGGER = logging.getLogger("data_science_lda.clustering.kmeans")


def _plot_clusters(
    vectors: List[List[float]],
    labels: List[str],
    vectors_name_map: Dict[str, List[Any]],
):
    """Plot clusters using UMAP: Uniform Manifold Approximation and Projection for Dimension Reduction"""
    # Create UMAP Projections
    reducer = umap.UMAP()
    projections = reducer.fit_transform(vectors)

    # Plot with KMEANS labels
    plt.scatter(
        projections[:, 0], projections[:, 1], c=[sns.color_palette()[x] for x in labels]
    )
    plt.gca().set_aspect("equal", "datalim")
    plt.title(
        f"UMAP projection of {len(vectors)} DS Package Vectors in {len(set(labels))} Groups",
        fontsize=12,
    )
    plt.grid()
    current_path = Path.cwd()
    repo_path = current_path.joinpath("data_science")
    complete_file_path = repo_path.joinpath("clustering", "clusters")
    plt.savefig(complete_file_path)


def _run_kmeans(corpus: List[Any], file_names: List[str]):
    """Use model on corpus."""
    LDA_MODEL_REPO_PATH = os.getenv("LDA_MODEL_REPO_PATH")
    if not LDA_MODEL_REPO_PATH:
        raise InputFileMissingError(
            "LDA_MODEL_REPO_PATH environment variable was not provided."
        )
    ldamodel = models.ldamodel.LdaModel.load(str(LDA_MODEL_REPO_PATH))

    topics = ldamodel.print_topics()
    for topic in topics:
        _LOGGER.debug(f"Topic: {topic}")

    NUMBER_CLUSTERS = os.getenv("NUMBER_CLUSTERS") or len(topics)
    if not NUMBER_CLUSTERS:
        raise InputFileMissingError(
            "NUMBER_CLUSTERS environment variable was not provided."
        )
    NUMBER_CLUSTERS = int(NUMBER_CLUSTERS)

    topics_files_map = {}
    # TODO: Improve vectors with package2vec and packagehealth2vec
    X = []
    for i in range(len(corpus)):
        top_topics = ldamodel.get_document_topics(corpus[i], minimum_probability=0.0)
        topic_vec = [top_topics[i][1] for i in range(len(top_topics))]
        X.append(np.array(topic_vec))
        topics_files_map[file_names[i]] = top_topics
    X = np.array(X)
    kmeans = KMeans(n_clusters=NUMBER_CLUSTERS, random_state=42).fit(X)

    _plot_clusters(X, kmeans.labels_, topics_files_map)

    # Add names to groups
    groups = {}
    for idx, label in enumerate(list(kmeans.labels_)):
        name = file_names[idx]

        if label not in groups:
            groups[label] = {}
            groups[label][name] = topics_files_map[name]
        else:
            groups[label][name] = topics_files_map[name]

    # Display group info
    for g in groups:
        _LOGGER.info(f"\n########## Group: {g} ##########\n")
        _LOGGER.info(f"\nDocuments:\n")
        for name, vector in groups[g].items():
            _LOGGER.info(f"Name: {name}")
            _LOGGER.debug(vector)


def clustering():
    """Run Clustering."""
    # TODO: Run using other clustering algorithms
    texts, dictionary, corpus, corpus_train, corpus_test, texts_names = (
        create_inputs_for_lda()
    )
    _run_kmeans(corpus=corpus, file_names=texts_names)
