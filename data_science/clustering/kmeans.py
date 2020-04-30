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
from ..exceptions import InputPathFileMissingError
from ..utils import _retrieve_file
from ..utils import _store_file
from ..lda.lda import create_inputs_for_lda

_LOGGER = logging.getLogger("data_science_lda.clustering.kmeans")


def _plot_clusters_umap(
    vectors: List[List[float]],
    labels: List[int],
    model_name: str,
    vectors_name_map: Dict[str, List[Any]],
    repo_path: Path,
):
    """Plot clusters using UMAP: Uniform Manifold Approximation and Projection for Dimension Reduction"""
    colours = sns.color_palette("RdBu", len(set(labels)))
    colours_map = {}

    for idx, colour in enumerate(colours):
        colours_map[idx] = colours[idx]

    # Create UMAP Projections ("Euclidean by deafult")
    reducer = umap.UMAP()
    projections = reducer.fit_transform(vectors)

    # Plot with KMEANS labels
    fig, ax = plt.subplots()
    for g in np.unique(labels):
        ix = np.where(labels == g)
        ax.scatter(
            projections[ix, 0],
            projections[ix, 1],
            c=[colour for idx, colour in colours_map.items() if idx == g],
            label=g,
            s=50,
        )

    ax.legend()
    plt.grid()
    plt.title(
        f"UMAP projection of {len(vectors)} DS Package Vectors in {len(set(labels))} Groups",
        fontsize=12,
    )
    plt.gca().set_aspect("equal", "datalim")

    complete_file_path = repo_path.joinpath("clustering", f"{model_name}_clusters_umap")
    plt.savefig(complete_file_path)

    return colours_map


def _run_kmeans(corpus: List[Any], file_names: List[str]):
    """Use model on corpus."""
    LDA_MODEL_REPO_PATH = os.getenv("LDA_MODEL_REPO_PATH")
    if not LDA_MODEL_REPO_PATH:
        raise InputFileMissingError(
            "LDA_MODEL_REPO_PATH environment variable was not provided."
        )

    if not LDA_MODEL_REPO_PATH.exists():
        raise InputPathFileMissingError(
            f"There is no LDA model present at this path, "
            f"you need to provide path to LDA model."
        )

    ldamodel = models.ldamodel.LdaModel.load(str(LDA_MODEL_REPO_PATH))

    model_name = str(LDA_MODEL_REPO_PATH).split("/")[-1]

    topics = ldamodel.print_topics()
    for topic in topics:
        _LOGGER.debug(f"Topic: {topic}")

    NUMBER_CLUSTERS = os.getenv("NUMBER_CLUSTERS") or len(topics)
    if not NUMBER_CLUSTERS:
        raise InputFileMissingError(
            "NUMBER_CLUSTERS environment variable was not provided."
        )
    NUMBER_CLUSTERS = int(NUMBER_CLUSTERS)
    _LOGGER.info(f"Number of clusters set to: {NUMBER_CLUSTERS}")

    current_path = Path.cwd()
    repo_path = current_path.joinpath("data_science")

    # TODO: Improve vectors with package2vec and projecthealth2vec

    topics_files_map = {}
    X = []
    for i in range(len(corpus)):
        top_topics = ldamodel.get_document_topics(corpus[i], minimum_probability=0.0)
        topic_vec = [top_topics[i][1] for i in range(len(top_topics))]

        X.append(np.array(topic_vec))
        topics_files_map[file_names[i]] = top_topics

    X = np.array(X)
    kmeans = KMeans(n_clusters=NUMBER_CLUSTERS, random_state=42).fit(X)

    _LOGGER.info(f"Kmeans labels: {kmeans.labels_}")

    _plot_clusters_umap(
        vectors=X,
        labels=kmeans.labels_,
        model_name=model_name,
        vectors_name_map=topics_files_map,
        repo_path=repo_path,
    )

    complete_file_path = repo_path.joinpath("clustering", f"{model_name}_clusters.json")

    # Add names to groups
    groups_document = {}
    groups = {}
    for idx, label in enumerate(list(kmeans.labels_)):
        name = file_names[idx]

        if label not in groups:
            groups[label] = {}
            groups[label][name] = topics_files_map[name]
        else:
            groups[label][name] = topics_files_map[name]

        if str(label) not in groups_document:
            groups_document[str(label)] = []
            groups_document[str(label)].append(name)
        else:
            groups_document[str(label)].append(name)

    _store_file(
        file_path=complete_file_path, file_type="json", collected_data=groups_document
    )

    # Display group info
    for group in sorted(groups):
        _LOGGER.info(f"########## Group: {group} ##########")
        _LOGGER.info(f"Documents:")
        for name, vector in groups[group].items():
            _LOGGER.info(f"Name: {name}")
            _LOGGER.debug(vector)


def clustering():
    """Run Clustering."""
    # TODO: Run using other clustering algorithms
    texts, dictionary, corpus, corpus_train, corpus_test, texts_names = (
        create_inputs_for_lda()
    )
    _run_kmeans(corpus=corpus, file_names=texts_names)
