from typing import Tuple

import networkx as nx
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from tqdm import tqdm
from itertools import permutations

from .matrix_utils import get_permutation_matrix, compute_laplacian, determine_correspondences

rng = np.random.default_rng()


def find_optimal_matching(G1: nx.Graph, G2: nx.Graph, verbose: bool = False) -> Tuple[float, np.array]:
    """
    Find The optimal vertices permutation to maximize Trace(A1PA2P^T).
    We assume here that the graph are of the same size and edges values are in R
    :param verbose:
    :param G1: First graph
    :param G2:
    :return: Optimal value and optimal permutation
    """
    n = G1.number_of_nodes()
    assert n == G2.number_of_nodes(), "Both graph need to have the same number of nodes"

    # Compute adjacency matrix
    A1 = nx.adj_matrix(G1).toarray()
    A2 = nx.adj_matrix(G2).toarray()

    # Compute pos matrix
    Pos1 = np.asarray(
        [G1.graph['pos'][k] for k in range(n)]
    )
    Pos2 = np.asarray(
        [G2.graph['pos'][k] for k in range(n)]
    )

    # List all permutation od size n
    perm = list(permutations(range(n)))

    # Initialize variables
    value = np.infty
    best_mat_p = None

    # Loop over every permutation (n!)
    for p in tqdm(perm, desc=f"Finding optimal permutation ({n}!)", unit="P", disable=not verbose):
        mat_p = get_permutation_matrix(p)  # Get permutation matrix
        # res = np.trace(A1 @ mat_p @ A2 @ mat_p.T)
        res = np.linalg.norm(Pos1 - mat_p.T @ Pos2)  # Compute obj
        if res <= value:
            value = res
            best_mat_p = mat_p

    # The optimal value is actually the norm diff
    return objective(A1, A2, best_mat_p), best_mat_p


def objective(A1: np.array, A2: np.array, P: np.array) -> float:
    """
    Return objective Value
    :param A1:
    :param A2:
    :param P:
    :return:
    """
    return np.linalg.norm(A1 - P @ A2 @ P.T)


def inexact_graph_matching(G1: nx.Graph, G2: nx.Graph, num_clusters: int = 2):
    """
    Computes inexact graph matching as stated in https://ieeexplore.ieee.org/document/1265866c
    :param G1:
    :param G2:
    :param num_clusters:
    :return:
    """
    raise NotImplemented("WIP")
    # Compute Laplacian matrices for both graphs
    A1 = nx.adj_matrix(G1).toarray()
    A2 = nx.adj_matrix(G2).toarray()
    L1 = compute_laplacian(A1)
    L2 = compute_laplacian(A2)

    # Compute eigenvectors and eigenvalues of the Laplacian matrices
    eigenvalues1, eigenvectors1 = np.linalg.eig(L1)
    eigenvalues2, eigenvectors2 = np.linalg.eig(L2)

    # Project the eigenvectors onto a lower-dimensional space using PCA
    pca = PCA(n_components=num_clusters)
    eigenvectors1_projected = pca.fit_transform(eigenvectors1)
    eigenvectors2_projected = pca.transform(eigenvectors2)

    # Use k-means to partition the projected eigenvectors into clusters
    kmeans1 = KMeans(n_clusters=num_clusters)
    clusters1 = kmeans1.fit_predict(eigenvectors1_projected)
    kmeans2 = KMeans(n_clusters=num_clusters)
    clusters2 = kmeans2.fit_predict(eigenvectors2_projected)

    # Use the clusters to determine the correspondences between the vertices of the two graphs
    correspondences = determine_correspondences(clusters1, clusters2)

    return correspondences
