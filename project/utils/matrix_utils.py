from typing import List

import numpy as np


def get_permutation_matrix(perm: List[int]) -> np.array:
    """
    Return the permutation matrix associated to the permutation
    :param perm: Permutation of range(n)
    :return: Permutation matrix
    """
    n = len(perm)
    P = np.zeros((n, n), dtype=int)
    for i in range(n):
        P[i, perm[i]] = 1
    return P


def get_random_permutation_matrix(n: int) -> np.array:
    """
    Return a random permutation matrix
    :param n: Number of vertices
    :return: P the permutation matrix
    """
    perm = list(np.random.permutation(n))
    return get_permutation_matrix(perm)


def compute_laplacian(adjacency_matrix: np.array) -> np.array:
    """
    Compute laplacian for a given adjacency matrix
    :param adjacency_matrix:
    :return: Laplacian
    """
    # Get the number of vertices in the graph
    num_vertices = adjacency_matrix.shape[0]

    # Compute the degree matrix by summing the columns of the adjacency matrix
    degree_matrix = np.diag(np.sum(adjacency_matrix, axis=0))

    # Initialize the Laplacian matrix as a copy of the degree matrix
    laplacian_matrix = degree_matrix - adjacency_matrix

    return laplacian_matrix


def determine_correspondences(clusters1, clusters2):
    """
    Determine correspondence between cluster TODO
    :param clusters1:
    :param clusters2:
    :return:
    """
    num_vertices1 = len(clusters1)
    num_vertices2 = len(clusters2)

    # Initialize the correspondences as a list of tuples
    correspondences = [(i, i) for i in range(num_vertices1)]

    # Iterate over the clusters in the first graph
    for c in np.unique(clusters1):
        # Find the cluster in the second graph with the highest number of correspondences
        cluster2 = np.argmax([np.sum(clusters2 == c2) for c2 in np.unique(clusters2)])

        # Assign the correspondences between the vertices in these two clusters based on their indices
        correspondences[c == clusters1] = (c == clusters1).nonzero()[0]
        correspondences[c == clusters2] = (cluster2 == clusters2).nonzero()[0]

    return correspondences
