import copy
import random
from typing import List, Callable

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import random

from tqdm import tqdm

from utils.matching import find_optimal_matching
from utils.math_utils import draw_on_ball

MAX_ITER = 15
COLOR_CYCLE = ['#7ac6fa', '#ef9041', '#80fa7a', '#fa7add', '#9d7afa'] * 4


def generate_random_weighted_graph(n: int, p: float, r: float = 1) -> nx.Graph:
    """
    Generate a random weighted graph using fast gnp method. Node position are generated randomly on the sphere
    :param r: Ray of the sphere
    :param n: Number of vertices
    :param p: Edge probability
    :return: Random weighed graph
    """
    G = nx.gnp_random_graph(n, p, directed=False)
    # Generate position on the ball
    vertices = draw_on_ball(n, r, dim=2)
    # Assign position to each node
    pos = {
        k: vertices[k] for k in range(n)
    }
    # Add position as graph attributes
    G.graph['pos'] = pos
    G.graph['ray'] = r
    # Assign edge weight according to nodes distance
    for (u, v) in G.edges():
        G.edges[u, v]['weight'] = r - np.linalg.norm(pos[u] - pos[v])
    return G


def plot_graphs_same_fig(
        graph_list: List[nx.Graph],
        labels: List[str],
        **kwargs
) -> plt.Figure:
    """

    :param graph_list:
    :param labels:
    :param kwargs:
    :return:
    """
    n_graph = len(graph_list)
    scaling = 2.0
    weights = [
        [(G[u][v]['weight'] + 0.1) * scaling for u, v in G.edges()] for G in graph_list
    ]
    # Visualizes attention weights
    arc_rad = 0.25
    figsize = kwargs.get('figsize')
    tight_layout = kwargs.get('tight_layout', False)
    fig = plt.figure(figsize=figsize, tight_layout=tight_layout)
    for i, G in enumerate(graph_list):
        layout = G.graph['pos']
        if labels:
            lab = labels[i]
        else:
            lab = f'$G_{i}$'
        nx.draw_networkx(
            G,
            pos=layout,
            width=weights[i],
            node_size=110,
            node_color=COLOR_CYCLE[i],
            edge_color=COLOR_CYCLE[i],
            font_size=8,
            connectionstyle=f'arc3, rad = {arc_rad}',
            label=lab
        )

        plt.legend()
    return fig


def plot_weighted_graphs(
        graph_list: List[nx.Graph],
        labels: List[str] = None,
        pos: List[int] = None,
        **kwargs
) -> plt.Figure:
    """
    Wrapper around nx.draw_network to visualize edge weight in the plot for a list of graph
    :param pos: Specify reference layout for drawing. pos[3] = 0 means that the 4th graph
    should be drawn according to the first graph layout. We always need pos[0]=0. If not specified
    will use the attribute G.graph['pos']
    :param labels:
    :param graph_list: List of graph to plot
    :return: Figure
    """
    n_graph = len(graph_list)
    scaling = 2.0
    weights = [
        [(G[u][v]['weight'] + 0.1) * scaling for u, v in G.edges()] for G in graph_list
    ]

    # Visualizes attention weights
    pos_ref = [
        nx.spring_layout(G) for G in graph_list
    ]
    if pos:
        assert pos[0] == 0
    arc_rad = 0.25
    n_cols = min(3, n_graph)
    n_rows = max(1, n_graph // n_cols)
    figsize = kwargs.get('figsize')
    tight_layout = kwargs.get('tight_layout', False)
    fig = plt.figure(figsize=figsize, tight_layout=tight_layout)
    for i, G in enumerate(graph_list):
        plt.subplot(n_rows, n_cols, i + 1)
        if pos:
            layout = pos_ref[pos[i]]
        else:
            layout = G.graph['pos']
        nx.draw_networkx(
            G,
            pos=layout,
            width=weights[i],
            node_size=110,
            node_color='#7ac6fa',
            font_size=8,
            connectionstyle=f'arc3, rad = {arc_rad}'
        )
        if labels:
            plt.xlabel(labels[i])
    return fig


def graph_mean_pos(graphs: List[nx.Graph], registration: bool = True) -> nx.Graph:
    """
    Compute mean graph based on node position
    :param registration:
    :param graphs: List of graph
    :return: Mean graph
    """
    n_samples = len(graphs)
    n = graphs[0].number_of_nodes()
    mean = copy.deepcopy(graphs[0])
    for k in tqdm(range(MAX_ITER), desc='Graph mean', unit='steps'):
        for i, G in enumerate(graphs):
            if registration:
                _, P = find_optimal_matching(mean, G)
            else:
                P = np.eye(n)
            G.graph['pos'] = {
                k: G.graph['pos'][np.argmax(P[:, k])] for k in range(n)
            }
            graphs[i] = copy.deepcopy(G)
        mean.graph['pos'] = {
            k: np.mean([graphs[i].graph['pos'][k] for i in range(n_samples)], axis=0) for k in range(n)
        }
        for (u, v) in mean.edges():
            mean[u][v]['weight'] = mean.graph['ray'] - np.linalg.norm(mean.graph['pos'][u] - mean.graph['pos'][v])
    return mean


def graph_mean(graphs: List[nx.Graph]) -> nx.Graph:
    """
    Compute Mean graph
    :param graphs: List of graph whose mean are computed
    :return:
    """
    mean = graphs[0]
    list_adjacency = [nx.adjacency_matrix(graphs[k]).toarray() for k in range(len(graphs))]
    for k in range(MAX_ITER):
        # TODO : align with Procrustes method
        for i, A in enumerate(list_adjacency):
            _, P = find_optimal_matching(mean, nx.from_numpy_matrix(A))
            A = P @ A @ P.T
            list_adjacency[i] = A
        mean = np.mean(list_adjacency, axis=0)
        mean = nx.from_numpy_matrix(mean)
    return mean


def distance_to_ref(dist: Callable, graph_list: List[nx.Graph], ref: nx.Graph) -> np.array:
    """
    Return the distance matrix for a list of graph with a given distance function
    :param ref:
    :param dist:
    :param graph_list:
    :return:
    """
    n = len(graph_list)
    D = np.asarray([distG(graph_list[k], ref) for k in range(n)])
    return D


def distance_matrix(dist: Callable, graphs: List[nx.Graph]) -> np.array:
    """

    :param dist:
    :param graphs:
    :return:
    """
    n = len(graphs)
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            D[i, j] = dist(graphs[i], graphs[j])
            D[j ,i] = D[i, j]
    return D


def distB(G1: nx.Graph, G2: nx.Graph) -> float:
    """

    :param G1:
    :param G2:
    :return:
    """
    n = G1.number_of_nodes()
    dist = [
        np.linalg.norm(G1.graph['pos'][k] - G2.graph['pos'][k]) for k in range(n)
    ]
    return float(np.mean(dist))


def distG(G1: nx.Graph, G2: nx.Graph) -> float:
    """

    :param G1:
    :param G2:
    :return:
    """
    _, P = find_optimal_matching(G1, G2)
    dist = [
        np.linalg.norm(G1.graph['pos'][k] - G2.graph['pos'][np.argmax(P[:, k])]) for k in range(G2.number_of_nodes())
    ]
    dist = np.mean(dist)
    return float(dist)


def reg_graph(G: nx.Graph, ref: nx.Graph) -> nx.Graph:
    """

    :param G:
    :param ref:
    :return:
    """
    _, P = find_optimal_matching(ref, G)
    G.graph['pos'] = {
        k: G.graph['pos'][np.argmax(P[:, k])] for k in range(G.number_of_nodes())
    }
    for (u, v) in G.edges():
        G[u][v]['weight'] = np.linalg.norm(G.graph['pos'][u] - G.graph['pos'][v])
    return G
