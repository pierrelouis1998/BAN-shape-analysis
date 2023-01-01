from typing import List

import networkx as nx
from matplotlib import pyplot as plt
import numpy as np

from utils.graph_toolbox import plot_weighted_graphs, plot_graphs_same_fig, generate_random_weighted_graph, graph_mean, \
    graph_mean_pos, reg_graph, distance_to_ref, distB, distG, distance_matrix
from utils.matching import find_optimal_matching


def visualize_matching(G1: nx.Graph, G2: nx.Graph, verbose: bool = False) -> None:
    """
    Visualize graph matching
    :return:
    """
    translate = np.ones(2) * 0.15
    res, P = find_optimal_matching(G1, G2)
    new_G = nx.from_numpy_matrix(P @ nx.adjacency_matrix(G2).toarray() @ P.T)
    new_G.graph['pos'] = {
        k: G2.graph['pos'][np.argmax(P[k, :])] + translate for k in range(G2.number_of_nodes())
    }
    if verbose:
        print(f"Optimal value : {res:.2f}\nPermutation matrix: {P}")

    labels = [
        r'$G_1$',
        r'$G_2$',
        r'$G_P$'
    ]
    # pos = [0, 1, 0]
    fig = plot_graphs_same_fig(
        [G1, G2, new_G],
        labels,
    )
    # fig = plot_weighted_graphs(
    #     [G1, G2, new_G],
    #     labels,
    #     figsize=(12, 4),
    #     tight_layout=False
    # )
    fig.savefig('matching.pdf')
    fig.show()


def plot_histogram(graphs: List[nx.Graph]):
    """

    :param graphs:
    :return:
    """
    mean_reg = graph_mean_pos(graphs, True)
    graphs_reg = [
        reg_graph(G, graphs[0]) for G in graphs
    ]
    mean_one_reg = graph_mean_pos(graphs_reg, False)
    mean_no_reg = graph_mean_pos(graphs, False)
    d_reg = distance_to_ref(distG, graphs, mean_reg)
    d_one_reg = distance_to_ref(distB, graphs_reg, mean_one_reg)
    d_no_reg = distance_to_ref(distG, graphs, mean_no_reg)
    bins = np.linspace(0.1, 0.8, 10)
    plt.hist(d_reg, label="Mean with registration and $d_G$", alpha=0.5)
    plt.hist(d_one_reg, label="Mean with one registration and $d_B$", alpha=0.5)
    # plt.hist(d_no_reg, label="No registration", alpha=0.5)
    plt.legend()
    plt.savefig("hist_mean.pdf")
    plt.show()

    return None


def plot_hist_dist_mat(graphs: List[nx.Graph]):
    """

    :param graphs:
    :return:
    """
    graphs_reg = [
        reg_graph(G, graphs[0]) for G in graphs
    ]
    db = distance_matrix(distB, graphs).reshape(-1)
    db_tilde = distance_matrix(distB, graphs_reg).reshape(-1)
    dg = distance_matrix(distG, graphs).reshape(-1)
    # bins = np.linspace(0.1, 0.8, 10)
    plt.hist(db - db_tilde, label=r"$d_b-\tilde{d}_b$", alpha=0.5)
    plt.hist(db - dg, label=r"$d_b - d_g$", alpha=0.5)
    # plt.hist(d_no_reg, label="No registration", alpha=0.5)
    plt.legend()
    plt.savefig("hist.pdf")
    plt.show()

    return None


def main(n, p, n_sample):
    G1 = generate_random_weighted_graph(n, p)
    G2 = generate_random_weighted_graph(n, p)
    visualize_matching(G1, G2)

    graph_list = [
        generate_random_weighted_graph(n, p) for k in range(n_sample)
    ]
    # mean = graph_mean_pos(graph_list, registration=False)
    # labels = [
    #     f"$G_{i}$" for i in range(n_sample)
    # ]
    # labels.append(r'$G_{\mu}$')
    # fig = plot_graphs_same_fig(
    #     graph_list + [mean],
    #     labels=labels
    # )
    # fig = plot_weighted_graphs(
    #     graph_list + [mean],
    #     labels=labels,
    #     figsize=(12, 8)
    # )
    # fig.savefig('mean.pdf')
    # fig.show()
    plot_histogram(graph_list)
    # plot_hist_dist_mat(graph_list)
    pass


if __name__ == '__main__':
    main(8, 0.4, 15)
