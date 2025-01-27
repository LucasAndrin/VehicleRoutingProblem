from matplotlib import pyplot as plt
import networkx as nx
import random as rd
import numpy as np

class VRPSolver:
    def __init__(self, graph: nx.DiGraph):
        self.graph = graph

if __name__ == "__main__":
    # Constants
    SEED = 123
    NODES = 16
    CLIENTS = 4
    MIN_WEIGHT = 1
    MAX_WEIGHT = 20

    # Set random seeds for reproducibility
    np.random.seed(SEED)
    rd.seed(SEED)

    graph: nx.DiGraph = nx.erdos_renyi_graph(NODES, CLIENTS / NODES, seed=SEED, directed=True)
    for u, v in graph.edges():
        graph[u][v]['weight'] = rd.randint(MIN_WEIGHT, MAX_WEIGHT)

    solver = VRPSolver(
        graph
    )

    pos = nx.spring_layout(graph, seed=SEED)

    # Draw Nodes
    nx.draw_networkx_nodes(graph, pos, node_size=200)
    nx.draw_networkx_labels(graph, pos, font_size=8)

    # Draw Edges
    nx.draw_networkx_edges(graph, pos, style="dashed", alpha=0.5)
    nx.draw_networkx_edge_labels(graph, pos, font_size=8,
        edge_labels=nx.get_edge_attributes(graph, 'weight'),
    )

    ax = plt.gca()
    ax.margins(0.08)
    plt.axis("off")
    plt.tight_layout()
    plt.show()