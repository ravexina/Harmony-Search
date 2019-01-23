import numpy as np
from HarmonySearchTSP import HarmonySearch as HS
import networkx as nx
import matplotlib.pyplot as plt

L = np.array([
    [0, 4, 5, 1, 9],
    [4, 0, 3, 7, 4],
    [5, 3, 0, 5, 2],
    [1, 7, 5, 0, 1],
    [9, 4, 2, 1, 0]
])


def show_matrix(matrix):
    x = len(matrix)
    G = nx.Graph()
    positions = [(2, 0), (8, 0), (5, 3), (1, 2), (9, 2)]
    if x != 5:
        positions = [(np.random.randint(1, 20), np.random.randint(20, 30)) for _ in range(x)]

    for i, pos in enumerate(positions):
        G.add_node(i, pos=pos)
        for j in range(x):
            G.add_edge(i, j, weight=(matrix[i][j]))

    weights = nx.get_edge_attributes(G, 'weight')
    pos = nx.get_node_attributes(G, 'pos')
    nx.draw(G, pos, with_labels=True)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=weights)
    plt.show()


def cost(steps):
    steps = list(steps)

    cities = [city for city in range(len(L))]

    sum_all = 0
    for i in range(len(cities)):
        if i != len(steps) - 1:
            x, y = steps[i], steps[i + 1]
            if x > 4 or y > 4:
                print(x, y)
                raise False
            step_cost = L[x][y]
            sum_all += step_cost
    return sum_all


def main():
    show_matrix(L)
    hs = HS(N=6, M=2, HMCR=0.90, PAR=0, L=(0, 5), max_iter=100, end_condition=11)
    hs.cost_func = cost
    hs.solve()
    hs.results()
    hs.plot_results()


if __name__ == '__main__':
    main()
