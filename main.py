import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
import time
import matplotlib.pyplot as plt



def get_data(N):
    return np.random.rand(N, 3)

def main(CONDENSE_FACTOR = 0.5, NUM_NEIGHBORS = 5, NUM_POINTS = 1000):
    data = get_data(NUM_POINTS)

    # Cluster data
    data = KMeans(n_clusters=int(len(data) * CONDENSE_FACTOR),).fit(data).cluster_centers_
    N = data.shape[0]

    E = euclidean_distances(data, data)

    nearest_indices = np.argsort(E, axis=1)[:, 1:NUM_NEIGHBORS + 1]  # Exclude self (1:N+1)

    A = np.zeros((N, N), dtype=int)  # Adjacency matrix

    rows = np.repeat(np.arange(N), NUM_NEIGHBORS)
    cols = nearest_indices.flatten()
    A[rows, cols] = 1
    A = np.maximum(A, A.T)

    G = np.where(A > 0, E, np.inf)
    np.fill_diagonal(G, 0)
    for point in range(len(G)):
        G = np.minimum(G, G[:, point][:, np.newaxis] + G[point, :])
    return G


def test():
    condense_factors = np.arange(0.1, 0.95, 0.05)
    num_neighbors = range(2, 21)
    # num_points = range(10000, 50000, 5000)
    num_points = range(1000, 10000, 1000)

    condense_times = []
    neighbor_times = []
    point_times = []

    for cf in condense_factors:
        start_time = time.time()
        main(CONDENSE_FACTOR=cf)
        condense_times.append(1000 * (time.time() - start_time))

    for nn in num_neighbors:
        start_time = time.time()
        main(NUM_NEIGHBORS=nn)
        neighbor_times.append(1000 * (time.time() - start_time))

    for n in num_points:
        start_time = time.time()
        main(NUM_POINTS=n)
        point_times.append((time.time() - start_time))

    plt.figure(figsize=(12, 6))

    plt.subplot(1, 3, 1)
    plt.plot(condense_factors, condense_times, marker='o')
    plt.title("Computation Time vs Condense Factor")
    plt.xlabel("Condense Factor")
    plt.ylabel("Time (ms)")
    plt.grid()

    plt.subplot(1, 3, 2)
    plt.plot(num_neighbors, neighbor_times, marker='o')
    plt.title("Computation Time vs Num Neighbors")
    plt.xlabel("Num Neighbors")
    plt.ylabel("Time (ms)")
    plt.grid()

    plt.subplot(1, 3, 3)
    plt.plot(num_points, point_times, marker='o')
    plt.title("Computation Time vs Num Points")
    plt.xlabel("Num Points")
    plt.ylabel("Time (sec)")
    plt.grid()

    # Show plots
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    test()