import numpy as np
import torch
from sklearn.cluster import KMeans
import time
import matplotlib.pyplot as plt

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

def get_data(N):
   return torch.rand(N, 3)

def main(CONDENSE_FACTOR=0.5, NUM_NEIGHBORS=5, NUM_POINTS=1000):
   data = get_data(NUM_POINTS).to("cpu")  # KMeans must be on CPU

   # Cluster data
   data = KMeans(n_clusters=int(len(data) * CONDENSE_FACTOR)).fit(data.numpy()).cluster_centers_
   data = torch.tensor(data, device=device)


   N = data.shape[0]
   D = torch.cdist(data, data)

   nearest_indices = D.argsort(dim=1)[:, 1 : NUM_NEIGHBORS + 1]  # Exclude self


   A = torch.zeros((N, N), dtype=torch.int32, device=device)
   rows = torch.arange(N, device=device).repeat_interleave(NUM_NEIGHBORS)
   cols = nearest_indices.flatten()
   A[rows, cols] = 1
   A = torch.maximum(A, A.T)


   G = torch.where(A > 0, D, float("inf"))
   G.fill_diagonal_(0)

   for point in range(N):
       G = torch.minimum(G, G[:, point].unsqueeze(1) + G[point, :])

   return G.cpu().numpy()


def test():
    condense_factors = np.arange(0.1, 0.95, 0.05)
    num_neighbors = range(2, 21)
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