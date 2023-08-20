import matplotlib.pyplot as plt


def plot_progress(X, centroids, previous_centroids, idx, K, i):
    """
    Plot the progress of K-Means algorithm.

    Args:
        X (ndarray): Input data matrix with shape (m, n)
        centroids (ndarray): Current centroids with shape (K, n)
        previous_centroids (ndarray): Centroids from previous iteration with shape (K, n)
        idx (ndarray): Index of the closest centroid for each example
        K (int): Number of centroids
        i (int): Current iteration number
    """
    # Plot data points
    plt.scatter(X[:, 0], X[:, 1], facecolors='none', edgecolors=c, linewidth=0.1, alpha=0.7)

    # Plot centroids
    plt.scatter(centroids[:, 0], centroids[:, 1], marker='x', c='k', linewidths=3)

    # Plot history of centroids with lines
    for j in range(centroids.shape[0]):
        plt.plot([centroids[j, 0], previous_centroids[j, 0]],
                 [centroids[j, 1], previous_centroids[j, 1]], '-k', linewidth=1)

    plt.title("Iteration number %d" % i)
    plt.show()
