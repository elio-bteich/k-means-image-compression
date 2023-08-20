import numpy as np
import matplotlib.pyplot as plt
from utils import plot_progress


def find_closest_centroids(X, centroids):
    """
    Find the closest centroids for each example.

    Args:
        X (ndarray): Input data matrix with shape (m, n)
        centroids (ndarray): Centroid locations with shape (K, n)

    Returns:
        idx (ndarray): Index of the closest centroid for each example
    """
    K = centroids.shape[0]
    idx = np.zeros(X.shape[0], dtype=int)

    for i in range(len(X)):
        distances = []
        for j in range(K):
            distance = np.linalg.norm(X[i] - centroids[j])
            distances.append(distance)

        idx[i] = distances.index(min(distances))

    return idx


def compute_centroids(X, idx, K):
    """
    Compute new centroids based on assigned data points.

    Args:
        X (ndarray): Input data matrix with shape (m, n)
        idx (ndarray): Index of the closest centroid for each example
        K (int): Number of centroids

    Returns:
        centroids (ndarray): New centroids with shape (K, n)
    """
    m, n = X.shape
    centroids = np.zeros((K, n))

    for i in range(K):
        points = X[idx == i]
        centroids[i] = np.mean(points, axis=0)

    return centroids


def run_kmeans(X, initial_centroids, max_iters=8, display_progress=False):
    """
    Run the K-Means algorithm on data matrix X.

    Args:
        X (ndarray): Input data matrix with shape (m, n)
        initial_centroids (ndarray): Initial centroid locations with shape (K, n)
        max_iters (int): Maximum number of iterations
        plot_progress (bool): Whether to plot the progress

    Returns:
        centroids (ndarray): Final centroids with shape (K, n)
        idx (ndarray): Index of the closest centroid for each example
    """
    m, n = X.shape
    K = initial_centroids.shape[0]
    centroids = initial_centroids
    previous_centroids = centroids
    idx = np.zeros(m)

    for i in range(max_iters):
        print("K-Means iteration %d/%d" % (i, max_iters - 1))
        idx = find_closest_centroids(X, centroids)

        if display_progress:
            plot_progress(X, centroids, previous_centroids, idx, K, i)
            previous_centroids = centroids

        centroids = compute_centroids(X, idx, K)

    return centroids, idx


def init_centroids(X, K):
    """
    Initialize centroids for K-Means clustering.

    Args:
        X (ndarray): Input data matrix with shape (m, n)
        K (int): Number of centroids/clusters

    Returns:
        centroids (ndarray): Initialized centroids with shape (K, n)
    """
    randidx = np.random.permutation(X.shape[0])
    centroids = X[randidx[:K]]
    return centroids


if __name__ == "__main__":

    # Get user input for image path, number of centroids, and number of iterations
    image_path = input("Enter the path of the input image: ")
    num_centroids = int(input("Enter the number of centroids: "))
    num_iterations = int(input("Enter the number of iterations: "))

    # Load the image and preprocess
    original_img = plt.imread(image_path)
    original_img = original_img / 255
    X_img = np.reshape(original_img, (original_img.shape[0] * original_img.shape[1], 3))

    # Set parameters for K-Means
    K = num_centroids
    max_iters = num_iterations

    # Initialize centroids and run K-Means
    initial_centroids = init_centroids(X_img, K)
    centroids, idx = run_kmeans(X_img, initial_centroids, max_iters)

    # Generate and display the recovered image
    X_recovered = centroids[idx, :]
    X_recovered = np.reshape(X_recovered, original_img.shape)

    # Display the original and quantized images side by side
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(original_img)
    plt.title("Original Image")

    plt.subplot(1, 2, 2)
    plt.imshow(X_recovered)
    plt.title("Quantized Image")

    plt.tight_layout()
    plt.show()
