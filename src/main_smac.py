import numpy as np

from sklearn.datasets import make_blobs

from ConfigSpace import Configuration, ConfigurationSpace

def get_cluster_centroids(config: Configuration):

    binary_centroids = [[int(bit)for bit in bin(i)[2:].zfill(config["n_features"])] for i in range(2 ** config["n_features"])]
    if config["n_clusters"] < len(binary_centroids):
        return binary_centroids[:config["n_clusters"]]
    else:
        indexer = lambda cluster_idx : ((cluster_idx - len(binary_centroids)) % (len(binary_centroids) - 1)) + 1
        multiplier = lambda cluster_idx : int((cluster_idx - len(binary_centroids)) / (len(binary_centroids) - 1)) + 2
        return np.array([
            np.array(binary_centroids[cluster]) if cluster < len(binary_centroids) else np.array(binary_centroids[indexer(cluster)]) * multiplier(cluster)
            for cluster in range(config["n_clusters"])
            ])

def get_cluster_instances(config: Configuration):

    instances_per_cluster = int(config["n_instances"] / config["n_clusters"])
    extra_instances = config["n_instances"] % config["n_clusters"]
    to_return = [instances_per_cluster] * config["n_clusters"]
    return [(elem + 1) if idx < extra_instances else (elem) for idx, elem in enumerate(to_return)]

def generate_clusters(config: Configuration):
    X, y = make_blobs(
        n_samples=get_cluster_instances(config),
        n_features=config["n_features"],
        centers=get_cluster_centroids(config),
        cluster_std=[config["n_instances"]] * config["n_clusters"],
        shuffle=True,
        random_state=42
    )



if __name__ == "__main__":

    cs = ConfigurationSpace({
    "n_features": (2, 13),
    "n_instances": [100, 500, 1000, 5000],
    "n_clusters": (2, 13),
    "cluster_std": (1., 2.),
    "noisy_features": (0., 1.),
    "correlated_features": (0., 1.),
    "distorted_features": (0., 1.),
    })

    [generate_clusters(config) for config in cs.sample_configuration(20)]