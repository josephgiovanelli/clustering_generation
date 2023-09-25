import numpy as np
import pandas as pd

from sklearn.datasets import make_blobs, make_classification
from sklearn.discriminant_analysis import StandardScaler
from sklearn.preprocessing import MinMaxScaler

from ConfigSpace import Configuration


def create_configs(cs):
    configs = {
        idx: config.get_dictionary()
        for idx, config in enumerate(cs.sample_configuration(20))
    }

    for _, config in configs.items():
        centroids = get_cluster_centroids(config)
        config["support_centroids"] = (
            centroids if type(centroids) == list else centroids.tolist()
        )
        config["support_instances"] = list(get_cluster_instances(config))
        config["support_cluster_std"] = [config["cluster_std"]] * config["n_clusters"]
        config["support_noisy_features"] = int(
            config["noisy_features"] * config["n_features"]
        )

    return configs


def get_cluster_centroids(config):
    binary_centroids = [
        [int(bit) for bit in bin(i)[2:].zfill(config["n_features"])]
        for i in range(2 ** config["n_features"])
    ]
    if config["n_clusters"] < len(binary_centroids):
        return binary_centroids[: config["n_clusters"]]
    else:
        indexer = (
            lambda cluster_idx: (
                (cluster_idx - len(binary_centroids)) % (len(binary_centroids) - 1)
            )
            + 1
        )
        multiplier = (
            lambda cluster_idx: int(
                (cluster_idx - len(binary_centroids)) / (len(binary_centroids) - 1)
            )
            + 2
        )
        return np.array(
            [
                np.array(binary_centroids[cluster])
                if cluster < len(binary_centroids)
                else np.array(binary_centroids[indexer(cluster)]) * multiplier(cluster)
                for cluster in range(config["n_clusters"])
            ]
        )


def get_cluster_instances(config: Configuration):
    instances_per_cluster = int(config["n_instances"] / config["n_clusters"])
    extra_instances = config["n_instances"] % config["n_clusters"]
    to_return = [instances_per_cluster] * config["n_clusters"]
    return [
        (elem + 1) if idx < extra_instances else (elem)
        for idx, elem in enumerate(to_return)
    ]


def generate_clusters(config, pbar):
    X, y = make_blobs(
        n_samples=config["support_instances"],
        n_features=config["n_features"],
        centers=config["support_centroids"],
        cluster_std=config["support_cluster_std"],
        shuffle=True,
        random_state=42,
    )

    # X, y = make_classification(
    #     n_samples=config["n_instances"],
    #     n_features=config["n_features"]
    #     + int(config["n_features"] * config["noisy_features"])
    #     + int(config["n_features"] * config["correlated_features"]),
    #     n_informative=config["n_features"],
    #     n_redundant=int(config["n_features"] * config["correlated_features"]),
    #     n_repeated=int(config["n_features"] * config["noisy_features"]),
    #     # centers=config["support_centroids"],
    #     n_classes=config["n_clusters"],
    #     n_clusters_per_class=1,
    #     shuffle=True,
    #     random_state=42,
    # )

    if config["support_noisy_features"] > 0:
        print(X.max().max())
        print(X.min().min())
        X = np.concatenate(
            [
                X,
                # StandardScaler().fit_transform(
                (
                    np.random.rand(X.shape[0], config["support_noisy_features"])
                    * (X.max().max() - X.min().min())
                )
                + X.min().min()
                # ),
            ],
            axis=1,
        )

    pbar.update()

    return pd.DataFrame(
        np.concatenate([X, np.array([y]).T], axis=1),
        columns=[str(idx) for idx in range(X.shape[1])] + ["target"],
    )
