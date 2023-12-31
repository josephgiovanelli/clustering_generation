import math
import random
import numpy as np
import pandas as pd

from sklearn.datasets import make_blobs, make_classification
from sklearn.discriminant_analysis import StandardScaler
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import MinMaxScaler

from ConfigSpace import Configuration
from smac import BlackBoxFacade, Scenario


def create_configs(cs, n_configs):
    configs = [
        config.get_dictionary()
        for config in BlackBoxFacade.get_initial_design(
            Scenario(cs), n_configs=n_configs
        ).select_configurations()
    ]

    for config in configs:
        config["cluster_std"] = round(config["cluster_std"], 2)
        try:
            config["noisy_features"] = round(config["noisy_features"], 2)
            config["support_noisy_features"] = max(
                1, int(config["noisy_features"] * config["n_features"])
            )
        except:
            config["noisy_features"] = 0.0
            config["support_noisy_features"] = 0

        try:
            config["correlated_features"] = round(config["correlated_features"], 2)
            config["support_correlated_features"] = max(
                1, int(config["correlated_features"] * config["n_features"])
            )
        except:
            config["correlated_features"] = 0.0
            config["support_correlated_features"] = 0

        config["support_total_features"] = (
            config["n_features"]
            + config["support_noisy_features"]
            + config["support_correlated_features"]
        )

        try:
            config["distorted_features"] = round(config["distorted_features"], 2)
            config["support_distorted_features"] = max(
                1, int(config["distorted_features"] * config["support_total_features"])
            )
        except:
            config["distorted_features"] = 0.0
            config["support_distorted_features"] = 0

        config["n_clusters_ratio"] = round(config["n_clusters_ratio"], 2)
        config["n_clusters"] = max(
            2, int(config["n_clusters_ratio"] * math.sqrt(config["n_instances"]))
        )

        centroids = get_cluster_centroids(config)

        config["support_centroids"] = (
            centroids if type(centroids) == list else centroids.tolist()
        )
        config["support_instances"] = list(get_cluster_instances(config))
        config["support_cluster_std"] = [config["cluster_std"]] * config["n_clusters"]

        config["support_total_features"] += config["support_distorted_features"]

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


def generate_clusters(config):
    X, y = make_blobs(
        n_samples=config["support_instances"],
        n_features=config["n_features"],
        centers=config["support_centroids"],
        cluster_std=config["support_cluster_std"],
        shuffle=True,
        random_state=42,
    )

    to_df = lambda first, second: pd.DataFrame(
        np.concatenate([first, np.array([second]).T.astype(int)], axis=1),
        columns=[str(idx) for idx in range(first.shape[1])] + ["target"],
    )

    dict_X = {"original": X.copy(), "final": X.copy()}
    step = 0
    dict_to_return = {f"{step}original": to_df(X, y)}

    if config["support_noisy_features"] > 0:
        dict_X["noisy"] = (
            np.random.rand(
                dict_X["original"].shape[0], config["support_noisy_features"]
            )
            * (dict_X["original"].max().max() - dict_X["original"].min().min())
        ) + dict_X["original"].min().min()

        dict_X["final"] = np.concatenate([dict_X["final"], dict_X["noisy"]], axis=1)
        step += 1
        dict_to_return[f"{step}noisy"] = to_df(dict_X["final"], y)

    if config["support_correlated_features"] > 0:
        std = 0.02
        dict_X["correlated"] = np.array(
            [
                dict_X["original"][:, feature]
                + np.random.normal(0, std, dict_X["original"].shape[0])
                for feature in random.sample(
                    range(0, dict_X["original"].shape[1] - 1),
                    config["support_correlated_features"],
                )
            ]
        ).T

        dict_X["final"] = np.concatenate(
            [dict_X["final"], dict_X["correlated"]], axis=1
        )
        step += 1
        dict_to_return[f"{step}correlated"] = to_df(dict_X["final"], y)

    if config["support_distorted_features"] > 0:
        dict_X["distorted"] = dict_X["final"]
        for feature in random.sample(
            range(0, dict_X["final"].shape[1] - 1), config["support_distorted_features"]
        ):
            dict_X["distorted"][:, feature] *= random.randint(2, 10)

        dict_X["final"] = dict_X["distorted"]
        step += 1
        dict_to_return[f"{step}distorted"] = to_df(dict_X["final"], y)

    step += 1
    dict_to_return[f"{step}final"] = to_df(dict_X["final"], y)

    initial_X = dict_to_return["0original"].copy().iloc[:, :-1].to_numpy()
    final_X = dict_to_return[f"{step}final"].copy().iloc[:, :-1].to_numpy()
    initial_Xt = (
        TSNE(n_components=2, random_state=42).fit_transform(initial_X)
        if initial_X.shape[1] > 2
        else initial_X
    )
    final_Xt = (
        TSNE(n_components=2, random_state=42).fit_transform(final_X)
        if final_X.shape[1] > 2
        else final_X
    )

    config["initial_sil"] = round(silhouette_score(initial_Xt, y), 2).astype(np.float64)
    config["final_sil"] = round(silhouette_score(final_Xt, y), 2).astype(np.float64)

    is_valid = (
        config["initial_sil"] > 0.4
        and config["final_sil"] > 0.1
        and config["initial_sil"] - config["final_sil"] > 0.1
        and config["initial_sil"] - config["final_sil"] < 0.7
    )
    # print(f"""valid: {is_valid}\tinitial sil: {round(config["initial_sil"], 2)},\tfinal sil: {round(config["final_sil"], 2)}""")
    if is_valid:
        return dict_to_return
    else:
        raise Exception("Conf not valid")
