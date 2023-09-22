import numpy as np
import pandas as pd

from sklearn.datasets import make_blobs
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt

from ConfigSpace import Configuration, ConfigurationSpace


def get_cluster_centroids(config: Configuration):
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


def generate_clusters(config: Configuration):
    X, y = make_blobs(
        n_samples=get_cluster_instances(config),
        n_features=config["n_features"],
        centers=get_cluster_centroids(config),
        cluster_std=[config["n_instances"]] * config["n_clusters"],
        shuffle=True,
        random_state=42,
    )
    print(X)
    print(y)
    return pd.DataFrame(
        np.concatenate([X, np.array([y]).T], axis=1),
        columns=[str(idx) for idx in range(X.shape[1])] + ["target"],
    )


colors = np.array(
    [
        "blue",
        "orange",
        "green",
        "red",
        "purple",
        "brown",
        "pink",
        "grey",
        "olive",
        "cyan",
        "indigo",
        "black",
    ]
)


def single_plot(ax, df, target_column, unique_clusters, type):
    if type != "PARA":
        if df.shape[1] > 3:
            Xt = pd.concat(
                [
                    pd.DataFrame(
                        TSNE(n_components=2, random_state=42).fit_transform(
                            # pd.read_csv(
                            #     "results/optimization/smbo/details/ecoli_sil/ecoli_sil_478_X_normalize.csv"
                            # ).to_numpy(),
                            df.iloc[:, :-1].to_numpy(),
                            df.iloc[:, -1].to_numpy(),
                        )
                        if type == "TSNE"
                        else PCA(n_components=2, random_state=42).fit_transform(
                            # pd.read_csv(
                            #     "results/optimization/smbo/details/ecoli_sil/ecoli_sil_478_X_normalize.csv"
                            # ).to_numpy(),
                            df.iloc[:, :-1].to_numpy(),
                            df.iloc[:, -1].to_numpy(),
                        ),
                        columns=[f"{type}_0", f"{type}_1"],
                    ),
                    df[target_column],
                ],
                axis=1,
            )
        else:
            Xt = df
        # print(Xt)

        for i, cluster_label in enumerate(unique_clusters):
            cluster_data = Xt[Xt[target_column] == cluster_label]
            plt.scatter(
                cluster_data.iloc[:, 0],
                cluster_data.iloc[:, 1],
                c=[colors[i]] * cluster_data.shape[0],
                label=f"Cluster {cluster_label}",
            )

            n_selected_features = Xt.shape[1]
            Xt = Xt.iloc[:, :n_selected_features]
            min, max = Xt.min().min(), Xt.max().max()
            range = (max - min) / 10
            xs = Xt.iloc[:, 0]
            ys = (
                [(max + min) / 2] * Xt.shape[0]
                if n_selected_features < 2
                else Xt.iloc[:, 1]
            )
            ax.set_xlim([min - range, max + range])
            ax.set_ylim([min - range, max + range])
            ax.set_xlabel(list(Xt.columns)[0], fontsize=16)
            ax.set_ylabel(
                "None" if n_selected_features < 2 else list(Xt.columns)[1], fontsize=16
            )
    else:
        ax = pd.plotting.parallel_coordinates(df, "target", color=colors)
    ax.set_title(type)


def plot_cluster_data(df, target_column):
    """
    Plot a dataframe with clustering results using a scatter plot.

    Parameters:
    df (pandas.DataFrame): The dataframe containing the data to be plotted.
    target_column (str): The name of the column that contains the cluster labels.

    Returns:
    None
    """

    # Make sure the target_column is in the dataframe
    if target_column not in df.columns:
        raise ValueError(f"'{target_column}' column not found in the dataframe.")

    # Create a scatter plot for each cluster
    unique_clusters = df[target_column].unique()

    fig = plt.figure(figsize=(24, 4.5))
    for idx, subplot_type in enumerate(["TSNE", "PCA", "PARA"]):
        ax = fig.add_subplot(1, 3, idx + 1)
        single_plot(
            ax=ax,
            df=df,
            target_column=target_column,
            unique_clusters=unique_clusters,
            type=subplot_type,
        )

    return fig


if __name__ == "__main__":
    cs = ConfigurationSpace(
        {
            "n_features": (2, 12),
            "n_instances": [100, 500, 1000, 5000],
            "n_clusters": (2, 12),
            "cluster_std": (1.0, 2.0),
            "noisy_features": (0.0, 1.0),
            "correlated_features": (0.0, 1.0),
            "distorted_features": (0.0, 1.0),
        }
    )

    figs = [
        plot_cluster_data(generate_clusters(config), "target")
        for config in cs.sample_configuration(20)
    ]

    for idx, fig in enumerate(figs):
        fig.savefig(f"/home/{idx}.png", dpi=300, bbox_inches="tight")
        plt.close(fig)
