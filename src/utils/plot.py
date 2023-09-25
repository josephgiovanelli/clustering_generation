import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

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
                            df.iloc[:, :-1].to_numpy(),
                            df.iloc[:, -1].to_numpy(),
                        )
                        if type == "TSNE"
                        else PCA(n_components=2, random_state=42).fit_transform(
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
