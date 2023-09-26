import json
import os
import random
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import make_blobs
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

from ConfigSpace import Configuration, ConfigurationSpace

from tqdm import tqdm

from utils.cluster import create_configs, generate_clusters
from utils.common import make_dir, json_to_csv
from utils.plot import plot_cluster_data


if __name__ == "__main__":
    output_path = make_dir(os.path.join("/", "home", "results"))
    seed = 42
    random.seed(seed)
    np.random.seed(seed)

    cs = ConfigurationSpace(
        {
            "n_features": (2, 6),
            # "n_instances": [100, 500, 1000, 5000],
            "n_instances": [100, 200, 500, 1000, 2000],
            "n_clusters": (2, 6),
            # "cluster_std": (1.0, 1.5),
            "cluster_std": (0.1, 0.3),
            "noisy_features": (0.0, 0.5),
            "correlated_features": (0.0, 1.0),
            "distorted_features": (0.0, 1.0),
        },
        seed=seed,
    )

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")

        print("--- GENERATE CONFIGURATIONS ---")

        configs = create_configs(cs)

        with open(os.path.join(output_path, "configs.json"), "w") as file:
            json.dump(configs, file)
        json_to_csv(list(configs.values()), os.path.join(output_path, "configs.csv"))

        print("\tDone.")
        print("--- GENERATE CLUTERSINGS ---")

        with tqdm(total=len(configs)) as pbar:
            clusterings = [
                generate_clusters(config, pbar) for config in configs.values()
            ]

        print("\tDone.")
        print("--- EXPORT CLUTERSINGS ---")

        with tqdm(total=len(clusterings)) as pbar:
            for id_clustering, clustering_dict in enumerate(clusterings):
                for label, clustering in clustering_dict.items():
                    clustering_name = f"syn{id_clustering}_{label}"
                    clustering.to_csv(
                        os.path.join(
                            make_dir(os.path.join(output_path, "raw")),
                            f"{clustering_name}.csv",
                        ),
                        index=False,
                        header=None,
                    )
                    fig = plot_cluster_data(clustering, "target")
                    fig.savefig(
                        os.path.join(
                            make_dir(os.path.join(output_path, "img")),
                            f"{clustering_name}.png",
                        ),
                        dpi=300,
                        bbox_inches="tight",
                    )
                    pbar.update()
                    plt.close(fig)

        print("\tDone.")
