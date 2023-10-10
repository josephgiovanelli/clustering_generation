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

from ConfigSpace import Configuration, ConfigurationSpace, Float, Integer, EqualsCondition, InCondition

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
            "n_features": Integer("n_features", (2, 10), log=True),
            # "n_instances": [100, 200, 500, 1000, 2000, 5000],
            "n_instances": Integer("n_instances", (100, 5000), log=True),
            # "n_clusters": (2, 6),
            "n_clusters_ratio": Float("n_clusters_ratio", (0.05, 1.0), log=True),
            # "cluster_std": (1.0, 1.5),
            "cluster_std": (0.1, 0.3),

            "noisy_features": Float("noisy_features", (0.2, 0.5), log=False),

            "correlated_features": Float("correlated_features", (0.2, 0.5), log=False),

            "distorted_features": Float("distorted_features", (0.2, 0.5), log=False),

            "kind": ["100", "010", "001", "110", "101", "011", "111"],
        },
        seed=seed,
    )

    cs.add_condition(InCondition(cs['noisy_features'], cs['kind'], ["100", "110", "101", "111"]))
    cs.add_condition(InCondition(cs['correlated_features'], cs['kind'], ["010", "110", "011", "111"]))
    cs.add_condition(InCondition(cs['distorted_features'], cs['kind'], ["001", "101", "011", "111"]))


    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")

        tot_configs = 20
        current_round = 0
        final_configs = []

        print("--- GENERATE CONFIGURATIONS ---")
        with tqdm(total=tot_configs) as pbar:
            while len(final_configs) < tot_configs:

                current_configs = create_configs(cs=cs, n_configs=tot_configs-len(final_configs))

                clusterings = []
                current_round += 1
                for config in current_configs:
                    try:
                        config["round"] = current_round
                        clusterings.append(generate_clusters(config))
                        final_configs.append(config)
                        pbar.update()
                    except:
                        pass

        with open(os.path.join(output_path, "configs.json"), "w") as file:
            json.dump({idx: config for idx, config in enumerate(final_configs)}, file)

        to_export = ["n_instances","n_clusters","n_clusters_ratio","cluster_std","support_total_features","n_features","support_noisy_features","support_correlated_features","support_distorted_features","noisy_features","correlated_features","distorted_features", "round"]
        pd.read_json(os.path.join(output_path, "configs.json")).transpose()[to_export].to_csv(
            os.path.join(output_path, "configs.csv")
        )

        print("--- GENERATE CLUSTERINGS ---")
        with tqdm(total=tot_configs) as pbar:
            for id_clustering, clustering_dict in enumerate(clusterings):
                for label, clustering in clustering_dict.items():
                    suffix = "" if label == "final" else f"_{label}"
                    output_folder = "final" if label == "final" else "raw"
                    clustering_name = f"syn{id_clustering}{suffix}"
                    clustering.to_csv(
                        os.path.join(
                            make_dir(os.path.join(output_path, output_folder)),
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
                    plt.close(fig)
                pbar.update()