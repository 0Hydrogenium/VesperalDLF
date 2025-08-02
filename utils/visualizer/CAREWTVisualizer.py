import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
from tqdm import tqdm

from utils.time_series.TimeSeriesTool import TimeSeriesTool
from utils.visualizer.Visualizer import Visualizer


class CAREWTVisualizer(Visualizer):
    def __init__(self, dpi=300):
        super().__init__(dpi)

    def wind_turbines_time_curve(self, farm_datasets, alpha, s, figsize, save_path):
        for (dataset_name, dataset) in tqdm(farm_datasets.items()):
            plt.figure(figsize=figsize)

            indices = dataset["time_stamp"].values

            plt.axhline(y=0, color=self.color_cfg['base__color_2'], linestyle='--', alpha=0.5)
            plt.axhline(y=1, color=self.color_cfg['base__color_2'], linestyle='--', alpha=0.5)
            plt.axhline(y=2, color=self.color_cfg['base__color_2'], linestyle='--', alpha=0.5)
            plt.axhline(y=3, color=self.color_cfg['base__color_2'], linestyle='--', alpha=0.5)
            first_idx = TimeSeriesTool.match_first_index(1, dataset["label"].values)
            if first_idx != -1:
                plt.axvline(x=first_idx, color=self.color_cfg['base__color_5'], linestyle='--', alpha=0.5)
            last_idx = TimeSeriesTool.match_last_index(1, dataset["label"].values)
            last_idx = len(dataset) - 1 if last_idx == -1 else last_idx
            plt.axvline(x=last_idx, color=self.color_cfg['base__color_5'], linestyle='--', alpha=0.5)

            plt.scatter(
                indices,
                dataset["label"].values,
                color=self.color_cfg["base__color_1"],
                alpha=alpha,
                s=s,
                label=f"label"
            )
            plt.scatter(
                indices,
                dataset["status_type_id"].values + 2,
                color=self.color_cfg["base__color_3"],
                alpha=alpha,
                s=s,
                label=f"status_type_id"
            )

            plt.xticks([])
            plt.yticks([])
            plt.xlabel("Time", fontsize=14, labelpad=16)
            plt.ylabel("", fontsize=14, labelpad=16)
            plt.legend(loc="upper right", fontsize=14)
            plt.savefig(save_path.replace("@", str(dataset_name)))
            plt.close()


