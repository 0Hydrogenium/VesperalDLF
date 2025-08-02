import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd

from utils.visualizer.Visualizer import Visualizer


class PublicOpinionVisualizer(Visualizer):
    def __init__(self, dpi=300):
        super().__init__(dpi)

    def distribution_bar(self, distribution_dict, figsize, save_path):
        plt.figure(figsize=figsize)

        plt.bar(
            list(distribution_dict.keys()),
            list(distribution_dict.values()),
            width=0.5,
            color=self.color_cfg["base__color_3"],
            # edgecolor='black',
            # linewidth=2
        )

        plt.xlabel("Date", fontsize=14, labelpad=16)
        plt.ylabel("Blog Num", fontsize=14, labelpad=16)
        plt.xticks(rotation=90)
        plt.legend(loc="upper right", fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.savefig(save_path)
        plt.close()

    def time_curve(self, time_dict, figsize, save_path):
        plt.figure(figsize=figsize)

        for i, (label, time_count_dict) in enumerate(time_dict.items()):
            plt.plot(
                list(time_count_dict.keys())[::2],
                list(time_count_dict.values())[::2],
                label=label,
                color=self.color_cfg[f"base__color_{i + 1}"]
            )

        plt.xlabel("Date", fontsize=14, labelpad=16)
        plt.ylabel("Blog Num", fontsize=14, labelpad=16)
        plt.xticks(rotation=90)
        plt.legend(loc="upper right", fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
