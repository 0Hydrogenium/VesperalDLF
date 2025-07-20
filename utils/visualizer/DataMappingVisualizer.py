import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd

from utils.visualizer.Visualizer import Visualizer


class DataMappingVisualizer(Visualizer):
    def __init__(self, dpi=300):
        super().__init__(dpi)

    def line_chart_mapping_y_distribution_param_trend(self, baselines, etas, target_value, figsize, alpha, s, save_path1, save_path2):
        plt.figure(figsize=figsize, dpi=self.dpi)

        plt.plot(
            np.arange(1, len(baselines) + 1),
            baselines,
            'o-',
            color=self.color_cfg["base__color_4"],
            alpha=alpha,
            markersize=s,
            label=f"fault {target_value} baseline"
        )

        plt.grid(alpha=0.3)
        plt.xlabel("Index", fontsize=14, labelpad=16)
        plt.ylabel("Value", fontsize=14, labelpad=16)
        plt.legend(loc="best", fontsize=14)
        plt.savefig(save_path1)
        plt.close()

        plt.figure(figsize=figsize, dpi=self.dpi)

        plt.plot(
            np.arange(1, len(etas) + 1),
            etas,
            'o-',
            color=self.color_cfg["base__color_5"],
            alpha=alpha,
            markersize=s,
            label=f"fault {target_value} etas"
        )

        plt.grid(alpha=0.3)
        plt.xlabel("Index", fontsize=14, labelpad=16)
        plt.ylabel("Value", fontsize=14, labelpad=16)
        plt.legend(loc="best", fontsize=14)
        plt.savefig(save_path2)
        plt.close()

    def line_chart_mapping_y_distribution_discrete_with_continuous(self, data_y, mapped_data_y, target_value, figsize, alpha, s, save_path):
        plt.figure(figsize=figsize, dpi=self.dpi)

        # 原始离散点
        plt.scatter(
            np.arange(len(data_y)),
            data_y,
            color=self.color_cfg["base__color_1"],
            alpha=alpha,
            s=s,
            label=f"fault {target_value} raw"
        )

        # 映射后的连续曲线
        plt.plot(
            mapped_data_y,
            '-',
            color=self.color_cfg["base__color_3"],
            alpha=alpha,
            linewidth=2,
            label=f"fault {target_value} mapped"
        )

        plt.grid(alpha=0.3)
        plt.xlabel("Timestamp", fontsize=14, labelpad=16)
        plt.ylabel("Target", fontsize=14, labelpad=16)
        plt.legend(loc="lower right", fontsize=14)
        plt.savefig(save_path)
        plt.close()
