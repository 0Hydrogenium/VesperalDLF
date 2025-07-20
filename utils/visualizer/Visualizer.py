import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd

from utils.preference_color.PreferenceColor import PreferenceColor


class Visualizer:
    def __init__(self, dpi: int = 300):
        self.dpi = dpi
        self.color_cfg = PreferenceColor.cfg

    def deviation_scatter_plot(self, data_y, data_y_proba, alpha, s, figsize, save_path):
        plt.figure(figsize=figsize)

        n_idx = np.where(data_y == 0)[0].tolist()
        p_idx = np.where(data_y == 1)[0].tolist()
        n_data_y_proba = data_y_proba[n_idx]
        p_data_y_proba = data_y_proba[p_idx]

        plt.scatter(
            np.arange(len(n_data_y_proba)),
            n_data_y_proba,
            color=self.color_cfg["base__color_3"],
            alpha=alpha,
            s=s,
            label=f"negative deviation"
        )
        plt.scatter(
            np.arange(len(p_data_y_proba)),
            p_data_y_proba,
            color=self.color_cfg["base__color_1"],
            alpha=alpha,
            s=s,
            label=f"positive deviation"
        )

        plt.xlabel("Deviation", fontsize=14, labelpad=16)
        plt.ylabel("Value", fontsize=14, labelpad=16)
        plt.legend(loc="upper right", fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.savefig(save_path)
        plt.close()

    def pr_curve_on_thresholds(self, recalls, precisions, pr_ap, baseline, figsize, save_path):
        plt.figure(figsize=figsize)

        plt.plot(
            recalls,
            precisions,
            label=f"PR(AP={pr_ap:.4f})",
            color=self.color_cfg["base__color_1"]
        )
        plt.axhline(
            y=baseline,
            linestyle='-',
            label=f"Random",
            color=self.color_cfg["base__color_3"]
        )

        plt.xlabel("Recall", fontsize=14, labelpad=16)
        plt.ylabel("Precision", fontsize=14, labelpad=16)
        plt.legend(loc="upper right", fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.savefig(save_path)
        plt.close()

    def roc_curve_on_thresholds(self, fpr, tpr, roc_auc, figsize, save_path):
        plt.figure(figsize=figsize)

        plt.plot(
            fpr,
            tpr,
            label=f"ROC(AUC={roc_auc:.4f})",
            color=self.color_cfg["base__color_1"],
            lw=2,
        )
        plt.plot(
            [0, 1],
            [0, 1],
            label="Random",
            color=self.color_cfg["base__color_3"],
            lw=2,
            linestyle='--'
        )

        plt.xlim([0.0, 1.0])
        plt.ylim([0., 1.05])
        plt.xlabel("FPR", fontsize=14, labelpad=16)
        plt.ylabel("TPR", fontsize=14, labelpad=16)
        plt.legend(loc="lower right", fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.savefig(save_path)
        plt.close()

    def loss_plot_train_test_curve_with_std(self, train_loss_mean, train_loss_std, test_loss_mean, test_loss_std, scale: float, alpha: float, figsize, save_path):
        plt.figure(figsize=figsize, dpi=self.dpi)

        plt.plot(
            np.arange(len(train_loss_mean)),
            train_loss_mean,
            label="train",
            color=self.color_cfg["base__color_1"]
        )
        plt.fill_between(
            np.arange(len(train_loss_mean)),
            train_loss_mean - scale * train_loss_std,
            train_loss_mean + scale * train_loss_std,
            color=self.color_cfg["base__color_1"],
            alpha=alpha
        )

        plt.plot(
            np.arange(len(test_loss_mean)),
            test_loss_mean,
            label="test",
            color=self.color_cfg["base__color_3"]
        )
        plt.fill_between(
            np.arange(len(test_loss_mean)),
            test_loss_mean - scale * test_loss_std,
            test_loss_mean + scale * test_loss_std,
            color=self.color_cfg["base__color_3"],
            alpha=alpha
        )

        plt.xlabel("Epoch", fontsize=14, labelpad=16)
        plt.ylabel("Loss", fontsize=14, labelpad=16)
        plt.legend(loc="upper right", fontsize=14)
        plt.savefig(save_path)
        plt.close()

    def pareto_projection_scatter_3d_plot(self, metrics: pd.DataFrame, label_list, alpha, s, figsize, save_path):
        # 绘制帕累托前沿二维投影图
        metrics_array = metrics.to_numpy()

        assert metrics.shape[1] == 3, "pareto projection scatter 3d plot input metrics dimension error"

        # 初始化所有点都是帕累托点
        pareto_mask = np.ones(metrics_array.shape[0], dtype=bool)
        for i, point in enumerate(metrics_array):
            # 当前点若为帕累托点则进行检查
            if pareto_mask[i]:
                # 检查是否存在其他点支配当前点
                pareto_mask[i] = not np.any(
                    (metrics_array[:, 0] > point[0]) &  # 其他点的第一维 > 当前点第一维
                    (metrics_array[:, 1] > point[1]) &  # 其他点的第二维 > 当前点第二维
                    (metrics_array[:, 2] > point[2])  # 其他点的第三维 > 当前点第三维
                )
        pareto_points = metrics_array[pareto_mask]

        fig = plt.figure(figsize=figsize, facecolor="white", dpi=self.dpi)
        ax = fig.add_subplot(111, projection="3d")
        # 设置坐标轴背景为白色
        ax.set_facecolor("white")

        # 绘制xy平面的投影和帕累托点
        xy_offset = 0.02
        ax.scatter(
            metrics_array[:, 0] + xy_offset,
            metrics_array[:, 1] + xy_offset,
            np.zeros_like(metrics_array[:, 2]),
            color=self.color_cfg["base__color_1"],
            alpha=alpha,
            s=s,
            label="solution xy"
        )
        ax.scatter(
            pareto_points[:, 0] + xy_offset,
            pareto_points[:, 1] + xy_offset,
            np.zeros_like(pareto_points[:, 2]),
            color=self.color_cfg["base__color_1"],
            alpha=alpha,
            edgecolors=self.color_cfg["base__color_5"],
            s=s,
            label="pareto front xy"
        )

        # 绘制xz平面的投影和帕累托点
        ax.scatter(
            metrics_array[:, 0],
            np.zeros_like(metrics_array[:, 1]),
            metrics_array[:, 2],
            color=self.color_cfg["base__color_4"],
            alpha=alpha,
            s=s,
            label="solution xz"
        )
        ax.scatter(
            pareto_points[:, 0],
            np.zeros_like(pareto_points[:, 1]),
            pareto_points[:, 2],
            color=self.color_cfg["base__color_4"],
            alpha=alpha,
            edgecolors=self.color_cfg["base__color_5"],
            s=s,
            label="pareto front xz"
        )

        # 绘制yz平面的投影和帕累托点
        ax.scatter(
            np.zeros_like(metrics_array[:, 0]),
            metrics_array[:, 1],
            metrics_array[:, 2],
            color=self.color_cfg["base__color_3"],
            alpha=alpha,
            s=s,
            label="solution yz"
        )
        ax.scatter(
            np.zeros_like(pareto_points[:, 0]),
            pareto_points[:, 1],
            pareto_points[:, 2],
            color=self.color_cfg["base__color_3"],
            alpha=alpha,
            edgecolors=self.color_cfg["base__color_5"],
            s=s,
            label="pareto front yz"
        )

        ax.legend(loc="best")
        ax.set_xlabel(label_list[0], labelpad=12)
        ax.set_ylabel(label_list[1], labelpad=12)
        ax.set_zlabel(label_list[2], labelpad=12)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        ax.set_xticks(np.arange(0, 1, 0.1))
        ax.set_yticks(np.arange(0, 1, 0.1))
        ax.view_init(azim=45)
        plt.tight_layout()
        plt.savefig(save_path)

    def bar_plot_param_importance(self, study, optimize_metric, save_path):
        # 绘制各个超参数对结果影响的重要性柱状图
        optuna.visualization.matplotlib.plot_param_importances(study)
        ax = plt.gca()
        ax.set_title("")
        ax.legend(labels=optimize_metric)
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
