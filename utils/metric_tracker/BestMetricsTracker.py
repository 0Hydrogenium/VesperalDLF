import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pandas as pd


class BestMetricsTracker:
    def __init__(self):
        self.train_loss_list = []
        self.test_loss_list = []
        self.train_metrics_list = []
        self.test_metrics_list = []
        self.cfg_list = []
        self.epoch_list = []

    def add(self, train_loss, test_loss, train_metrics, test_metrics, cfg, epoch):
        self.train_loss_list.append(train_loss)
        self.test_loss_list.append(test_loss)
        self.train_metrics_list.append(train_metrics)
        self.test_metrics_list.append(test_metrics)
        self.cfg_list.append(cfg)
        self.epoch_list.append(epoch)

    def save_data(self, save_path):
        epoch_df = pd.DataFrame(data=self.epoch_list, columns=["epoch"])
        train_loss_df = pd.DataFrame(data=self.train_loss_list, columns=["train_loss"])
        test_loss_df = pd.DataFrame(data=self.test_loss_list, columns=["test_loss"])
        train_metrics_df = pd.DataFrame(self.train_metrics_list)
        train_metrics_df.columns = [f"train_{col}" for col in train_metrics_df.columns]
        test_metrics_df = pd.DataFrame(self.test_metrics_list)
        test_metrics_df.columns = [f"test_{col}" for col in test_metrics_df.columns]
        cfg_df = pd.DataFrame(self.cfg_list)

        combined_df = pd.concat([epoch_df, train_loss_df, test_loss_df, train_metrics_df, test_metrics_df, cfg_df], axis=1)
        combined_df.to_csv(save_path, index=False)

    def get_best(self, metrics_filter: list, optimize_direction: list):
        # 选取方法为：归一化+阈值筛选+Max-Min均衡
        # None为无阈值要求

        cfg_df = pd.DataFrame(self.cfg_list)
        metrics_df = pd.DataFrame(self.test_metrics_list)
        cfg_array = np.array(cfg_df)
        metrics_array = np.array(metrics_df)
        epoch_array = np.array(self.epoch_list)

        # 将所有最小化的指标值转化为最大化
        transformed_metrics = metrics_array.copy()
        for i, direction in enumerate(optimize_direction):
            if direction == "minimize":
                transformed_metrics[:, i] = - transformed_metrics[:, i]
                # 同时调整阈值
                metrics_filter[i] = - metrics_filter[i] if metrics_filter[i] is not None else None

        # 阈值筛选
        filter_mask = np.array([val is not None for val in metrics_filter])
        feasible_mask = np.ones(len(transformed_metrics), dtype=bool)
        # 仅在有阈值的维度上进行筛选
        if np.any(filter_mask):
            feasible_mask = np.all(
                transformed_metrics[:, filter_mask] >= np.array(metrics_filter)[filter_mask],
                axis=1
            )
        feasible_solutions = transformed_metrics[feasible_mask]

        if len(feasible_solutions) == 1:
            # 只有1个可行指标组合无法归一化，直接返回该可行指标组合
            return self.test_metrics_list[0], self.cfg_list[0], self.epoch_list[0]
        elif len(feasible_solutions) == 0:
            # 没有可行指标组合返回空值
            return {}, {}, None

        # 归一化
        scaler = MinMaxScaler()
        solutions_norm = scaler.fit_transform(feasible_solutions)

        # Max-Min均衡

        # 每个解的最小指标值
        min_vals = np.min(solutions_norm, axis=1)
        # 最小指标最大的解的子索引
        best_index = np.argmax(min_vals)

        best_metrics = dict(zip(metrics_df.columns.tolist(), metrics_array[feasible_mask][best_index].tolist()))
        best_cfg = dict(zip(cfg_df.columns.tolist(), cfg_array[feasible_mask][best_index].tolist()))
        best_epoch = epoch_array[feasible_mask][best_index]

        return best_metrics, best_cfg, best_epoch

