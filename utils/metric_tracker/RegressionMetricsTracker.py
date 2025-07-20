import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


class RegressionMetricsTracker:
    def __init__(self, round_digits=8):
        self.round_digits = round_digits
        self.metrics = {
            "mse": [],
            "rmse": [],
            "mae": [],
            "r2": [],
            "adjusted_r2": [],
            "mape": []
        }
        self.metric = {}

    def add_new_metrics(self, metric_name, metric_list):
        self.metrics[metric_name] = metric_list

    def add_new_metric(self, metric_name, metric):
        self.metric[metric_name] = metric

    def get_metrics(self) -> dict:
        avg_func = lambda total, num: round(total / num, self.round_digits) if num != 0 else 0.0
        result = {metric_name: avg_func(sum(self.metrics[metric_name]), len(self.metrics[metric_name])) for metric_name in self.metrics.keys()}
        result.update(self.metric)
        return result

    def update(self, real_array: np.ndarray, pred_array: np.ndarray):
        self.update_mse(real_array=real_array, pred_array=pred_array)
        self.update_rmse(real_array=real_array, pred_array=pred_array)
        self.update_mae(real_array=real_array, pred_array=pred_array)
        self.update_r2(real_array=real_array, pred_array=pred_array)
        self.update_adjusted_r2(real_array=real_array, pred_array=pred_array)
        self.update_mape(real_array=real_array, pred_array=pred_array)

    def update_mse(self, real_array: np.ndarray, pred_array: np.ndarray) -> float:
        mse = mean_squared_error(y_true=real_array, y_pred=pred_array)
        self.metrics["mse"].append(mse)
        return mse

    def update_rmse(self, real_array: np.ndarray, pred_array: np.ndarray) -> float:
        mse = mean_squared_error(y_true=real_array, y_pred=pred_array)
        rmse = np.sqrt(mse)
        self.metrics["rmse"].append(rmse)
        return rmse

    def update_mae(self, real_array: np.ndarray, pred_array: np.ndarray) -> float:
        mae = mean_absolute_error(y_true=real_array, y_pred=pred_array)
        self.metrics["mae"].append(mae)
        return mae

    def update_r2(self, real_array: np.ndarray, pred_array: np.ndarray) -> float:
        r2 = r2_score(y_true=real_array, y_pred=pred_array)
        if not np.isnan(r2):
            self.metrics["r2"].append(r2)
        return r2

    def update_adjusted_r2(self, real_array: np.ndarray, pred_array: np.ndarray, p=None) -> float:
        # p为自变量数量
        r2 = r2_score(y_true=real_array, y_pred=pred_array)
        n = len(real_array)
        p = real_array.shape[1]
        adjusted_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1) if n > p + 1 else np.nan  # 样本量不足时无法计算
        if not np.isnan(adjusted_r2):
            self.metrics["adjusted_r2"].append(adjusted_r2)
        return adjusted_r2

    def update_mape(self, real_array: np.ndarray, pred_array: np.ndarray) -> float:
        mask = real_array != 0  # 创建掩码：仅保留真实值非零的样本
        mape = np.mean(np.abs((real_array[mask] - pred_array[mask]) / real_array[mask])) * 100
        if not np.isnan(mape):
            self.metrics["mape"].append(mape)
        return mape  # 为百分数
