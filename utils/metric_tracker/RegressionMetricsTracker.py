import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


class RegressionMetricsTracker:
    def __init__(self, round_digits=8):
        self.round_digits = round_digits

        self.total_mse = 0
        self.total_rmse = 0
        self.total_mae = 0
        self.total_r2 = 0
        self.total_adjusted_r2 = 0
        self.total_mape = 0

        self.mse_num = 0
        self.rmse_num = 0
        self.mae_num = 0
        self.r2_num = 0
        self.adjusted_r2_num = 0
        self.mape_num = 0

    def get_metrics(self) -> dict:
        avg_func = lambda total, num: round(total / num, self.round_digits) if num != 0 else 0.0

        avg_mse = avg_func(self.total_mse, self.mse_num)
        avg_rmse = avg_func(self.total_rmse, self.rmse_num)
        avg_mae = avg_func(self.total_mae, self.mae_num)
        avg_r2 = avg_func(self.total_r2, self.r2_num)
        avg_adjusted_r2 = avg_func(self.total_adjusted_r2, self.adjusted_r2_num)
        avg_mape = avg_func(self.total_mape, self.mape_num)

        return {
            "mse": avg_mse,
            "rmse": avg_rmse,
            "mae": avg_mae,
            "r2": avg_r2,
            "adjusted_r2": avg_adjusted_r2,
            "mape": avg_mape,
        }

    def update(self, real_array: np.ndarray, pred_array: np.ndarray, p):
        # p为自变量数量
        self.update_mse(real_array=real_array, pred_array=pred_array)
        self.update_rmse(real_array=real_array, pred_array=pred_array)
        self.update_mae(real_array=real_array, pred_array=pred_array)
        self.update_r2(real_array=real_array, pred_array=pred_array)
        self.update_adjusted_r2(real_array=real_array, pred_array=pred_array, p=p)
        self.update_mape(real_array=real_array, pred_array=pred_array)

    def update_mse(self, real_array: np.ndarray, pred_array: np.ndarray) -> float:
        mse = mean_squared_error(y_true=real_array, y_pred=pred_array)
        self.total_mse += mse
        self.mse_num += 1
        return mse

    def update_rmse(self, real_array: np.ndarray, pred_array: np.ndarray) -> float:
        mse = mean_squared_error(y_true=real_array, y_pred=pred_array)
        rmse = np.sqrt(mse)
        self.total_rmse += rmse
        self.rmse_num += 1
        return rmse

    def update_mae(self, real_array: np.ndarray, pred_array: np.ndarray) -> float:
        mae = mean_absolute_error(y_true=real_array, y_pred=pred_array)
        self.total_mae += mae
        self.mae_num += 1
        return mae

    def update_r2(self, real_array: np.ndarray, pred_array: np.ndarray) -> float:
        r2 = r2_score(y_true=real_array, y_pred=pred_array)
        if not np.isnan(r2):
            self.total_r2 += r2
            self.r2_num += 1
        return r2

    def update_adjusted_r2(self, real_array: np.ndarray, pred_array: np.ndarray, p) -> float:
        r2 = r2_score(y_true=real_array, y_pred=pred_array)
        n = len(real_array)
        adjusted_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1) if n > p + 1 else np.nan  # 样本量不足时无法计算
        if not np.isnan(adjusted_r2):
            self.total_adjusted_r2 += adjusted_r2
            self.adjusted_r2_num += 1
        return adjusted_r2

    def update_mape(self, real_array: np.ndarray, pred_array: np.ndarray) -> float:
        mask = real_array != 0  # 创建掩码：仅保留真实值非零的样本
        mape = np.mean(np.abs((real_array[mask] - pred_array[mask]) / real_array[mask])) * 100
        if not np.isnan(mape):
            self.total_mape += mape
            self.mape_num += 1
        return mape  # 为百分数
