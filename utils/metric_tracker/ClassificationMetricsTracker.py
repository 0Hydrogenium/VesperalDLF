import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


class ClassificationMetricsTracker:
    # TODO: 多分类指标计算未实现
    def __init__(self, round_digits=4):
        self.round_digits = round_digits
        self.metrics = {
            "accuracy": [],
            "precision": [],
            "recall": [],
            "f1": [],
        }
        self.metric = {}

    def add_new_metric_list(self, metric_name, metric_list):
        self.metrics[metric_name] = metric_list

    def add_new_metric(self, metric_name, metric):
        self.metric[metric_name] = metric

    def get_metrics(self) -> dict:
        avg_func = lambda total, num: round(total / num, self.round_digits) if num != 0 else 0.0
        result = {metric_name: avg_func(sum(self.metrics[metric_name]), len(self.metrics[metric_name])) for metric_name in self.metrics.keys()}
        result.update(self.metric)
        return result

    def update(self, real_array: np.ndarray, pred_array: np.ndarray):
        self.update_accuracy(real_array=real_array, pred_array=pred_array)
        self.update_precision(real_array=real_array, pred_array=pred_array)
        self.update_recall(real_array=real_array, pred_array=pred_array)
        self.update_f1(real_array=real_array, pred_array=pred_array)

    def update_accuracy(self, real_array: np.ndarray, pred_array: np.ndarray) -> float:
        accuracy = accuracy_score(y_true=real_array, y_pred=pred_array)
        self.metrics["accuracy"].append(accuracy)
        return accuracy

    def update_precision(self, real_array: np.ndarray, pred_array: np.ndarray) -> float:
        precision = precision_score(y_true=real_array, y_pred=pred_array, zero_division=np.nan)
        if not np.isnan(precision):
            self.metrics["precision"].append(precision)
        return precision

    def update_recall(self, real_array: np.ndarray, pred_array: np.ndarray) -> float:
        recall = recall_score(y_true=real_array, y_pred=pred_array, zero_division=np.nan)
        if not np.isnan(recall):
            self.metrics["recall"].append(recall)
        return recall

    def update_f1(self, real_array: np.ndarray, pred_array: np.ndarray) -> float:
        f1 = f1_score(y_true=real_array, y_pred=pred_array, zero_division=np.nan)
        if not np.isnan(f1):
            self.metrics["f1"].append(f1)
        return f1



