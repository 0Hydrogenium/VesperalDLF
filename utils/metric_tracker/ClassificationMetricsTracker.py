import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


class ClassificationMetricsTracker:
    # TODO: 多分类指标计算未实现
    def __init__(self, round_digits=4):
        self.round_digits = round_digits

        self.total_accuracy = 0
        self.total_precision = 0
        self.total_recall = 0
        self.total_f1 = 0

        self.accuracy_num = 0
        self.precision_num = 0
        self.recall_num = 0
        self.f1_num = 0

    def get_metrics(self) -> dict:
        avg_func = lambda total, num: round(total / num, self.round_digits) if num != 0 else 0.0
        avg_accuracy = avg_func(self.total_accuracy, self.accuracy_num)
        avg_precision = avg_func(self.total_precision, self.precision_num)
        avg_recall = avg_func(self.total_recall, self.recall_num)
        avg_f1 = avg_func(self.total_f1, self.f1_num)

        return {
            "accuracy": avg_accuracy,
            "precision": avg_precision,
            "recall": avg_recall,
            "f1": avg_f1,
        }

    def update(self, real_array: np.ndarray, pred_array: np.ndarray):
        self.update_accuracy(real_array=real_array, pred_array=pred_array)
        self.update_precision(real_array=real_array, pred_array=pred_array)
        self.update_recall(real_array=real_array, pred_array=pred_array)
        self.update_f1(real_array=real_array, pred_array=pred_array)

    def update_accuracy(self, real_array: np.ndarray, pred_array: np.ndarray) -> float:
        accuracy = accuracy_score(y_true=real_array, y_pred=pred_array)
        self.total_accuracy += accuracy
        self.accuracy_num += 1
        return accuracy

    def update_precision(self, real_array: np.ndarray, pred_array: np.ndarray) -> float:
        precision = precision_score(y_true=real_array, y_pred=pred_array, zero_division=np.nan)
        if not np.isnan(precision):
            self.total_precision += precision
            self.precision_num += 1
        return precision

    def update_recall(self, real_array: np.ndarray, pred_array: np.ndarray) -> float:
        recall = recall_score(y_true=real_array, y_pred=pred_array, zero_division=np.nan)
        if not np.isnan(recall):
            self.total_recall += recall
            self.recall_num += 1
        return recall

    def update_f1(self, real_array: np.ndarray, pred_array: np.ndarray) -> float:
        f1 = f1_score(y_true=real_array, y_pred=pred_array, zero_division=np.nan)
        if not np.isnan(f1):
            self.total_f1 += f1
            self.f1_num += 1
        return f1



