import numpy as np

from utils.metric_tracker.ClassificationMetricsTracker import ClassificationMetricsTracker
from utils.time_series.TimeSeriesTool import TimeSeriesTool


class CAREMetricsTracker(ClassificationMetricsTracker):
    def __init__(self, beta=0.5, omega_1=1, omega_2=1, omega_3=1, omega_4=2, criticality_threshold=72):
        super().__init__()
        self.metrics = {
            "accuracy": [],
            "precision": [],
            "recall": [],
            "f1": [],
            "CARE_coverage": [],  # Point-wise F(beta)-score
            "CARE_accuracy": [],  # Point-wise Acc
            "CARE_reliability": [],  # Event-wise F(beta)-score or EF(beta)
            "CARE_earliness": [],  # Event-wise the weighted score or WS,
            "care": []
        }
        self.event_pred_list = []  # for computing reliability
        self.event_real_list = []  # for computing reliability

        self.beta = beta
        self.omega_1 = omega_1
        self.omega_2 = omega_2
        self.omega_3 = omega_3
        self.omega_4 = omega_4
        self.criticality_threshold = criticality_threshold

    def get_metrics(self) -> dict:
        avg_func = lambda total, num: round(total / num, self.round_digits) if num != 0 else 0.0

        if len(self.metrics["CARE_reliability"]) == 0:
            tp = np.sum((np.array(self.event_pred_list) == 1) & (np.array(self.event_real_list) == 1)).item()
            fp = np.sum((np.array(self.event_pred_list) == 1) & (np.array(self.event_real_list) == 0)).item()
            fn = np.sum((np.array(self.event_pred_list) == 0) & (np.array(self.event_real_list) == 1)).item()
            care_reliability = ((1 + self.beta ** 2) * tp) / ((1 + self.beta ** 2) * tp + self.beta ** 2 * fn + fp) if tp + fn + fp != 0 else 0.0
            self.metrics["CARE_reliability"].append(care_reliability)
        if len(self.metrics["care"]) == 0:
            tp = np.sum((np.array(self.event_pred_list) == 1) & (np.array(self.event_real_list) == 1)).item()
            avg_care_coverage = round(float(avg_func(sum(self.metrics["CARE_coverage"]), len(self.metrics["CARE_coverage"]))), self.round_digits)
            avg_care_accuracy = round(float(avg_func(sum(self.metrics["CARE_accuracy"]), len(self.metrics["CARE_accuracy"]))), self.round_digits)
            avg_care_reliability = round(float(avg_func(sum(self.metrics["CARE_reliability"]), len(self.metrics["CARE_reliability"]))), self.round_digits)
            avg_care_earliness = round(float(avg_func(sum(self.metrics["CARE_earliness"]), len(self.metrics["CARE_earliness"]))), self.round_digits)
            if tp == 0:
                care = 0
            elif avg_care_accuracy < 0.5:
                care = avg_care_accuracy
            else:
                care = (self.omega_1 * avg_care_coverage + self.omega_2 * avg_care_earliness + self.omega_3 * avg_care_reliability + self.omega_4 * avg_care_accuracy) / (self.omega_1 + self.omega_2 + self.omega_3 + self.omega_4)
            self.metrics["care"].append(care)

        result = {metric_name: round(float(avg_func(sum(self.metrics[metric_name]), len(self.metrics[metric_name]))), self.round_digits) for metric_name in self.metrics.keys()}
        result.update(self.metric)
        return result

    def update_CARE(self, real_array: np.ndarray, pred_array: np.ndarray, status_array: np.ndarray):
        self.update_accuracy(real_array=real_array, pred_array=pred_array) 
        self.update_precision(real_array=real_array, pred_array=pred_array)
        self.update_recall(real_array=real_array, pred_array=pred_array)
        self.update_f1(real_array=real_array, pred_array=pred_array)

        if 1 in real_array:
            # only anomaly event dataset
            filter_mask = (status_array == 0)
            # only normal status_id dataset
            normal_status_id_real_array = real_array[filter_mask]
            normal_status_id_pred_array = pred_array[filter_mask]
            self.update_CARE_coverage(real_array=normal_status_id_real_array, pred_array=normal_status_id_pred_array)
            self.update_CARE_earliness(real_array=real_array, pred_array=pred_array, status_array=status_array)
        else:
            # only normal event dataset
            filter_mask = (status_array == 0)
            # only normal status_id dataset
            normal_status_id_real_array = real_array[filter_mask]
            normal_status_id_pred_array = pred_array[filter_mask]
            self.update_CARE_accuracy(real_array=normal_status_id_real_array, pred_array=normal_status_id_pred_array)
        self.update_CARE_reliability(real_array=real_array, pred_array=pred_array, status_array=status_array)

    def update_CARE_earliness(self, real_array: np.ndarray, pred_array: np.ndarray, status_array: np.ndarray) -> float:
        event_start = TimeSeriesTool.match_first_index(1, real_array)
        if event_start == -1:
            return 0.0
        event_end = TimeSeriesTool.match_last_index(1, real_array)
        if event_end == -1:
            event_end = len(real_array) - 1
        event_end += 1  # the reason why "+1" because include the first unreachable index always weights 0
        matched_status_idx_list = TimeSeriesTool.match_range_indexes(1, status_array, event_start, event_end)
        if len(matched_status_idx_list) > 0:
            event_end = min(event_end, matched_status_idx_list[0])
        ws_score_func = lambda x: 1 if x <= 0.5 else 2 - 2 * x
        dividend = 0
        divisor = 0
        for idx in range(event_start, event_end + 1):
            if (event_end - event_start).item() == 0:  # not included
                return 0.0
            relative = ((idx - event_start) / (event_end - event_start)).item()
            weight = ws_score_func(relative)
            dividend += weight * pred_array[idx].item() if idx < len(pred_array) else 0
            divisor += weight
        care_earliness = dividend / divisor if divisor != 0 else 0
        self.metrics["CARE_earliness"].append(care_earliness)
        return 0.0

    def update_CARE_reliability(self, real_array: np.ndarray, pred_array: np.ndarray, status_array: np.ndarray) -> float:
        max_crit = max(self.compute_CARE_criticality(status_array=status_array, pred_array=pred_array)).item()
        event_pred = 1 if max_crit >= self.criticality_threshold else 0
        event_real = 1 if 1 in real_array else 0
        self.event_pred_list.append(event_pred)
        self.event_real_list.append(event_real)
        return 0.0

    def compute_CARE_criticality(self, status_array: np.ndarray, pred_array: np.ndarray) -> np.ndarray:
        N = len(status_array)
        crit = np.zeros(N + 1)
        for i in range(1, N + 1):
            j = i - 1
            if status_array[j] == 1:  # our definition of the status_id is opposite to that in the paper
                # anomaly status_id
                if pred_array[j] == 1:
                    crit[i] = crit[i - 1] + 1
                else:
                    crit[i] = max(crit[i - 1] - 1, 0)
            else:
                crit[i] = crit[i - 1]
        return crit[1:]

    def update_CARE_accuracy(self, real_array: np.ndarray, pred_array: np.ndarray) -> float:
        tn = np.sum((pred_array == 0) & (real_array == 0)).item()
        fp = np.sum((pred_array == 1) & (real_array == 0)).item()

        care_accuracy = tn / (fp + tn)
        self.metrics["CARE_accuracy"].append(care_accuracy)
        return 0.0

    def update_CARE_coverage(self, real_array: np.ndarray, pred_array: np.ndarray) -> float:
        tp = np.sum((pred_array == 1) & (real_array == 1)).item()
        fp = np.sum((pred_array == 1) & (real_array == 0)).item()
        fn = np.sum((pred_array == 0) & (real_array == 1)).item()
        care_coverage = ((1 + self.beta ** 2) * tp) / ((1 + self.beta ** 2) * tp + self.beta ** 2 * fn + fp) if tp + fn + fp != 0 else 0.0
        self.metrics["CARE_coverage"].append(care_coverage)
        return 0.0
