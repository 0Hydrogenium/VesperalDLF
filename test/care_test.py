import pandas as pd
import numpy as np
from sklearn.metrics import fbeta_score


def calculate_coverage(df: pd.DataFrame, anomaly_start: pd.Timestamp, anomaly_end: pd.Timestamp,
                       beta: float = 0.5) -> float:
    """
    Calculate the Coverage sub-score (Fβ score) for an anomaly event dataset.

    Parameters:
    - df (pd.DataFrame): Dataset with columns 'timestamp', 'train_test', 'status_ID', 'prediction'.
    - anomaly_start (pd.Timestamp): Start timestamp of the anomaly event.
    - anomaly_end (pd.Timestamp): End timestamp of the anomaly event (fault start).
    - beta (float): Beta parameter for Fβ score, default is 0.5.

    Returns:
    - float: Fβ score for coverage, or 0 if no valid data points.
    """
    # Filter to prediction time_series frame and normal status (status_ID 0 or 2)
    pred_df = df[(df['train_test'] == 'test') & (df['status_ID'].isin([0, 2]))].copy()
    if pred_df.empty:
        return 0.0

    # Set true labels: 1 if within anomaly event, 0 otherwise
    pred_df['true_label'] = ((pred_df['timestamp'] >= anomaly_start) &
                             (pred_df['timestamp'] <= anomaly_end)).astype(int)
    y_true = pred_df['true_label']
    y_pred = pred_df['prediction']

    # Compute Fβ score
    try:
        return fbeta_score(y_true, y_pred, beta=beta)
    except ValueError:
        # Handle cases where Fβ is undefined (e.g., no positive predictions or labels)
        return 0.0


def calculate_accuracy(df: pd.DataFrame) -> float:
    """
    Calculate the Accuracy sub-score for a normal behavior dataset.

    Parameters:
    - df (pd.DataFrame): Dataset with columns 'train_test', 'status_ID', 'prediction'.

    Returns:
    - float: Accuracy score (proportion of true negatives), or 1 if no valid data points.
    """
    # Filter to prediction time_series frame and normal status (status_ID 0 or 2)
    pred_df = df[(df['train_test'] == 'test') & (df['status_ID'].isin([0, 2]))]
    if pred_df.empty:
        return 1.0

    y_pred = pred_df['prediction']
    tn = (y_pred == 0).sum()  # True negatives: predicted normal when normal
    fp = (y_pred == 1).sum()  # False positives: predicted anomaly when normal
    total = tn + fp
    return tn / total if total > 0 else 1.0


def calculate_criticality(df: pd.DataFrame) -> int:
    """
    Calculate the maximum criticality for a dataset using Algorithm 1.

    Parameters:
    - df (pd.DataFrame): Dataset with columns 'train_test', 'status_ID', 'prediction'.

    Returns:
    - int: Maximum criticality value.
    """
    # Filter to prediction time_series frame
    pred_df = df[df['train_test'] == 'test']
    if pred_df.empty:
        return 0

    # Status: 1 if normal (status_ID 0 or 2), 0 if abnormal
    s = pred_df['status_ID'].isin([0, 2]).astype(int)
    p = pred_df['prediction']
    N = len(pred_df)
    crit = [0] * (N + 1)

    # Compute criticality per Algorithm 1
    for i in range(1, N + 1):
        if s.iloc[i - 1] == 0:  # Abnormal status
            if p.iloc[i - 1] == 1:
                crit[i] = crit[i - 1] + 1
            else:
                crit[i] = max(crit[i - 1] - 1, 0)
        else:  # Normal status
            crit[i] = crit[i - 1]

    return max(crit[1:])  # Exclude initial 0


def calculate_ws(df: pd.DataFrame, anomaly_start: pd.Timestamp, anomaly_end: pd.Timestamp) -> float:
    """
    Calculate the Weighted Score (WS) for earliness in an anomaly event dataset.

    Parameters:
    - df (pd.DataFrame): Dataset with columns 'timestamp', 'prediction'.
    - anomaly_start (pd.Timestamp): Start timestamp of the anomaly event.
    - anomaly_end (pd.Timestamp): End timestamp of the anomaly event (fault start).

    Returns:
    - float: Weighted score for earliness, or 0 if no valid data points.
    """
    # Filter to anomaly event period
    event_df = df[(df['timestamp'] >= anomaly_start) & (df['timestamp'] <= anomaly_end)]
    if event_df.empty:
        return 0.0

    # Calculate relative position x and weights w(t)
    duration = (anomaly_end - anomaly_start).total_seconds()
    x = (event_df['timestamp'] - anomaly_start).dt.total_seconds() / duration
    w = np.where(x <= 0.5, 1.0, 2 * (1 - x))  # Piecewise linear weight function
    p = event_df['prediction']

    # Compute WS
    weighted_sum = (w * p).sum()
    weight_sum = w.sum()
    return weighted_sum / weight_sum if weight_sum > 0 else 0.0


def compute_care_score(datasets: list[pd.DataFrame], anomaly_info: dict, predictions: dict) -> float:
    """
    Compute the overall CARE score for a set of datasets.

    Parameters:
    - datasets (list[pd.DataFrame]): List of DataFrames, each with required columns.
    - anomaly_info (dict): Dict mapping dataset index to (is_anomaly, anomaly_start, anomaly_end).
                           If not anomaly, anomaly_start and anomaly_end can be None.
    - predictions (dict): Dict mapping dataset index to prediction Series.

    Returns:
    - float: Final CARE score.
    """
    f_beta_list = []
    ws_list = []
    acc_list = []
    event_true = []
    event_pred = []

    # Process each dataset
    for idx, df in enumerate(datasets):
        df['prediction'] = predictions[idx]
        is_anomaly, anomaly_start, anomaly_end = anomaly_info[idx]

        if is_anomaly:
            f_beta = calculate_coverage(df, anomaly_start, anomaly_end)
            ws = calculate_ws(df, anomaly_start, anomaly_end)
            f_beta_list.append(f_beta)
            ws_list.append(ws)
        else:
            acc = calculate_accuracy(df)
            acc_list.append(acc)

        max_crit = calculate_criticality(df)
        event_pred.append(1 if max_crit >= 72 else 0)
        event_true.append(1 if is_anomaly else 0)

    # Compute averages and event-based Fβ
    f_beta_avg = np.mean(f_beta_list) if f_beta_list else 0.0
    ws_avg = np.mean(ws_list) if ws_list else 0.0
    acc_avg = np.mean(acc_list) if acc_list else 1.0
    ef_beta = fbeta_score(event_true, event_pred, beta=0.5) if event_true else 0.0

    # Check special cases
    any_anomalies_detected = any(np.any(pred == 1) for pred in predictions.values())
    if not any_anomalies_detected:
        return 0.0
    elif acc_avg < 0.5:
        return acc_avg
    else:
        # Weighted average with ω1=ω2=ω3=1, ω4=2
        wa = (f_beta_avg + ws_avg + ef_beta + 2 * acc_avg) / 5
        return wa


if __name__ == '__main__':
    import pandas as pd

    df = pd.DataFrame({
        'timestamp': pd.date_range(start='2023-01-01', periods=100, freq='10min'),
        'train_test': ['train'] * 50 + ['test'] * 50,
        'status_ID': [0] * 60 + [4] * 40,
        'prediction': [0] * 70 + [1] * 30
    })
    anomaly_start = pd.Timestamp('2023-01-01 08:00:00')
    anomaly_end = pd.Timestamp('2023-01-01 10:00:00')
    coverage = calculate_coverage(df, anomaly_start, anomaly_end)
    accuracy = calculate_accuracy(df)
    criticality = calculate_criticality(df)
    ws = calculate_ws(df, anomaly_start, anomaly_end)
    datasets = [df]
    anomaly_info = {0: (True, anomaly_start, anomaly_end)}
    predictions = {0: df['prediction']}
    care = compute_care_score(datasets, anomaly_info, predictions)
    print(care)