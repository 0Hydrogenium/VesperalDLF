import pandas as pd
import torch
from torch.utils.data import Dataset


class TimeSeriesDataset(Dataset):
    def __init__(self, data: pd.DataFrame, target_name, timestep, predict_range):
        self.x_list = []
        self.y_list = []

        data_x = data.iloc[:, :].to_numpy()
        data_y = data.loc[:, target_name].to_numpy()
        # 转换为时间窗口
        for idx in range(len(data_x) - timestep - predict_range + 1):
            self.x_list.append(torch.tensor(data_x[idx: idx + timestep], dtype=torch.float32))
            self.y_list.append(torch.tensor(data_y[idx + timestep + predict_range - 1], dtype=torch.float32))

    def __len__(self):
        return len(self.x_list)

    def __getitem__(self, item):
        return self.x_list[item], self.y_list[item]

