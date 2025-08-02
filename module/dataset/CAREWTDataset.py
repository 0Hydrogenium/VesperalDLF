import pandas as pd
import torch
from torch.utils.data import Dataset


class CAREWTDataset(Dataset):
    def __init__(self, data: pd.DataFrame, target_name):
        self.x_list = []
        self.y_list = []

        data_x = data.iloc[:, :].to_numpy()
        data_y = data.loc[:, target_name].to_numpy()
        for idx in range(len(data_x)):
            self.x_list.append(torch.tensor(data_x[idx], dtype=torch.float32))
            self.y_list.append(torch.tensor(data_y[idx], dtype=torch.float32))

    def __len__(self):
        return len(self.x_list)

    def __getitem__(self, item):
        return self.x_list[item], self.y_list[item]

