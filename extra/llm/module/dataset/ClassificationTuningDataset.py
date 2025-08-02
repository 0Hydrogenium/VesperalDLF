import torch
from torch.utils.data import Dataset


class ClassificationTuningDataset(Dataset):
    def __init__(self, data, target_name, tokenizer, max_length):
        self.input_ids = []
        self.target_ids = []
        self.pad_token_id = int(tokenizer.text_to_token_ids("<|endoftext|>").item())

        data_x = data.loc[:, [col for col in data.columns.tolist() if col != target_name]].to_numpy()
        data_y = data.loc[:, target_name].to_numpy()

        for idx in range(len(data_x)):
            token_ids = tokenizer.encode(data_x[idx])
            truncated_token_ids = token_ids[:max_length]
            padded_token_ids = truncated_token_ids + [self.pad_token_id] * (max_length - len(truncated_token_ids))
            self.input_ids.append(torch.tensor(padded_token_ids))
            self.target_ids.append(torch.tensor(data_y[idx]))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, item):
        return self.input_ids[item], self.target_ids[item]

