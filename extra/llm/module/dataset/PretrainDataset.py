import torch
from torch.utils.data import Dataset


class PretrainDataset(Dataset):
    def __init__(self, text, tokenizer, max_length, stride, predict_range):
        self.input_ids = []
        self.target_ids = []

        token_ids = tokenizer.encode(text)

        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i: i + max_length]
            target_chunk = token_ids[i + predict_range: i + max_length + predict_range]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, item):
        return self.input_ids[item], self.target_ids[item]

