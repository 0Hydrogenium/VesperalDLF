import torch
from torch.utils.data import Dataset


class SpamDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=None):
        self.pad_token_id = int(tokenizer.text_to_token_ids("<|endoftext|>").item())
        self.data = data
        self.input_token_ids = [tokenizer.encode(text) for text in self.data["text"]]
        if max_length is None:
            self.max_length = self._longest_encoded_length()
        else:
            self.max_length = max_length
            # truncate
            self.input_token_ids = [token_ids[:self.max_length] for token_ids in self.input_token_ids]
        # padding
        self.input_token_ids = [token_ids + [self.pad_token_id] * (self.max_length - len(token_ids)) for token_ids in self.input_token_ids]

    def __getitem__(self, item):
        encoded = self.input_token_ids[item]
        label = self.data.iloc[item]["label"]
        return (torch.tensor(encoded, dtype=torch.long), torch.tensor(label, dtype=torch.long))

    def __len__(self):
        return len(self.data)

    def _longest_encoded_length(self):
        max_length = 0
        for token_ids in self.input_token_ids:
            encoded_length = len(token_ids)
            if encoded_length > max_length:
                max_length = encoded_length
        return max_length

