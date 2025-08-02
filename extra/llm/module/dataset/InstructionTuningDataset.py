import torch
from torch.utils.data import Dataset

from extra.llm.module.text_generator.InstructionTuningTextGenerator import InstructionTuningTextGenerator


class InstructionTuningDataset(Dataset):
    def __init__(self, data, tokenizer, max_length, stride, predict_range):
        self.input_ids = []
        self.target_ids = []
        self.pad_token_id = int(tokenizer.text_to_token_ids("<|endoftext|>").item())

        for entry in data:
            instruction_plus_input = InstructionTuningTextGenerator.format_input(entry)
            resp_text = InstructionTuningTextGenerator.format_output(entry)
            full_text = instruction_plus_input + resp_text
            self.input_ids.append(tokenizer.encode(full_text))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, item):
        return self.input_ids[item], self.target_ids[item]

    def collate_fn(self, batch, ignore_index=-100, allowed_max_length=None):
        batch_max_length = max(len(item) + 1 for item in batch)
        inputs_lst, targets_lst = [], []

        for item in batch:
            new_item = item.copy()
            new_item += [self.pad_token_id]
            # Pad sequences to max_length
            padded = new_item + [self.pad_token_id] * (batch_max_length - len(new_item))
            inputs = torch.tensor(padded[:-1])  # Truncate the last token for inputs
            targets = torch.tensor(padded[1:])  # Shift +1 to the right targets

            # 在 targets 中，将除第一个以外的所有填充标记替换为 ignore_index
            mask = targets == self.pad_token_id
            indices = torch.nonzero(mask).squeeze()
            if indices.numel() > 1:
                targets[indices[1:]] = ignore_index

            # 可选择性地将序列截断到最大长度
            if allowed_max_length is not None:
                inputs = inputs[:allowed_max_length]
                targets = targets[:allowed_max_length]

            inputs_lst.append(inputs)
            targets_lst.append(targets)

        inputs_tensor = torch.stack(inputs_lst)
        targets_tensor = torch.stack(targets_lst)
        return inputs_tensor, targets_tensor

    def collate_draft_2(self, batch):
        batch_max_length = max(len(item) + 1 for item in batch)
        inputs_lst, targets_lst = [], []
        for item in batch:
            new_item = item.copy()
            new_item += [self.pad_token_id]
            padded = new_item + [self.pad_token_id] * (batch_max_length - len(new_item))
            inputs = torch.tensor(padded[:-1])  # 截断输入序列的最后一个 token
            targets = torch.tensor(padded[1:])  # 将目标序列中的每个 token 向右移动一个位置
            inputs_lst.append(inputs)
            targets_lst.append(targets)

        inputs_tensor = torch.stack(inputs_lst)
        targets_tensor = torch.stack(targets_lst)
        return inputs_tensor, targets_tensor

    def collate_draft_1(self, batch):
        batch_max_length = max(len(item) + 1 for item in batch)
        inputs_lst = []

        for item in batch:
            new_item = item.copy()
            new_item += [self.pad_token_id]
            padded = new_item + [self.pad_token_id] * (batch_max_length - len(new_item))
            inputs = torch.tensor(padded[-1])  # 删除之前添加的多余填充 token
            inputs_lst.append(inputs)

        inputs_tensor = torch.stack(inputs_lst)
        return inputs_tensor
