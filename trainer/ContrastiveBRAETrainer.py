import torch

from trainer.Trainer import Trainer


class ContrastiveBRAETrainer(Trainer):
    def __init__(self, cfg, cfg_name):
        super().__init__(cfg, cfg_name)
        self.positive_dataloader = None

    def get_data_loader(self, x_train_tensor, y_train_tensor, x_test_tensor, y_test_tensor):
        train_dataset = torch.utils.data.TensorDataset(x_train_tensor, y_train_tensor)
        test_dataset = torch.utils.data.TensorDataset(x_test_tensor, y_test_tensor)

        train_loader = torch.utils.data.DataLoader(train_dataset, self.cfg["batch_size"], True)
        test_loader = torch.utils.data.DataLoader(test_dataset, self.cfg["batch_size"], False)

        return train_loader, test_loader

