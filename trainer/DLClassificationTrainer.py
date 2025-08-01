from tqdm import tqdm
import numpy as np
import torch

from trainer.Trainer import Trainer
from utils.metric_tracker.ClassificationMetricsTracker import ClassificationMetricsTracker


class DLClassificationTrainer(Trainer):
    def __init__(self, cfg, cfg_name):
        super().__init__(cfg, cfg_name)

    def train_model(self, model, loss_function, optimizer, train_loader, epoch):
        model.train()
        train_metrics_tracker = ClassificationMetricsTracker()
        loss_list = []
        train_bar = tqdm(train_loader)
        for idx, train_tensors in enumerate(train_bar):
            x_train, y_train = train_tensors
            x_train, y_train = x_train.to(self.device), y_train.to(self.device)
            optimizer.zero_grad()
            x_train_pred = model(x_train)
            loss = loss_function(x_train_pred, y_train.long())
            loss.backward()
            optimizer.step()

            y_train = y_train.detach().cpu().numpy()
            x_train_pred = x_train_pred.detach().cpu().numpy()

            train_metrics_tracker.update(real_array=y_train, pred_array=np.argmax(x_train_pred, axis=1))
            loss_list.append(loss.item())
            train_bar.desc = "train epoch[{}/{}] (init:{:4f}) loss:{:.4f}".format(epoch, self.cfg["epochs"], loss_list[0], loss)

        train_metrics_tracker.add_new_metric_list("loss", loss_list)
        print(f"[train] metrics: {train_metrics_tracker.get_metrics()}\n")
        return train_metrics_tracker, model

    def test_model(self, model, loss_function, test_loader, epoch):
        model.eval()
        test_metrics_tracker = ClassificationMetricsTracker()
        loss_list = []
        test_bar = tqdm(test_loader)
        with torch.no_grad():
            for idx, test_tensors in enumerate(test_bar):
                x_test, y_test = test_tensors
                x_test, y_test = x_test.to(self.device), y_test.to(self.device)
                x_test_pred = model(x_test)
                loss = loss_function(x_test_pred, y_test.long())

                y_test = y_test.detach().cpu().numpy()
                x_test_pred = x_test_pred.detach().cpu().numpy()

                test_metrics_tracker.update(real_array=y_test, pred_array=np.argmax(x_test_pred, axis=1))
                loss_list.append(loss.item())
                test_bar.desc = "test epoch[{}/{}] (init:{:4f}) loss:{:.4f}".format(epoch, self.cfg["epochs"], loss_list[0], loss)

        test_metrics_tracker.add_new_metric_list("loss", loss_list)
        print(f"[test] metrics: {test_metrics_tracker.get_metrics()}\n")
        return test_metrics_tracker