import json
import os.path
import pickle
from collections import Counter

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import torch
import optuna
from datetime import datetime

from torch.utils.data import DataLoader
from tqdm import tqdm
from collections import Counter

from extra.llm.module.dataset.ClassificationTuningDataset import ClassificationTuningDataset
from extra.llm.module.dataset.PretrainDataset import PretrainDataset
from extra.llm.module.model.GPTModel import GPTModel
from extra.llm.module.tokenizer.BPETokenizer import BPETokenizer
from extra.llm.trainer.PretrainTrainer import PretrainTrainer
from extra.llm.utils.LLMLoader import LLMLoader
from module.LossFunctionFactory import LossFunctionFactory
from module.ModelFactory import ModelFactory
from module.OptimizerFactory import OptimizerFactory
from trainer.Trainer import Trainer
from utils.GeneralTool import GeneralTool
from utils.metric_tracker.BestMetricsTracker import BestMetricsTracker
from utils.metric_tracker.ClassificationMetricsTracker import ClassificationMetricsTracker
from utils.visualizer.Visualizer import Visualizer


"""
    classification finetune
    - drawback: can only predict the categories that model encounter during training
"""
class ClassificationTuningTrainer(PretrainTrainer):
    def __init__(self, cfg: dict, cfg_name: str):
        super().__init__(cfg, cfg_name)

    def run(self, trial: optuna.trial = None, to_visualize=False):
        # 加载动态参数值
        self.load_dynamic_params(trial)

        # 定义分词器
        tokenizer = BPETokenizer()

        # 导入数据表
        df = self.import_data(self.cfg["data_path"])
        # 处理缺失值
        df = self.process_null_value(df)

        # downsampling data to balance
        num_spam = df[df["label"] == "spam"].shape[0]
        ham_subset = df[df["label"] == "ham"].sample(num_spam, random_state=GeneralTool.seed)
        df = pd.concat([ham_subset, df[df["label"] == "spam"]])
        df["label"] = df["label"].map({
            "ham": 0,
            "spam": 1
        })

        # 处理目标值
        df = self.process_target_value(df)
        # 查看目标值分布
        self.show_y_distribution(df[self.cfg["target_name"]])
        # 数据标准化
        df = self.normalize_value(df)
        # 分割训练集和测试集
        train_data, test_data = self.split_into_train_with_test_data(df, self.cfg["train_ratio"])
        # 创建dataset
        train_dataset = ClassificationTuningDataset(train_data, self.cfg["target_name"], tokenizer, self.cfg["context_length"])
        test_dataset = ClassificationTuningDataset(test_data, self.cfg["target_name"], tokenizer, self.cfg["context_length"])
        # 创建dataloader
        train_dataloader = DataLoader(train_dataset, self.cfg["batch_size"], shuffle=True)
        test_dataloader = DataLoader(test_dataset, self.cfg["batch_size"], shuffle=False)

        # 创建模型
        model_name = "GPT2-124M"
        model, cfg = LLMLoader.load(model_name, self.cfg)
        model = model.to(self.device)
        # 创建损失函数
        loss_function = LossFunctionFactory.get(self.cfg, self.cfg["loss_function_name"])
        # 创建优化器
        optimizer = OptimizerFactory.get(self.cfg, self.cfg["optimizer_name"], model)

        """
            add classification head
        """

        # freeze all layer params
        for param in model.parameters():
            param.requires_grad = False

        num_classes = 2
        model.num_head = torch.nn.Linear(in_features=self.cfg["emb_dim"], out_features=num_classes)

        # make the output layer, final LayerNorm, and the last Transformer block trainable
        for param in model.trf_blocks[-1].parameters():
            param.requires_grad = True
        for param in model.final_norm.parameters():
            param.requires_grad = True
        for param in model.num_head.parameters():
            param.requires_grad = True

        # 迭代训练测试模型
        best_metrics_tracker = BestMetricsTracker()  # 创建epochs内最佳指标追踪器
        for epoch in range(1, self.cfg["epochs"] + 1):
            # 训练模型
            train_metrics_tracker, model = self.train_model(model, loss_function, optimizer, train_dataloader, epoch)
            # 测试模型
            test_metrics_tracker = self.test_model(model, loss_function, test_dataloader, epoch)
            # 更新epochs内最佳模型+参数+指标
            best_metrics_tracker.add(train_metrics_tracker, test_metrics_tracker, self.cfg, epoch)

            # 早停策略
            if test_metrics_tracker.get_metrics()["accuracy"] > 0.99:
                print(f"epoch {epoch} early stopping")
                break

        # 获取epochs内最佳模型+参数+指标
        best_metrics, best_cfg, best_epoch = best_metrics_tracker.get_best(self.cfg["metrics_filter"], self.cfg["optimize_metric"], self.cfg["optimize_direction"])
        # 保存数据信息
        self.save_data_info(best_metrics_tracker)

        # 绘制模型训练测试曲线
        if to_visualize:
            self.visualizer.loss_plot_train_test_curve_with_std(
                best_metrics_tracker,
                scale=0.5,
                alpha=0.2,
                figsize=(8, 6),
                save_path=f"{self.result_save_path}/loss_plot.{self.pic}"
            )

        # 更新超参数优化次数
        self.update_trial_idx()

        return self.best_return(best_metrics)

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
            # only focus on the last line token
            # the last token aggregates all the previous tokens info
            loss = loss_function(x_train_pred[:, -1, :], y_train)
            loss.backward()
            optimizer.step()

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
                # only focus on the last line token
                # the last token aggregates all the previous tokens info
                loss = loss_function(x_test_pred[:, -1, :], y_test)

                loss_list.append(loss.item())
                test_bar.desc = "test epoch[{}/{}] (init:{:4f}) loss:{:.4f}".format(epoch, self.cfg["epochs"], loss_list[0], loss)

        test_metrics_tracker.add_new_metric_list("loss", loss_list)
        print(f"[test] metrics: {test_metrics_tracker.get_metrics()}\n")
        return test_metrics_tracker