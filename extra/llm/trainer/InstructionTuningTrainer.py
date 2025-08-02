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

from extra.llm.module.dataset.PretrainDataset import PretrainDataset
from extra.llm.module.model.GPTModel import GPTModel
from extra.llm.module.tokenizer.BPETokenizer import BPETokenizer
from module.LossFunctionFactory import LossFunctionFactory
from module.ModelFactory import ModelFactory
from module.OptimizerFactory import OptimizerFactory
from trainer.Trainer import Trainer
from utils.GeneralTool import GeneralTool
from utils.metric_tracker.BestMetricsTracker import BestMetricsTracker
from utils.metric_tracker.ClassificationMetricsTracker import ClassificationMetricsTracker
from utils.visualizer.Visualizer import Visualizer


class InstructionTuningTrainer(Trainer):
    def __init__(self, cfg: dict, cfg_name: str):
        super().__init__(cfg, cfg_name)

    def run(self, trial: optuna.trial = None, to_visualize=False):
        # 加载动态参数值
        self.load_dynamic_params(trial)

        # 定义分词器
        tokenizer = BPETokenizer()

        # 导入数据表
        text_data = self.import_text_data(self.cfg["data_path"], tokenizer)
        # 分割训练集和测试集
        train_data, test_data = self.split_text_into_train_with_test_data(text_data, self.cfg["train_ratio"], tokenizer)
        # 创建dataset
        train_dataset = PretrainDataset(train_data, tokenizer, max_length=self.cfg["context_length"], stride=self.cfg["context_length"], predict_range=self.cfg["predict_range"])
        test_dataset = PretrainDataset(test_data, tokenizer, max_length=self.cfg["context_length"], stride=self.cfg["context_length"], predict_range=self.cfg["predict_range"])
        # 创建dataloader
        train_dataloader = DataLoader(train_dataset, self.cfg["batch_size"], shuffle=True, drop_last=True)
        test_dataloader = DataLoader(test_dataset, self.cfg["batch_size"], shuffle=False, drop_last=False)

        # 创建模型
        model = GPTModel(self.cfg).to(self.device)
        # 创建损失函数
        loss_function = LossFunctionFactory.get(self.cfg, self.cfg["loss_function_name"])
        # 创建优化器
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.cfg["learning_rate"], weight_decay=self.cfg["weight_decay"])

        # 迭代训练测试模型
        best_metrics_tracker = BestMetricsTracker()  # 创建epochs内最佳指标追踪器
        for epoch in range(1, self.cfg["epochs"] + 1):
            # 训练模型
            train_metrics_tracker, model = self.train_model(model, loss_function, optimizer, train_dataloader, epoch)
            # 测试模型
            test_metrics_tracker = self.test_model(model, loss_function, test_dataloader, epoch)
            # 更新epochs内最佳模型+参数+指标
            best_metrics_tracker.add(train_metrics_tracker, test_metrics_tracker, self.cfg, epoch)
            # 保存当前epoch的模型
            self.save_model(model, f"{self.model_save_path}/model_{epoch}.pth")

        # 获取epochs内最佳模型+参数+指标
        best_metrics, best_cfg, best_epoch = best_metrics_tracker.get_best(self.cfg["metrics_filter"], self.cfg["optimize_metric"], self.cfg["optimize_direction"])
        # 保存数据信息
        self.save_data_info(best_metrics_tracker)

        # 只保留最佳模型存储
        self.filter_save_model(f"{self.model_save_path}/model_{best_epoch}.pth")

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

    def filter_save_model(self, best_epoch_path):
        for sub_path in os.listdir(self.model_save_path):
            current_path = f"{self.model_save_path}/{sub_path}"
            if current_path != best_epoch_path:
                try:
                    os.remove(current_path)
                except Exception as e:
                    print(e)
                    continue

    def save_model(self, model, save_path):
        torch.save(model.state_dict(), save_path)

    def split_text_into_train_with_test_data(self, data, train_ratio, tokenizer):
        train_size = int(train_ratio * len(data))
        train_data = data[:train_size]
        test_data = data[train_size:]

        print(f"train_data shape: {len(train_data)}, test_data shape: {len(test_data)}")
        self.data_info["train_data_shape"], self.data_info["test_data_shape"] = len(train_data), len(test_data)

        train_data_tokens_num = len(tokenizer.encode(train_data))
        test_data_tokens_num = len(tokenizer.encode(test_data))
        print(f"train_data_tokens shape: {train_data_tokens_num}, test_data_tokens shape: {test_data_tokens_num}")
        self.data_info["train_data_tokens_shape"], self.data_info["test_data_tokens_shape"] = train_data_tokens_num, test_data_tokens_num

        return train_data, test_data

    def import_text_data(self, data_path, tokenizer):
        with open(data_path.replace("@", GeneralTool.root_path), "r", encoding="utf-8") as f:
            text_data = f.read()
        text_characters_num = len(text_data)
        text_tokens_num = len(tokenizer.encode(text_data))
        print(f"characters num: {text_characters_num}")
        print(f"tokens num: {text_tokens_num}")
        self.data_info["text_characters_num"] = text_characters_num
        self.data_info["text_tokens_num"] = text_tokens_num

        return text_data
