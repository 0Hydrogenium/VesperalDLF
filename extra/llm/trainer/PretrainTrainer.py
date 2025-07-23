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
from tqdm import tqdm
from collections import Counter

from extra.llm.module.tokenizer.BPETokenizer import BPETokenizer
from module.LossFunctionFactory import LossFunctionFactory
from module.ModelFactory import ModelFactory
from module.OptimizerFactory import OptimizerFactory
from trainer.Trainer import Trainer
from utils.GeneralTool import GeneralTool
from utils.metric_tracker.BestMetricsTracker import BestMetricsTracker
from utils.metric_tracker.ClassificationMetricsTracker import ClassificationMetricsTracker
from utils.visualizer.Visualizer import Visualizer


class PretrainTrainer(Trainer):
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
        train_data, test_data = self.split_into_train_with_test_data(df, self.cfg["train_ratio"])
        # 创建dataset
        train_dataset = TimeSeriesDataset(train_data, self.cfg["target_name"], self.cfg["timestep"], self.cfg["predict_range"])
        test_dataset = TimeSeriesDataset(test_data, self.cfg["target_name"], self.cfg["timestep"], self.cfg["predict_range"])
        # 创建dataloader
        train_dataloader = DataLoader(train_dataset, self.cfg["batch_size"], True)
        test_dataloader = DataLoader(test_dataset, self.cfg["batch_size"], False)

        # 创建模型
        model = ModelFactory.get(self.cfg, self.cfg["model_name"]).to(self.device)
        # 创建损失函数
        loss_function = LossFunctionFactory.get(self.cfg, self.cfg["loss_function_name"])
        # 创建优化器
        optimizer = OptimizerFactory.get(self.cfg, self.cfg["optimizer_name"], model)

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
                save_path=f"{self.result_save_path}/loss_plot.svg"
            )

        # 更新超参数优化次数
        self.update_trial_idx()

        return self.best_return(best_metrics)

    def import_text_data(self, data_path, tokenizer):
        with open(data_path, "r", encoding="utf-8") as f:
            text_data = f.read()
        text_characters_num = len(text_data)
        text_tokens_num = len(tokenizer.encode(text_data))
        print(f"characters num: {text_characters_num}")
        print(f"tokens num: {text_tokens_num}")
        self.data_info["text_characters_num"] = text_characters_num
        self.data_info["text_tokens_num"] = text_tokens_num

        return text_data
