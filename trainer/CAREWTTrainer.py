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
from torch.utils.data.dataloader import DataLoader

from module.LossFunctionFactory import LossFunctionFactory
from module.ModelFactory import ModelFactory
from module.OptimizerFactory import OptimizerFactory
from module.dataset.CAREWTDataset import CAREWTDataset
from module.dataset.TimeSeriesDataset import TimeSeriesDataset
from utils.GeneralTool import GeneralTool
from utils.metric_tracker.BestMetricsTracker import BestMetricsTracker
from utils.metric_tracker.CAREMetricsTracker import CAREMetricsTracker
from utils.metric_tracker.ClassificationMetricsTracker import ClassificationMetricsTracker
from utils.metric_tracker.RegressionMetricsTracker import RegressionMetricsTracker
from utils.visualizer.CAREWTVisualizer import CAREWTVisualizer
from utils.visualizer.Visualizer import Visualizer
from trainer.Trainer import Trainer


class CAREWTTrainer(Trainer):
    def __init__(self, cfg, cfg_name):
        super().__init__(cfg, cfg_name)
        self.visualizer = CAREWTVisualizer()

    def run(self, trial: optuna.trial = None, to_visualize=False):

        non_input_cols = [
            'time_stamp',
            'asset_id',
            'id',
            'train_test',
            'status_type_id',
            'label'
        ]

        # 加载动态参数值
        self.load_dynamic_params(trial)
        # 导入数据表
        farm_datasets, farm_event_info_dataset = self.import_farm_data(self.cfg["data_path"], farm_name=self.cfg["farm_name"])
        # 处理缺失值
        farm_datasets = self.process_null_value(farm_datasets)
        # 数据标准化
        self.farm_normalize_value(farm_datasets, non_input_cols)
        # 分割训练集和测试集
        train_farm_datasets, test_farm_datasets = self.farm_split_into_train_with_test_data(farm_datasets)
        if to_visualize:
            # self.visualizer.wind_turbines_time_curve(train_farm_datasets, alpha=0.3, s=4, figsize=(12, 8), save_path=f"{self.result_save_path}/train_wind_turbines_time_curve_@.{self.pic}")
            # self.visualizer.wind_turbines_time_curve(test_farm_datasets, alpha=0.7, s=4, figsize=(12, 8), save_path=f"{self.result_save_path}/test_wind_turbines_time_curve_@.{self.pic}")
            pass

        if self.cfg["model_name"] == "Random":
            return self.random_trivial_run(train_farm_datasets, test_farm_datasets, non_input_cols)
        elif self.cfg["model_name"] == "AllNormal":
            return self.all_normal_trivial_run(train_farm_datasets, test_farm_datasets, non_input_cols)
        elif self.cfg["model_name"] == "AllAnomaly":
            return self.all_anomaly_trivial_run(train_farm_datasets, test_farm_datasets, non_input_cols)
        elif self.cfg["model_name"] == "IsolationForestClassification":
            return self.isolation_forest_run(train_farm_datasets, test_farm_datasets, non_input_cols)
        else:

            # 创建dataset
            # 创建dataloader
            train_dataloaders = {}
            test_dataloaders = {}
            for dataset_name, dataset in train_farm_datasets.items():
                train_dataloaders[dataset_name] = DataLoader(CAREWTDataset(dataset, "label"), self.cfg["batch_size"], shuffle=False)
            for dataset_name, dataset in test_farm_datasets.items():
                test_dataloaders[dataset_name] = DataLoader(CAREWTDataset(dataset, "label"), self.cfg["batch_size"], shuffle=False)

            if self.cfg["model_name"] == "AutoEncoder":
                return self.autoencoder_run(train_dataloaders, test_dataloaders, non_input_cols, to_visualize)

    def random_trivial_run(self, train_farm_datasets, test_farm_datasets, non_input_cols):
        predictor = lambda X: np.random.randint(0, 2, size=len(X))

        # 迭代训练测试模型
        best_metrics_tracker = BestMetricsTracker()  # 创建epochs内最佳指标追踪器
        for epoch in range(1, 2):
            # 测试模型
            test_metrics_tracker = self.trivial_test_model(predictor, test_farm_datasets, epoch, non_input_cols)
            # 更新epochs内最佳模型+参数+指标
            best_metrics_tracker.add(ClassificationMetricsTracker(), test_metrics_tracker, self.cfg, epoch)

        # 获取epochs内最佳模型+参数+指标
        best_metrics, best_cfg, best_epoch = best_metrics_tracker.get_best(self.cfg["metrics_filter"], self.cfg["optimize_metric"], self.cfg["optimize_direction"])
        # 保存数据信息
        self.save_data_info(best_metrics_tracker)

        # 更新超参数优化次数
        self.update_trial_idx()

        return self.best_return(best_metrics)

    def all_normal_trivial_run(self, train_farm_datasets, test_farm_datasets, non_input_cols):
        predictor = lambda X: np.zeros(len(X))

        # 迭代训练测试模型
        best_metrics_tracker = BestMetricsTracker()  # 创建epochs内最佳指标追踪器
        for epoch in range(1, 2):
            # 测试模型
            test_metrics_tracker = self.trivial_test_model(predictor, test_farm_datasets, epoch, non_input_cols)
            # 更新epochs内最佳模型+参数+指标
            best_metrics_tracker.add(ClassificationMetricsTracker(), test_metrics_tracker, self.cfg, epoch)

        # 获取epochs内最佳模型+参数+指标
        best_metrics, best_cfg, best_epoch = best_metrics_tracker.get_best(self.cfg["metrics_filter"], self.cfg["optimize_metric"], self.cfg["optimize_direction"])
        # 保存数据信息
        self.save_data_info(best_metrics_tracker)

        # 更新超参数优化次数
        self.update_trial_idx()

        return self.best_return(best_metrics)

    def all_anomaly_trivial_run(self, train_farm_datasets, test_farm_datasets, non_input_cols):
        predictor = lambda X: np.ones(len(X))

        # 迭代训练测试模型
        best_metrics_tracker = BestMetricsTracker()  # 创建epochs内最佳指标追踪器
        for epoch in range(1, 2):
            # 测试模型
            test_metrics_tracker = self.trivial_test_model(predictor, test_farm_datasets, epoch, non_input_cols)
            # 更新epochs内最佳模型+参数+指标
            best_metrics_tracker.add(ClassificationMetricsTracker(), test_metrics_tracker, self.cfg, epoch)

        # 获取epochs内最佳模型+参数+指标
        best_metrics, best_cfg, best_epoch = best_metrics_tracker.get_best(self.cfg["metrics_filter"], self.cfg["optimize_metric"], self.cfg["optimize_direction"])
        # 保存数据信息
        self.save_data_info(best_metrics_tracker)

        # 更新超参数优化次数
        self.update_trial_idx()

        return self.best_return(best_metrics)

    def isolation_forest_run(self, train_farm_datasets, test_farm_datasets, non_input_cols):
        # 创建模型
        model = ModelFactory.get(self.cfg, self.cfg["model_name"])

        # 迭代训练测试模型
        best_metrics_tracker = BestMetricsTracker()  # 创建epochs内最佳指标追踪器
        for epoch in range(1, 2):
            # 训练模型
            train_metrics_tracker, model = self.isolation_forest_train_model(model, train_farm_datasets, epoch, non_input_cols)
            # 测试模型
            test_metrics_tracker = self.isolation_forest_test_model(model, test_farm_datasets, epoch, non_input_cols)
            # 更新epochs内最佳模型+参数+指标
            best_metrics_tracker.add(train_metrics_tracker, test_metrics_tracker, self.cfg, epoch)

        # 获取epochs内最佳模型+参数+指标
        best_metrics, best_cfg, best_epoch = best_metrics_tracker.get_best(self.cfg["metrics_filter"], self.cfg["optimize_metric"], self.cfg["optimize_direction"])
        # 保存数据信息
        self.save_data_info(best_metrics_tracker)

        # 更新超参数优化次数
        self.update_trial_idx()

        return self.best_return(best_metrics)

    def autoencoder_run(self, train_dataloaders, test_dataloaders, non_input_cols, to_visualize):
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
            train_metrics_tracker, model = self.ae_train_model(model, loss_function, optimizer, train_dataloaders, epoch)
            # 测试模型
            test_metrics_tracker = self.ae_test_model(model, loss_function, test_dataloaders, epoch)
            # 更新epochs内最佳模型+参数+指标
            best_metrics_tracker.add(train_metrics_tracker, test_metrics_tracker, self.cfg, epoch)

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

    def trivial_test_model(self, predictor, test_farm_datasets, epoch, non_input_cols):
        test_metrics_tracker = CAREMetricsTracker()
        for dataset_name, dataset in test_farm_datasets.items():
            x_test = np.array(dataset.loc[:, [col for col in dataset.columns if col not in non_input_cols]])
            y_test = np.array(dataset.loc[:, "label"])
            x_test_pred = predictor(x_test)
            status_array = np.array(dataset.loc[:, "status_type_id"])
            test_metrics_tracker.update_CARE(real_array=y_test, pred_array=x_test_pred, status_array=status_array)

        print(f"[test] metrics: {test_metrics_tracker.get_metrics()}\n")
        return test_metrics_tracker

    def isolation_forest_train_model(self, model, train_farm_datasets, epoch, non_input_cols):
        train_metrics_tracker = CAREMetricsTracker()
        for dataset_name, dataset in train_farm_datasets.items():
            x_train = np.array(dataset.loc[:, [col for col in dataset.columns if col not in non_input_cols]])
            y_train = np.array(dataset.loc[:, "label"])
            x_train_pred = model.train(x_train)
            status_array = np.array(dataset.loc[:, "status_type_id"])
            train_metrics_tracker.update_CARE(real_array=y_train, pred_array=x_train_pred, status_array=status_array)

        print(f"[train] metrics: {train_metrics_tracker.get_metrics()}\n")
        return train_metrics_tracker, model

    def isolation_forest_test_model(self, model, test_farm_datasets, epoch, non_input_cols):
        test_metrics_tracker = CAREMetricsTracker()
        for dataset_name, dataset in test_farm_datasets.items():
            x_test = np.array(dataset.loc[:, [col for col in dataset.columns if col not in non_input_cols]])
            y_test = np.array(dataset.loc[:, "label"])
            x_test_pred = model.test(x_test)
            status_array = np.array(dataset.loc[:, "status_type_id"])
            test_metrics_tracker.update_CARE(real_array=y_test, pred_array=x_test_pred, status_array=status_array)

        print(f"[test] metrics: {test_metrics_tracker.get_metrics()}\n")
        return test_metrics_tracker

    def ae_train_model(self, model, loss_function, optimizer, train_loaders, epoch):
        model.train()
        train_metrics_tracker = RegressionMetricsTracker()
        loss_list = []
        train_bar = tqdm(train_loader)
        for idx, train_tensors in enumerate(train_bar):
            x_train, y_train = train_tensors
            x_train, y_train = x_train.to(self.device), y_train.to(self.device)
            optimizer.zero_grad()
            x_train_pred = model(x_train)
            loss = loss_function(x_train_pred, x_train)
            loss.backward()
            optimizer.step()

            train_metrics_tracker.update(real_array=x_train.reshape(x_train.shape[0], -1), pred_array=x_train_pred.reshape(x_train_pred.shape[0], -1))
            loss_list.append(loss.item())
            train_bar.desc = "train epoch[{}/{}] (init:{:4f}) loss:{:.4f}".format(epoch, self.cfg["epochs"], loss_list[0], loss)

        train_metrics_tracker.add_new_metric_list("loss", loss_list)
        print(f"[train] metrics: {train_metrics_tracker.get_metrics()}\n")
        return train_metrics_tracker, model

    def ae_test_model(self, model, loss_function, test_loader, epoch):
        model.eval()
        test_metrics_tracker = CAREMetricsTracker()
        loss_list = []
        test_bar = tqdm(test_loader)
        with torch.no_grad():
            for idx, test_tensors in enumerate(test_bar):
                x_test, y_test = test_tensors
                x_test, y_test = x_test.to(self.device), y_test.to(self.device)
                x_test_pred = model(x_test)
                loss = loss_function(x_test_pred, x_test)

                status_array = np.array(dataset.loc[:, "status_type_id"])

                test_metrics_tracker.update_CARE(real_array=y_test, pred_array=x_test_pred, )
                loss_list.append(loss.item())
                test_bar.desc = "test epoch[{}/{}] (init:{:4f}) loss:{:.4f}".format(epoch, self.cfg["epochs"], loss_list[0], loss)

        test_metrics_tracker.add_new_metric_list("loss", loss_list)
        print(f"[test] metrics: {test_metrics_tracker.get_metrics()}\n")
        return test_metrics_tracker

    def farm_split_into_train_with_test_data(self, farm_datasets):
        train_farm_datasets = {}
        test_farm_datasets = {}
        for dataset_name, dataset in farm_datasets.items():
            train_mask = (dataset["train_test"].values == "train")
            train_farm_datasets[dataset_name] = dataset.iloc[train_mask, :].reset_index(drop=True)
            test_farm_datasets[dataset_name] = dataset.iloc[~train_mask, :].reset_index(drop=True)
        return train_farm_datasets, test_farm_datasets

    def farm_normalize_value(self, farm_datasets, non_input_cols):
        for dataset_name, dataset in farm_datasets.items():
            farm_datasets[dataset_name] = pd.concat([
                dataset.loc[:, non_input_cols],
                self.normalize_value(dataset.loc[:, [col for col in dataset.columns if col not in non_input_cols]])
            ], axis=1)

    def import_farm_data(self, data_path, farm_name, encoding='utf-8'):
        data_path = data_path.replace('@', GeneralTool.root_path)
        farm_path = os.path.join(data_path, farm_name)

        event_info_path = os.path.join(farm_path, "comma_event_info.csv")
        farm_event_info_dataset = pd.read_csv(event_info_path, encoding=encoding) if os.path.exists(event_info_path) else None
        # mapping "status_type_id" target to binary
        # 0: normal, 1: anomaly
        farm_event_info_dataset["event_label"] = farm_event_info_dataset["event_label"].map({
            "normal": 0,
            "anomaly": 1
        })

        farm_datasets = {}
        datasets_dir = os.path.join(farm_path, "datasets")
        for file_name in tqdm(os.listdir(datasets_dir), desc=f"loading farm {farm_name} datasets"):
            if file_name.endswith(".csv"):
                dataset_id = int(file_name.replace(".csv", "").replace("comma_", ""))
                file_path = os.path.join(datasets_dir, file_name)
                df = pd.read_csv(file_path, encoding=encoding)
                # mapping "status_type_id" target to binary
                # 0: normal, 1: anomaly
                df["status_type_id"] = df["status_type_id"].map({
                    0: 0,
                    1: 1,
                    2: 0,
                    3: 1,
                    4: 1,
                    5: 1
                })

                df["label"] = 0
                matched_data = farm_event_info_dataset[farm_event_info_dataset["event_id"] == dataset_id]
                if matched_data["event_label"].tolist()[0] == 1:
                    df.loc[(df["time_stamp"] >= matched_data["event_start"].to_list()[0]) & (df["time_stamp"] <= matched_data["event_end"].to_list()[0]), "label"] = 1

                farm_datasets[dataset_id] = df

        return farm_datasets, farm_event_info_dataset

    def process_null_value(self, farm_datasets):
        new_farm_datasets = {}
        for dataset_name, df in farm_datasets.items():
            # nan_idx = np.argwhere(df.isna().values)
            if self.cfg["null_data_method"] == "del_null":
                df = df.dropna()
            elif self.cfg["null_data_method"] == "fill_null_with_zero":
                df = df.fillna(0)
            elif self.cfg["null_data_method"] == "fill_null_with_mean":
                for col in df.columns:
                    if df[col].isnull().any():
                        df[col] = df[col].fillna(df[col].mean())
            print(f"processed null data (length: {df.shape[0]}, features: {df.shape[1]})")
            self.data_info["processed_null_data_length"] = df.shape[0]
            self.data_info["processed_null_data_features"] = df.shape[1]
            new_farm_datasets[dataset_name] = df

        return new_farm_datasets
