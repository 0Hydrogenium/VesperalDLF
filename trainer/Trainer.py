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
from module.dataset.TimeSeriesDataset import TimeSeriesDataset
from utils.GeneralTool import GeneralTool
from utils.metric_tracker.BestMetricsTracker import BestMetricsTracker
from utils.metric_tracker.ClassificationMetricsTracker import ClassificationMetricsTracker
from utils.visualizer.Visualizer import Visualizer


class Trainer:

    WARNING_NOT_FOUND_STR = "[config] '{}' is not found"
    SAVE_PATH = GeneralTool.root_path + "/result/{}"
    MODEL_PATH = GeneralTool.root_path + "/model/{}"

    def __init__(self, cfg: dict, cfg_name: str, chosen_metric: str = "recall"):
        self.cfg = cfg  # 当前参数配置
        self.dynamic_params_cfg = {}  # 备份动态参数配置
        self.cfg_name = cfg_name
        self.result_save_path = ""
        self.model_save_path = ""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"device: {self.device}")
        self.trial_idx = 1
        self.visualizer = Visualizer()
        self.chosen_metric = chosen_metric
        self.data_info = {}
        self.pic = "png"  # 图片保存格式

    def start(self):
        # 创建模型存储文件夹
        os.makedirs(self.MODEL_PATH.format(self.cfg_name), exist_ok=True)
        self.model_save_path = self.MODEL_PATH.format(self.cfg_name)

        # 创建配置类实验结果文件夹
        os.makedirs(self.SAVE_PATH.format(self.cfg_name), exist_ok=True)
        # 获取当前时间戳
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        # 创建当前优化结果文件夹
        self.result_save_path = f"{self.SAVE_PATH.format(self.cfg_name)}/{timestamp}"
        os.makedirs(self.result_save_path, exist_ok=True)

        if self.cfg["optimize_method"] == "bayesian_optimize":
            # 贝叶斯超参数优化
            assert "optimize_direction" in self.cfg.keys(), self.WARNING_NOT_FOUND_STR.format("optimize_direction")
            assert "trial_num" in self.cfg.keys(), self.WARNING_NOT_FOUND_STR.format("trial_num")

            # 加载配置文件中的动态参数
            for param_name, param_value in self.cfg.items():
                if isinstance(param_value, dict) and "start" in param_value.keys() and "end" in param_value.keys() and "type" in param_value.keys():
                    self.dynamic_params_cfg[param_name] = param_value

            study = optuna.create_study(study_name="study", directions=self.cfg["optimize_direction"])
            study.optimize(self.run, n_trials=int(self.cfg["trial_num"]))

            # 保存实验最优结果
            chosen_metric_idx = np.where(np.array(self.cfg["optimize_metric"]) == self.chosen_metric)[0][0]
            best_trial_idx = int(np.argmax(np.array([x.values[chosen_metric_idx] for x in study.best_trials])))
            best_trial = study.best_trials[best_trial_idx]
            best_trial_result = {"trial": best_trial_idx + 1}
            best_trial_result.update(dict(zip([f"test_{metric}" for metric in self.cfg["optimize_metric"]], best_trial.values)))
            best_trial_result.update(best_trial.gpt_params)
            print(f"[best] trial metrics: {best_trial_result}\n")
            with open(f"{self.result_save_path}/best_result.json", 'w', encoding='utf-8') as f:
                json.dump(best_trial_result, f, ensure_ascii=False, indent=4)

            # 更新当前变量为最优trial变量
            self.trial_idx = "best"
            for k, v in best_trial.gpt_params.items():
                if k in self.cfg.keys():
                    self.cfg[k] = v

            # 保存所有trial数据
            trials = study.get_trials()
            trials_index_df = pd.DataFrame(data=[_ for _ in range(1, len(trials) + 1)], columns=["trial"])
            trials_df = pd.DataFrame(data=[trial.values for trial in trials], columns=self.cfg["optimize_metric"])
            trials_df.columns = [f"test_{col}" for col in trials_df.columns]
            trials_metrics_df = pd.DataFrame([trial.gpt_params for trial in trials])
            combined_trials_df = pd.concat([trials_index_df, trials_df, trials_metrics_df], axis=1)
            combined_trials_df.to_csv(f"{self.result_save_path}/best_metrics.csv", index=False)

            # 绘制各个超参数对结果影响的重要性柱状图
            self.visualizer.bar_plot_param_importance(
                study,
                self.cfg["optimize_metric"],
                save_path=f"{self.result_save_path}/param_importance_plot.{self.pic}"
            )
            # 绘制帕累托前沿二维投影图
            label_list = ["accuracy", "recall", "precision"]
            metrics_pareto_df = pd.DataFrame(data=[trial.values for trial in trials], columns=self.cfg["optimize_metric"])
            metrics_pareto_df = metrics_pareto_df[label_list]
            self.visualizer.pareto_projection_scatter_3d_plot(
                metrics_pareto_df,
                label_list,
                alpha=0.3,
                s=10,
                figsize=(8, 6),
                save_path=f"{self.result_save_path}/pareto_projection_plot.{self.pic}"
            )

        # 根据最优参数组合额外进行一次模型训练测试及可视化 | 直接进行模型训练测试及可视化
        self.run(to_visualize=True)

    def run(self, trial: optuna.trial = None, to_visualize=False):
        # 加载动态参数值
        self.load_dynamic_params(trial)
        # 导入数据表
        df = self.import_data(self.cfg["data_path"])
        # 处理缺失值
        df = self.process_null_value(df)
        # 处理目标值
        df = self.process_target_value(df)
        # 查看目标值分布
        self.show_y_distribution(df[self.cfg["target_name"]])
        # 数据标准化
        df = self.normalize_value(df)
        # 分割训练集和测试集
        train_data, test_data = self.split_into_train_with_test_data(df, self.cfg["train_ratio"])
        # 创建dataset
        train_dataset = TimeSeriesDataset(train_data, self.cfg["target_name"], self.cfg["timestep"], self.cfg["predict_range"])
        test_dataset = TimeSeriesDataset(test_data, self.cfg["target_name"], self.cfg["timestep"], self.cfg["predict_range"])
        # 创建dataloader
        train_dataloader = DataLoader(train_dataset, self.cfg["batch_size"], shuffle=True)
        test_dataloader = DataLoader(test_dataset, self.cfg["batch_size"], shuffle=False)

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
            loss = loss_function(x_train_pred.flatten(0, 1), y_train.flatten())
            loss.backward()
            optimizer.step()

            train_metrics_tracker.update(real_array=x_train.reshape(x_train.shape[0], -1), pred_array=x_train_pred.reshape(x_train_pred.shape[0], -1))
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
                loss = loss_function(x_test_pred.flatten(0, 1), y_test.flatten())

                test_metrics_tracker.update(real_array=x_test.reshape(x_test.shape[0], -1), pred_array=x_test_pred.reshape(x_test_pred.shape[0], -1))
                loss_list.append(loss.item())
                test_bar.desc = "test epoch[{}/{}] (init:{:4f}) loss:{:.4f}".format(epoch, self.cfg["epochs"], loss_list[0], loss)

        test_metrics_tracker.add_new_metric_list("loss", loss_list)
        print(f"[test] metrics: {test_metrics_tracker.get_metrics()}\n")
        return test_metrics_tracker

    def best_return(self, best_metrics={}):
        if len(best_metrics) == 0:
            return tuple(0 for _ in self.cfg["optimize_metric"])

        return tuple(best_metrics[metric] for metric in self.cfg["optimize_metric"])

    def update_trial_idx(self):
        if isinstance(self.trial_idx, int):
            # trial_idx代表超参数优化次数
            self.trial_idx += 1

    def save_data_info(self, best_metrics_tracker, mark="trial"):
        # 保存所有epoch数据
        best_metrics_tracker.save_data(f"{self.result_save_path}/{mark}_{self.trial_idx}.csv")

        data_info_path = f"{self.result_save_path}/data_info_trial_{self.trial_idx}.json"
        if not os.path.exists(data_info_path):
            with open(data_info_path, 'w', encoding='utf-8') as f:
                json.dump(self.data_info, f, ensure_ascii=False, indent=4)

    def show_y_distribution(self, data, mark: str = ""):
        raw_y_distribution = dict(Counter(data).most_common())
        print(f"{mark} y distribution: {raw_y_distribution}")
        self.data_info[f"{mark}_y_distribution"] = raw_y_distribution

    def load_dynamic_params(self, trial):
        if trial is not None:
            for param_name, param_value in self.dynamic_params_cfg.items():
                if param_value["type"] == "int":
                    self.cfg[param_name] = trial.suggest_int(param_name, param_value["start"], param_value["end"])
                elif param_value["type"] == "float":
                    self.cfg[param_name] = trial.suggest_float(param_name, param_value["start"], param_value["end"])

    def import_data(self, data_path, encoding='utf-8'):
        df = pd.read_csv(data_path.replace('@', GeneralTool.root_path), encoding=encoding)
        print(f"imported data (length: {df.shape[0]}, features: {df.shape[1]})")
        self.data_info["imported_data_length"] = df.shape[0]
        self.data_info["imported_data_features"] = df.shape[1]

        return df

    def process_null_value(self, df):
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

        return df

    def process_target_value(self, df):
        target_name = self.cfg["target_name"]

        if self.cfg["target_method"] == "all_binarize":
            label_num = Counter(df[target_name]).total()  # 标签种类数
            print(f"types of label: {label_num}")
            df[target_name] = df[target_name].map(
                dict(zip([i for i in range(label_num + 1)], [0] + [1] * label_num))
            )
        elif self.cfg["target_method"] == "specific_target":
            assert "target_value" in self.cfg.keys(), self.WARNING_NOT_FOUND_STR.format("target_value")
            df[target_name] = np.where(df[target_name] == self.cfg["target_value"], 1, 0)
        elif self.cfg["target_method"] == "multi_labels":
            pass
        elif self.cfg["target_method"] == "single_var":
            pass
        elif self.cfg["target_method"] == "multi_var":
            pass

        print(f"processed target data (length: {df.shape[0]}, features: {df.shape[1]})")
        self.data_info["processed_target_data_length"] = df.shape[0]
        self.data_info["processed_target_data_features"] = df.shape[1]

        return df

    def normalize_value(self, df):
        # 如果输入数据为ndarray
        if isinstance(df, np.ndarray):
            if self.cfg["normalize_data_method"] == "z_score":
                scaler = StandardScaler()
            elif self.cfg["normalize_data_method"] == "min_max":
                scaler = MinMaxScaler()
            df = scaler.fit_transform(df)

            print(f"normalized data (length: {df.shape[0]}, features: {df.shape[1]})")
            self.data_info["normalized_data_length"] = df.shape[0]
            self.data_info["normalized_data_features"] = df.shape[1]

            return df

        # 分类任务的目标值不进行标准化，回归任务的目标值进行标准化
        if self.cfg["normalize_data_method"] == "z_score":
            scaler = StandardScaler()
            if self.cfg["task_type"] == "classification" and "target_name" in self.cfg:
                _target_df = df[self.cfg["target_name"]]
                x_col_list = [col for col in df.columns if col != self.cfg["target_name"]]
                df = pd.DataFrame(
                    scaler.fit_transform(df.loc[:, x_col_list]),
                    columns=x_col_list
                )
                df.insert(0, self.cfg["target_name"], _target_df)

            else:
                df = pd.DataFrame(
                    scaler.fit_transform(df),
                    columns=df.columns
                )
        elif self.cfg["normalize_data_method"] == "min_max":
            scaler = MinMaxScaler()
            if self.cfg["task_type"] == "classification" and "target_name" in self.cfg:
                _target_df = df[self.cfg["target_name"]]
                x_col_list = [col for col in df.columns if col != self.cfg["target_name"]]
                df = pd.DataFrame(
                    scaler.fit_transform(df.loc[:, x_col_list]),
                    columns=x_col_list
                )
                df.insert(0, self.cfg["target_name"], _target_df)

            else:
                df = pd.DataFrame(
                    scaler.fit_transform(df),
                    columns=df.columns
                )

        print(f"normalized data (length: {df.shape[0]}, features: {df.shape[1]})")
        self.data_info["normalized_data_length"] = df.shape[0]
        self.data_info["normalized_data_features"] = df.shape[1]

        return df

    def stratified_shuffle_split_into_train_with_test_data(self, df, train_ratio, target_name):
        # 创建分层抽样分割器
        splitter = StratifiedShuffleSplit(
            n_splits=1,  # 只进行一次分割
            train_size=train_ratio,
            random_state=GeneralTool.seed
        )

        # 获取训练和测试索引
        train_index, test_index = [x for x in splitter.split(df, df[target_name])][0]
        train_data = df.iloc[train_index, :]
        test_data = df.iloc[test_index, :]

        return train_data, test_data

    def split_into_train_with_test_data(self, data, train_ratio):
        # 随机打乱数据
        data.sample(
            frac=1,  # 采样比例为100%
            random_state=GeneralTool.seed
        ).reset_index(drop=True, inplace=True)

        train_size = int(train_ratio * len(data))
        train_data = data[:train_size]
        test_data = data[train_size:]

        print(f"train_data shape: {train_data.shape}, test_data shape: {test_data.shape}")
        self.data_info["train_data_shape"], self.data_info["test_data_shape"] = train_data.shape, test_data.shape

        return train_data, test_data

