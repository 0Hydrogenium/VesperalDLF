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

from module.LossFunctionFactory import LossFunctionFactory
from module.ModelFactory import ModelFactory
from module.OptimizerFactory import OptimizerFactory
from utils.GeneralTool import GeneralTool
from utils.metric_tracker.BestMetricsTracker import BestMetricsTracker
from utils.metric_tracker.ClassificationMetricsTracker import ClassificationMetricsTracker
from utils.visualizer.Visualizer import Visualizer

WARNING_NOT_FOUND_STR = "[config] '{}' is not found"

SAVE_PATH = "./result/{}"


class Trainer:
    def __init__(self, cfg: dict, cfg_name: str, chosen_metric: str = "recall"):
        self.cfg = cfg  # 当前参数配置
        self.dynamic_params_cfg = {}  # 备份动态参数配置
        self.cfg_name = cfg_name
        self.result_save_path = ""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"device: {self.device}")
        self.trial_idx = 1
        self.visualizer = Visualizer()

        self.chosen_metric = chosen_metric

    def start(self):
        # 创建配置类实验结果文件夹
        os.makedirs(SAVE_PATH.format(self.cfg_name), exist_ok=True)
        # 获取当前时间戳
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        # 创建当前优化结果文件夹
        self.result_save_path = f"{SAVE_PATH.format(self.cfg_name)}/{timestamp}"
        os.makedirs(self.result_save_path, exist_ok=True)

        if self.cfg["optimize_method"] == "bayesian_optimize":
            # 贝叶斯超参数优化
            assert "optimize_direction" in self.cfg.keys(), WARNING_NOT_FOUND_STR.format("optimize_direction")
            assert "trial_num" in self.cfg.keys(), WARNING_NOT_FOUND_STR.format("trial_num")

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
            best_trial_result.update(best_trial.params)
            print(f"[best] trial metrics: {best_trial_result}\n")
            with open(f"{self.result_save_path}/best_result.json", 'w', encoding='utf-8') as f:
                json.dump(best_trial_result, f, ensure_ascii=False, indent=4)

            # 更新当前变量为最优trial变量
            self.trial_idx = "best"
            for k, v in best_trial.params.items():
                if k in self.cfg.keys():
                    self.cfg[k] = v

            # 保存所有trial数据
            trials = study.get_trials()
            trials_index_df = pd.DataFrame(data=[_ for _ in range(1, len(trials) + 1)], columns=["trial"])
            trials_df = pd.DataFrame(data=[trial.values for trial in trials], columns=self.cfg["optimize_metric"])
            trials_df.columns = [f"test_{col}" for col in trials_df.columns]
            trials_metrics_df = pd.DataFrame([trial.params for trial in trials])
            combined_trials_df = pd.concat([trials_index_df, trials_df, trials_metrics_df], axis=1)
            combined_trials_df.to_csv(f"{self.result_save_path}/best_metrics.csv", index=False)

            # 绘制各个超参数对结果影响的重要性柱状图
            self.visualizer.bar_plot_param_importance(
                study,
                self.cfg["optimize_metric"],
                save_path=f"{self.result_save_path}/param_importance_plot.svg"
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
                save_path=f"{self.result_save_path}/pareto_projection_plot.svg"
            )

        # 根据最优参数组合额外进行一次模型训练测试及可视化 | 直接进行模型训练测试及可视化
        self.run(to_visualize=True)

    def run(self, trial: optuna.trial = None, to_visualize=False):
        # 加载动态参数值
        if trial is not None:
            for param_name, param_value in self.dynamic_params_cfg.items():
                if param_value["type"] == "int":
                    self.cfg[param_name] = trial.suggest_int(param_name, param_value["start"], param_value["end"])
                elif param_value["type"] == "float":
                    self.cfg[param_name] = trial.suggest_float(param_name, param_value["start"], param_value["end"])

        data_info = {}
        # 导入数据表
        df = self.import_data(self.cfg["data_path"])
        print(f"imported data (length: {df.shape[0]}, features: {df.shape[1]})")
        data_info["imported_data_length"] = df.shape[0]
        data_info["imported_data_features"] = df.shape[1]
        # 处理缺失值
        df = self.process_null_value(df)
        print(f"processed null data (length: {df.shape[0]}, features: {df.shape[1]})")
        data_info["processed_null_data_length"] = df.shape[0]
        data_info["processed_null_data_features"] = df.shape[1]
        # 处理目标值
        df = self.process_target_value(df)
        raw_y_distribution = dict(Counter(df[self.cfg["target_name"]]).most_common())
        print(f"raw y distribution: {raw_y_distribution}")
        data_info["raw_y_distribution"] = raw_y_distribution
        print(f"processed target data (length: {df.shape[0]}, features: {df.shape[1]})")
        data_info["processed_target_data_length"] = df.shape[0]
        data_info["processed_target_data_features"] = df.shape[1]
        # 数据标准化
        df = self.normalize_value(df)
        print(f"normalized data (length: {df.shape[0]}, features: {df.shape[1]})")
        data_info["normalized_data_length"] = df.shape[0]
        data_info["normalized_data_features"] = df.shape[1]

        # 分割x和y
        data_x, data_y = self.split_into_x_with_y(df)
        y_distribution = dict(Counter(data_y.tolist()).most_common())
        print(f"y distribution: {y_distribution}")
        data_info["y_distribution"] = y_distribution
        # 分割训练集和测试集
        train_data_x, train_data_y, test_data_x, test_data_y = self.split_into_train_with_test_data(data_x, data_y)
        # 转换为时间序列数据
        x_train, y_train = self.convert_from_data_to_time_series(train_data_x, train_data_y)
        x_test, y_test = self.convert_from_data_to_time_series(test_data_x, test_data_y)
        print(f"converted from data to time series (x_train shape: {x_train.shape}, y_train shape: {y_train.shape}, x_test shape: {x_test.shape}, y_test shape: {y_test.shape})")
        data_info["x_train_shape"], data_info["y_train_shape"], data_info["x_test_shape"], data_info["y_test_shape"] = x_train.shape, y_train.shape, x_test.shape, y_test.shape
        # ndarray格式转换为tensor格式
        x_train_tensor = self.convert_from_ndarray_to_tensor(x_train)
        y_train_tensor = self.convert_from_ndarray_to_tensor(y_train)
        x_test_tensor = self.convert_from_ndarray_to_tensor(x_test)
        y_test_tensor = self.convert_from_ndarray_to_tensor(y_test)
        # 转换为数据加载器
        train_loader, test_loader = self.get_data_loader(x_train_tensor, y_train_tensor, x_test_tensor, y_test_tensor)

        # 根据数据获取模型的输入维度和输出维度
        self.cfg["model__input_dim"] = data_x.shape[1]
        self.cfg["model__output_dim"] = 2

        # 创建模型
        model = ModelFactory.get(self.cfg, self.cfg["model_name"]).to(self.device)
        # 创建损失函数
        loss_function = LossFunctionFactory.get(self.cfg, self.cfg["loss_function_name"])
        # 创建优化器
        optimizer = OptimizerFactory.get(self.cfg, self.cfg["optimizer_name"], model)

        # 迭代训练测试模型
        # 创建epochs内最佳指标追踪器
        best_metrics_tracker = BestMetricsTracker()
        for epoch in range(1, self.cfg["epochs"] + 1):
            # 训练模型
            train_metrics_tracker, model = self.train_model(model, loss_function, optimizer, train_loader, epoch)
            print(f"[train] metrics: {train_metrics_tracker.get_metrics()}\n")
            # 测试模型
            test_metrics_tracker = self.test_model(model, loss_function, test_loader, epoch)
            print(f"[test] metrics: {test_metrics_tracker.get_metrics()}\n")
            # 更新epochs内最佳模型+参数+指标
            best_metrics_tracker.add(train_metrics_tracker, test_metrics_tracker, self.cfg, epoch)

            # 早停策略
            if test_metrics_tracker.get_metrics()["accuracy"] > 0.99:
                print(f"epoch {epoch} early stopping")
                break

        # 获取epochs内最佳模型+参数+指标
        best_metrics, best_cfg, best_epoch = best_metrics_tracker.get_best(self.cfg["metrics_filter"], self.cfg["optimize_metric"], self.cfg["optimize_direction"])
        print(f"[best] epoch of metrics: {best_epoch}")
        print(f"[best] metrics: {best_metrics}\n")
        # 保存所有epoch数据
        best_metrics_tracker.save_data(f"{self.result_save_path}/trial_{self.trial_idx}.csv")

        # 保存数据信息
        with open(f"{self.result_save_path}/data_info_trial_{self.trial_idx}.json", 'w', encoding='utf-8') as f:
            json.dump(data_info, f, ensure_ascii=False, indent=4)

        if isinstance(self.trial_idx, int):
            # trial_idx代表超参数优化次数
            self.trial_idx += 1

        # 无可行指标组合处理
        if best_epoch is None:
            return tuple(0 for _ in self.cfg["optimize_metric"])

        # 绘制模型训练测试曲线
        if to_visualize:
            train_loss_list = pd.DataFrame(best_metrics_tracker.train_metrics_list)["loss"].tolist()
            test_loss_list = pd.DataFrame(best_metrics_tracker.test_metrics_list)["loss"].tolist()

            train_loss_mean = np.array([sum(loss) / len(loss) for loss in train_loss_list])
            train_loss_std = np.array([[np.std(np.array(loss)) for loss in train_loss_list]]).reshape(-1)
            test_loss_mean = np.array([sum(loss) / len(loss) for loss in test_loss_list])
            test_loss_std = np.array([[np.std(np.array(loss)) for loss in test_loss_list]]).reshape(-1)

            self.visualizer.loss_plot_train_test_curve_with_std(
                train_loss_mean,
                train_loss_std,
                test_loss_mean,
                test_loss_std,
                scale=0.5,
                alpha=0.2,
                figsize=(8, 6),
                save_path=f"{self.result_save_path}/loss_plot.svg"
            )

        return tuple(best_metrics[metric] for metric in self.cfg["optimize_metric"])

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
            loss = loss_function(x_train_pred, y_train)
            loss.backward()
            optimizer.step()

            y_train = y_train.detach().cpu().numpy()
            x_train_pred = x_train_pred.detach().cpu().numpy()

            train_metrics_tracker.update(real_array=y_train, pred_array=x_train_pred)
            loss_list.append(loss.item())
            train_bar.desc = "train epoch[{}/{}] (init:{:4f}) loss:{:.4f}".format(epoch, self.cfg["epochs"], loss_list[0], loss)

        train_metrics_tracker.add_new_metric_list("loss", loss_list)
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
                loss = loss_function(x_test_pred, y_test)

                y_test = y_test.detach().cpu().numpy()
                x_test_pred = x_test_pred.detach().cpu().numpy()

                test_metrics_tracker.update(real_array=y_test, pred_array=x_test_pred)
                loss_list.append(loss.item())
                test_bar.desc = "test epoch[{}/{}] (init:{:4f}) loss:{:.4f}".format(epoch, self.cfg["epochs"], loss_list[0], loss)

        test_metrics_tracker.add_new_metric_list("loss", loss_list)
        return test_metrics_tracker

    def import_data(self, data_path):
        df = pd.read_csv(data_path)

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
            assert "target_value" in self.cfg.keys(), WARNING_NOT_FOUND_STR.format("target_value")
            df[target_name] = np.where(df[target_name] == self.cfg["target_value"], 1, 0)
        elif self.cfg["target_method"] == "multi_labels":
            pass
        elif self.cfg["target_method"] == "single_var":
            pass
        elif self.cfg["target_method"] == "multi_var":
            pass

        return df

    def normalize_value(self, df):
        # 如果输入数据为ndarray
        if isinstance(df, np.ndarray):
            if self.cfg["normalize_data_method"] == "z_score":
                scaler = StandardScaler()
            elif self.cfg["normalize_data_method"] == "min_max":
                scaler = MinMaxScaler()
            df = scaler.fit_transform(df)
            return df

        target_name = self.cfg["target_name"]
        # 分类任务的目标值不进行标准化，回归任务的目标值进行标准化
        if self.cfg["normalize_data_method"] == "z_score":
            scaler = StandardScaler()
            if self.cfg["task_type"] == "classification":
                _target_df = df[target_name]
                x_col_list = [col for col in df.columns if col != target_name]
                df = pd.DataFrame(
                    scaler.fit_transform(df.loc[:, x_col_list]),
                    columns=x_col_list
                )
                df.insert(0, target_name, _target_df)

            elif self.cfg["task_type"] == "regression":
                df = pd.DataFrame(
                    scaler.fit_transform(df),
                    columns=df.columns
                )
        elif self.cfg["normalize_data_method"] == "min_max":
            scaler = MinMaxScaler()
            if self.cfg["task_type"] == "classification":
                _target_df = df[target_name]
                x_col_list = [col for col in df.columns if col != target_name]
                df = pd.DataFrame(
                    scaler.fit_transform(df.loc[:, x_col_list]),
                    columns=x_col_list
                )
                df.insert(0, target_name, _target_df)

            elif self.cfg["task_type"] == "regression":
                df = pd.DataFrame(
                    scaler.fit_transform(df),
                    columns=df.columns
                )

        return df

    def split_into_x_with_y(self, df):
        target_name = self.cfg["target_name"]

        data_x = np.array(df)
        data_y = np.array(df.loc[:, target_name])

        return data_x, data_y

    def stratified_shuffle_split_into_train_with_test_data(self, data_x, data_y, mapped_data_y):
        # 创建分层抽样分割器
        splitter = StratifiedShuffleSplit(
            n_splits=1,  # 只进行一次分割
            train_size=self.cfg["train_ratio"],
            random_state=GeneralTool.seed
        )

        # 获取训练和测试索引
        for train_index, test_index in splitter.split(data_x, data_y):
            x_train = data_x[train_index]
            y_train = data_y[train_index]
            mapped_y_train = mapped_data_y[train_index]
            x_test = data_x[test_index]
            y_test = data_y[test_index]
            mapped_y_test = mapped_data_y[test_index]

        return x_train, y_train, mapped_y_train, x_test, y_test, mapped_y_test

    def split_into_train_with_test_data(self, data_x, data_y):
        # 生成随机索引并打乱
        indices = np.arange(len(data_x))
        np.random.shuffle(indices)
        data_x = data_x[indices]
        data_y = data_y[indices]

        train_size = int(self.cfg["train_ratio"] * len(data_x))
        train_data_x = data_x[: train_size, :, :]
        train_data_y = data_y[: train_size]
        test_data_x = data_x[train_size:, :, :]
        test_data_y = data_y[train_size:]

        return train_data_x, train_data_y, test_data_x, test_data_y

    def convert_from_data_to_time_series(self, data_x, data_y=None):
        if self.cfg["data_type"] == "time_series":
            time_y = None
            # 转换为时间窗口
            timestep = self.cfg["timestep"]
            predict_range = self.cfg["predict_range"]

            train_time_data_x = []
            train_time_data_y = []
            for idx in range(len(data_x) - timestep - predict_range + 1):
                train_time_data_x.append(data_x[idx: idx + timestep])
                if data_y is not None:
                    train_time_data_y.append(data_y[idx + timestep + predict_range - 1])
            time_x = np.array(train_time_data_x)
            if data_y is not None:
                time_y = np.array(train_time_data_y)

            return time_x, time_y

    def convert_from_ndarray_to_tensor(self, ndarray):
        tensor = torch.from_numpy(ndarray).to(torch.float32)

        return tensor

    def get_data_loader(self, x_train_tensor, y_train_tensor, x_test_tensor, y_test_tensor):
        train_dataset = torch.utils.data.TensorDataset(x_train_tensor, y_train_tensor)
        test_dataset = torch.utils.data.TensorDataset(x_test_tensor, y_test_tensor)

        train_loader = torch.utils.data.DataLoader(train_dataset, self.cfg["batch_size"], True)
        test_loader = torch.utils.data.DataLoader(test_dataset, self.cfg["batch_size"], False)

        return train_loader, test_loader
