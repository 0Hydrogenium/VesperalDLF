import os
import pandas as pd
import torch
import optuna
import json
from collections import Counter
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import roc_curve, auc, precision_recall_curve, accuracy_score, precision_score, recall_score, f1_score

from module.LossFunctionFactory import LossFunctionFactory
from module.ModelFactory import ModelFactory
from module.OptimizerFactory import OptimizerFactory
from module.dataset.WeibullMappingDataset import WeibullMappingDataset
from trainer.Trainer import Trainer
from utils.metric_tracker.BestMetricsTracker import BestMetricsTracker
from utils.metric_tracker.ClassificationMetricsTracker import ClassificationMetricsTracker
from utils.metric_tracker.RegressionMetricsTracker import RegressionMetricsTracker
from utils.visualizer.DataMappingVisualizer import DataMappingVisualizer


class WeibullTwinDAETrainer(Trainer):
    def __init__(self, cfg, cfg_name):
        super().__init__(cfg, cfg_name)
        self.visualizer = DataMappingVisualizer()

    def run(self, trial: optuna.trial = None, to_visualize=False):
        # 加载动态参数值
        self.load_dynamic_params(trial)

        if "." not in self.cfg["data_path"].split("/")[-1]:
            """
                多个同质数据集导入合并
            """

            train_data_list = []
            test_data_list = []
            for data_path in os.listdir(self.cfg["data_path"]):
                # 导入数据表
                current_data_path = f"{self.cfg['data_path']}/{data_path}"
                df = self.import_data(current_data_path)

                # 由于后续使用多头注意力机制，需要将特征数量凑成偶数
                df.insert(df.shape[1], "blank", [0 for _ in range(len(df))])

                # 处理缺失值
                df = self.process_null_value(df)
                # 查看目标值分布
                self.show_y_distribution(df[self.cfg["target_name"]], mark=f"{current_data_path.split('/')[-1]} dataset before")
                # 处理目标值
                df = self.process_target_value(df)
                # 查看目标值分布
                self.show_y_distribution(df[self.cfg["target_name"]], mark=f"{current_data_path.split('/')[-1]} dataset after")

                # 判断当前数据集是否有正样本
                if len(df[df[self.cfg["target_name"]] == 1]) < 2:
                    print(f"{current_data_path.split('/')[-1]} dataset skip")
                    continue

                # 数据标准化
                df = self.normalize_value(df)
                # 分割训练集和测试集
                sub_train_data, sub_test_data = self.stratified_shuffle_split_into_train_with_test_data(df, self.cfg["train_ratio"], self.cfg["target_name"])
                # 查看目标值分布
                self.show_y_distribution(sub_train_data[self.cfg["target_name"]], mark="train mapping")
                self.show_y_distribution(sub_test_data[self.cfg["target_name"]], mark="test mapping")

                train_data_list.append(sub_train_data)
                test_data_list.append(sub_test_data)

            # 合并多个数据集
            train_data = pd.concat(train_data_list, axis=0)
            test_data = pd.concat(test_data_list, axis=0)
            # 查看目标值分布
            self.show_y_distribution(train_data[self.cfg["target_name"]], mark="total train mapping")
            self.show_y_distribution(test_data[self.cfg["target_name"]], mark="total test mapping")

        else:
            """
                单个数据集数据导入
            """

            # 导入数据表
            df = self.import_data(self.cfg["data_path"])

            # 由于后续使用多头注意力机制，需要将特征数量凑成偶数
            df.insert(df.shape[1], "blank", [0 for _ in range(len(df))])

            # 处理缺失值
            df = self.process_null_value(df)
            # 查看目标值分布
            self.show_y_distribution(df[self.cfg["target_name"]], mark="before")
            # 处理目标值
            df = self.process_target_value(df)
            # 查看目标值分布
            self.show_y_distribution(df[self.cfg["target_name"]], mark="after")
            # 数据标准化
            df = self.normalize_value(df)

            # 数据用于可视化
            total_dataset = WeibullMappingDataset(
                df,
                self.cfg["target_name"],
                self.cfg["timestep"],
                self.cfg["predict_range"],
                k=self.cfg["target_mapping_k"],
                cumulative_prob=self.cfg["cumulative_prob"],
                degradation_factor=self.cfg["degradation_factor"],
                initial_baseline=self.cfg["initial_baseline"],
                baseline_growth=self.cfg["baseline_growth"]
            )
            if to_visualize:
                self.visualizer.line_chart_mapping_y_distribution_discrete_with_continuous(
                    total_dataset,
                    self.cfg["target_value"],
                    figsize=(14, 6),
                    alpha=0.3,
                    s=4,
                    save_path=f"{self.result_save_path}/y_distribution_plot_trial_{self.trial_idx}.svg"
                )
                self.visualizer.line_chart_mapping_y_distribution_param_trend(
                    total_dataset,
                    self.cfg["target_value"],
                    figsize=(8, 6),
                    alpha=0.3,
                    s=4,
                    save_path1=f"{self.result_save_path}/baseline_param_y_distribution_plot_trial_{self.trial_idx}.svg",
                    save_path2=f"{self.result_save_path}/etas_param_y_distribution_plot_trial_{self.trial_idx}.svg"
                )

            # 分割训练集和测试集
            train_data, test_data = self.stratified_shuffle_split_into_train_with_test_data(df, self.cfg["train_ratio"], self.cfg["target_name"])
            # 查看目标值分布
            self.show_y_distribution(train_data[self.cfg["target_name"]], mark="train mapping")
            self.show_y_distribution(test_data[self.cfg["target_name"]], mark="test mapping")

        # 创建dataset
        train_dataset = WeibullMappingDataset(
            train_data,
            self.cfg["target_name"],
            self.cfg["timestep"],
            self.cfg["predict_range"],
            k=self.cfg["target_mapping_k"],
            cumulative_prob=self.cfg["cumulative_prob"],
            degradation_factor=self.cfg["degradation_factor"],
            initial_baseline=self.cfg["initial_baseline"],
            baseline_growth=self.cfg["baseline_growth"]
        )
        test_dataset = WeibullMappingDataset(
            test_data,
            self.cfg["target_name"],
            self.cfg["timestep"],
            self.cfg["predict_range"],
            k=self.cfg["target_mapping_k"],
            cumulative_prob=self.cfg["cumulative_prob"],
            degradation_factor=self.cfg["degradation_factor"],
            initial_baseline=self.cfg["initial_baseline"],
            baseline_growth=self.cfg["baseline_growth"]
        )
        # 创建dataloader
        train_dataloader = DataLoader(train_dataset, self.cfg["batch_size"], True)
        test_dataloader = DataLoader(test_dataset, self.cfg["batch_size"], False)

        # 创建模型
        p_model = ModelFactory.get(self.cfg, self.cfg["positive_model_name"]).to(self.device)
        n_model = ModelFactory.get(self.cfg, self.cfg["negative_model_name"]).to(self.device)
        # 创建损失函数
        p_loss_function = LossFunctionFactory.get(self.cfg, self.cfg["loss_function_name"])
        n_loss_function = LossFunctionFactory.get(self.cfg, self.cfg["loss_function_name"])
        # 创建优化器
        p_optimizer = OptimizerFactory.get(self.cfg, self.cfg["optimizer_name"], p_model)
        n_optimizer = OptimizerFactory.get(self.cfg, self.cfg["optimizer_name"], n_model)

        # 迭代训练测试模型
        n_best_metrics_tracker = BestMetricsTracker()
        p_best_metrics_tracker = BestMetricsTracker()
        best_metrics_tracker = BestMetricsTracker()  # 创建epochs内最佳指标追踪器
        for epoch in range(1, self.cfg["epochs"] + 1):
            """
                训练测试负样本重构模型
            """
            # 训练负重构模型
            n_train_metrics_tracker, n_model = self.train_twin_model("negative", n_model, n_loss_function, n_optimizer, train_dataloader, epoch)
            # 测试负重构模型
            n_test_metrics_tracker = self.test_twin_model("negative", n_model, n_loss_function, test_dataloader, epoch)
            # 更新epochs内最佳模型+参数+指标
            n_best_metrics_tracker.add(n_train_metrics_tracker, n_test_metrics_tracker, self.cfg, epoch)

            """
                训练测试正样本重构模型
            """
            # 训练正重构模型
            p_train_metrics_tracker, p_model = self.train_twin_model("positive", p_model, p_loss_function, p_optimizer, train_dataloader, epoch)
            # 测试正重构模型
            p_test_metrics_tracker = self.test_twin_model("positive", p_model, p_loss_function, test_dataloader, epoch)
            # 更新epochs内最佳模型+参数+指标
            p_best_metrics_tracker.add(p_train_metrics_tracker, p_test_metrics_tracker, self.cfg, epoch)

            """
                计算偏差
            """
            train_deviation_label_array, train_deviation_array = self.compute_deviation(p_model, n_model, train_dataloader, round_digits=2)
            test_deviation_label_array, test_deviation_array = self.compute_deviation(p_model, n_model, test_dataloader, round_digits=2)

            if to_visualize:
                # 绘制偏差散点图
                self.visualizer.deviation_scatter_plot(
                    train_deviation_label_array,
                    train_deviation_array,
                    n_alpha=0.2,
                    p_alpha=0.2,
                    s=10,
                    figsize=(8, 6),
                    save_path=f"{self.result_save_path}/train_deviation_scatter_plot_epoch_{epoch}.svg"
                )
                self.visualizer.deviation_scatter_plot(
                    test_deviation_label_array,
                    test_deviation_array,
                    n_alpha=0.2,
                    p_alpha=0.2,
                    s=10,
                    figsize=(8, 6),
                    save_path=f"{self.result_save_path}/test_deviation_scatter_plot_epoch_{epoch}.svg"
                )

            """
                计算ROC曲线相关参数指标:

                遍历所有唯一的预测概率值作为阈值
                剔除次优阈值
                若相邻阈值的(FPR,TPR)相同或线性相关，则合并中间阈值
                添加额外两个固定端点
            """
            train_fpr, train_tpr, train_thresholds, train_roc_auc = self.compute_roc_auc(train_deviation_label_array, train_deviation_array)
            test_fpr, test_tpr, test_thresholds, test_roc_auc = self.compute_roc_auc(test_deviation_label_array, test_deviation_array)
            # 计算不同阈值下的分类指标
            total_train_metrics_tracker, train_threshold_accuracy, train_threshold_precision, train_threshold_recall, train_threshold_f1 = self.compute_classification_metrics_on_thresholds(train_deviation_label_array, train_deviation_array, train_thresholds)
            total_test_metrics_tracker, test_threshold_accuracy, test_threshold_precision, test_threshold_recall, test_threshold_f1 = self.compute_classification_metrics_on_thresholds(test_deviation_label_array, test_deviation_array, test_thresholds)
            # 计算PR曲线的AP分数
            train_pr_ap = self.compute_pr_ap(train_threshold_precision, train_threshold_recall)
            test_pr_ap = self.compute_pr_ap(test_threshold_precision, test_threshold_recall)

            total_train_metrics_tracker.add_new_metric("roc_auc", train_roc_auc)
            total_test_metrics_tracker.add_new_metric("roc_auc", test_roc_auc)
            total_train_metrics_tracker.add_new_metric("pr_ap", train_pr_ap)
            total_test_metrics_tracker.add_new_metric("pr_ap", test_pr_ap)

            # 更新epochs内最佳模型+参数+指标
            best_metrics_tracker.add(total_train_metrics_tracker, total_test_metrics_tracker, self.cfg, epoch)

            if to_visualize:
                # 绘制ROC曲线
                self.visualizer.roc_curve_on_thresholds(
                    train_fpr,
                    train_tpr,
                    train_roc_auc,
                    figsize=(8, 6),
                    save_path=f"{self.result_save_path}/train_roc_curve_on_thresholds_epoch_{epoch}.svg"
                )
                self.visualizer.roc_curve_on_thresholds(
                    test_fpr,
                    test_tpr,
                    test_roc_auc,
                    figsize=(8, 6),
                    save_path=f"{self.result_save_path}/test_roc_curve_on_thresholds_epoch_{epoch}.svg"
                )
                # 绘制PR曲线
                self.visualizer.pr_curve_on_thresholds(
                    train_threshold_recall,
                    train_threshold_precision,
                    train_pr_ap,
                    sum(train_dataset.y_list) / len(train_dataset.y_list),
                    figsize=(8, 6),
                    save_path=f"{self.result_save_path}/train_pr_curve_on_thresholds_epoch_{epoch}.svg"
                )
                self.visualizer.pr_curve_on_thresholds(
                    test_threshold_recall,
                    test_threshold_precision,
                    test_pr_ap,
                    sum(test_dataset.y_list) / len(test_dataset.y_list),
                    figsize=(8, 6),
                    save_path=f"{self.result_save_path}/test_pr_curve_on_thresholds_epoch_{epoch}.svg"
                )

        # 获取epochs内最佳模型+参数+指标
        best_metrics, best_cfg, best_epoch = best_metrics_tracker.get_best(self.cfg["metrics_filter"], self.cfg["optimize_metric"], self.cfg["optimize_direction"])
        # 保存数据信息
        self.save_data_info(p_best_metrics_tracker, "positive_model_trial")
        self.save_data_info(n_best_metrics_tracker, "negative_model_trial")
        self.save_data_info(best_metrics_tracker, "trial")

        # 绘制模型训练测试曲线
        if to_visualize:
            # 负重构模型
            self.visualizer.loss_plot_train_test_curve_with_std(
                n_best_metrics_tracker,
                scale=0.5,
                alpha=0.2,
                figsize=(8, 6),
                save_path=f"{self.result_save_path}/negative_model_loss_plot.svg"
            )

            # 正重构模型
            self.visualizer.loss_plot_train_test_curve_with_std(
                p_best_metrics_tracker,
                scale=0.5,
                alpha=0.2,
                figsize=(8, 6),
                save_path=f"{self.result_save_path}/positive_model_loss_plot.svg"
            )

        # 更新超参数优化次数
        self.update_trial_idx()

        return self.best_return(best_metrics)

    def compute_classification_metrics_on_thresholds(self, data_y, data_y_proba, thresholds):
        accuracy_list, precision_list, recall_list, f1_list = [], [], [], []
        metrics_tracker = ClassificationMetricsTracker()
        thresholds_bar = tqdm(thresholds)
        for idx, threshold in enumerate(thresholds_bar):
            # 应用当前阈值进行二分类
            pred = (data_y_proba >= threshold).astype(int)
            # 分类指标器中的无效值不考虑，但是函数返回的分类指标的数量需要相同，因此各自分别计算
            # 分类指标器计算的是代表性指标值
            # 而独立计算的分类指标为返回数组
            metrics_tracker.update(real_array=data_y, pred_array=pred)

            accuracy = accuracy_score(y_true=data_y, y_pred=pred)
            precision = precision_score(y_true=data_y, y_pred=pred, zero_division=0)
            recall = recall_score(y_true=data_y, y_pred=pred, zero_division=0)
            f1 = f1_score(y_true=data_y, y_pred=pred, zero_division=0)

            accuracy_list.append(accuracy)
            precision_list.append(precision)
            recall_list.append(recall)
            f1_list.append(f1)

            thresholds_bar.desc = "compute classification metrics on thresholds"

        threshold_accuracy = np.array(accuracy_list)
        threshold_precision = np.array(precision_list)
        threshold_recall = np.array(recall_list)
        threshold_f1 = np.array(f1_list)

        return metrics_tracker, threshold_accuracy, threshold_precision, threshold_recall, threshold_f1

    def compute_roc_auc(self, data_y, data_y_proba):
        fpr, tpr, thresholds_roc = roc_curve(data_y, data_y_proba)
        roc_auc = auc(fpr, tpr)
        return fpr, tpr, thresholds_roc, roc_auc

    def compute_pr_ap(self, precision, recalls):
        # 梯度积分法计算AP分数
        ap = auc(recalls, precision)
        return ap

    def compute_deviation(self, p_model, n_model, loader, round_digits):
        p_model.eval()
        n_model.eval()
        deviation_label_list = []
        deviation_list = []
        bar = tqdm(loader)
        with torch.no_grad():
            for idx, data_tensors in enumerate(bar):
                x_data, y_data, mapped_y_data = data_tensors
                x_data = x_data.to(self.device)
                p_x_data_pred, _ = p_model(x_data)
                n_x_data_pred, _ = n_model(x_data)

                deviation = ((n_x_data_pred - x_data) ** 2 - (p_x_data_pred - x_data) ** 2).mean(dim=2).mean(dim=1)

                deviation = deviation.detach().cpu().numpy()
                y_data = y_data.detach().cpu().numpy()
                deviation_list.append(deviation)
                deviation_label_list.append(y_data)
                bar.desc = "compute deviation"
        deviation_label_array = np.concatenate(deviation_label_list, axis=0).reshape(-1, 1)
        deviation_array = np.concatenate(deviation_list, axis=0).reshape(-1, 1)

        # Min-Max标准化
        scaler = MinMaxScaler()
        deviation_array = scaler.fit_transform(deviation_array)

        # 四舍五入故障发生概率值
        deviation_array = np.round(deviation_array, decimals=round_digits)

        return deviation_label_array, deviation_array

    def train_twin_model(self, twin_type, model, loss_function, optimizer, train_loader, epoch):
        model.train()
        train_metrics_tracker = RegressionMetricsTracker()
        loss_list = []
        train_bar = tqdm(train_loader)
        for idx, train_tensors in enumerate(train_bar):
            x_train, y_train, mapped_y_train = train_tensors
            x_train, mapped_y_train = x_train.to(self.device), mapped_y_train.to(self.device)
            optimizer.zero_grad()
            x_train_pred, kl = model(x_train)
            loss = loss_function(x_train_pred, x_train, mapped_y_train, twin_type)
            loss.backward()
            optimizer.step()

            x_train = x_train.detach().cpu().numpy()
            x_train_pred = x_train_pred.detach().cpu().numpy()

            train_metrics_tracker.update(real_array=x_train.reshape(x_train.shape[0], -1), pred_array=x_train_pred.reshape(x_train_pred.shape[0], -1))
            loss_list.append(loss.item())
            train_bar.desc = "train epoch[{}/{}] (init:{:4f}) loss:{:.4f}".format(epoch, self.cfg["epochs"], loss_list[0], loss)
        train_metrics_tracker.add_new_metrics("loss", loss_list)
        print(f"[train] {twin_type} metrics: {train_metrics_tracker.get_metrics()}\n")
        return train_metrics_tracker, model

    def test_twin_model(self, twin_type, model, loss_function, test_loader, epoch):
        model.eval()
        test_metrics_tracker = RegressionMetricsTracker()
        loss_list = []
        test_bar = tqdm(test_loader)
        with torch.no_grad():
            for idx, test_tensors in enumerate(test_bar):
                x_test, y_test, mapped_y_test = test_tensors
                x_test, mapped_y_test = x_test.to(self.device), mapped_y_test.to(self.device)
                x_test_pred, kl = model(x_test)
                loss = loss_function(x_test_pred, x_test, mapped_y_test, twin_type)

                x_test = x_test.detach().cpu().numpy()
                x_test_pred = x_test_pred.detach().cpu().numpy()

                test_metrics_tracker.update(real_array=x_test.reshape(x_test.shape[0], -1), pred_array=x_test_pred.reshape(x_test_pred.shape[0], -1))
                loss_list.append(loss.item())
                test_bar.desc = "test epoch[{}/{}] (init:{:4f}) loss:{:.4f}".format(epoch, self.cfg["epochs"], loss_list[0], loss)
        test_metrics_tracker.add_new_metrics("loss", loss_list)
        print(f"[test] {twin_type} metrics: {test_metrics_tracker.get_metrics()}\n")
        return test_metrics_tracker

